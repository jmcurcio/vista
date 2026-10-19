"""The `decide` tool: a decision layer beside VISTA's inspection tools.

A `DecidingSession` wraps any benchmark's task session and adds one tool,
`decide`, which asks a decision model to choose among options given the
agent's written evidence. The tool never changes task state: the agent still
acts through its own tools. Options come from the task's public answer
choices when it has them, otherwise from the agent's call.

Two modes:

- **adviser**: the agent calls `decide` when it chooses to.
- **cadence**: a `DecisionGate` refuses evidence and action calls after every
  K of them until the agent runs `decide`. Each decision also asks whether the
  evidence is sufficient, and the reply advises the agent to commit, or to
  gather the evidence that separates the leading options. The thresholds relax
  as the tool budget is spent, and the last decision is a forced commitment.

The decision model is text-only, so it judges the agent's account of the
evidence, not the images. Only the agent's `findings` reach it: atomic facts,
each about one subject (the task, or a single option), so that comparing
options is the decider's work and not the agent's. A finding that refers to
another option is refused; verdict-like words are flagged in the log. The
agent's `hypothesis` and `leaning` are recorded but never sent, so that the
decider cannot simply echo the agent, and so that agreement with the agent
can be measured separately from correctness. Every call is recorded as a
`decision` event, and every refusal as a `decision_gate` event.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Any

from vista.core.contracts import (
    Capability,
    Effect,
    ToolBinding,
    ToolReply,
    ToolSpec,
)
from vista.core.json import canonical, thaw
from vista.core.profiles import AgentProfile, Experiment

from .client import PROVIDERS, DeciderError, DeciderProvider, decide, ready

# One paragraph, in the register of the paper's Appendix A.5 prompts.
DECIDE_METHOD = (
    "A `decide` tool returns a fast, calibrated choice among options, with "
    "probabilities and a confidence, from your findings. It cannot see images "
    "and does not see your hypothesis. Report each finding as one fact about "
    "one subject, the task itself or a single option: a count, position, "
    "colour, size or shape. Do not compare options or judge them; the "
    "decision model compares. Treat its answer as advice; you remain "
    "responsible for what you do."
)
CADENCE_METHOD = (
    "The harness requires `decide` after every {every} evidence or action "
    "calls. Follow its advice: commit when it says the evidence supports one "
    "option; otherwise gather the evidence it asks for, then decide again."
)
PROMPT_VERSIONS = {"adviser": "decide-v3", "cadence": "decide-cadence-v3"}
SUFFICIENCY_QUESTION = (
    "Do the reported findings establish exactly one option, with nothing "
    "important left unverified?"
)
MAX_FINDINGS = 40
FINDING_CHARS = 300
# A finding naming another option compares instead of reporting: refused.
OPTION_REFERENCE = re.compile(
    r"\b(?:option|candidate|choice|alternative)\s*#?\s*([A-Za-z0-9]+)\b", re.I
)
# Words that usually carry a verdict rather than an observation: flagged in
# the log for leakage analysis, never refused (many are also factual).
VERDICT_WORDS = re.compile(
    r"\b(?:correct|incorrect|wrong|answer|fits?|fill(?:s|ed|ing)?|"
    r"match(?:es|ed|ing)?|align(?:s|ed|ing)?|consistent|inconsistent|must|"
    r"should|therefore|thus|hence|best|likely|unlikely|clearly|only|lacks?|"
    r"needed|overlap(?:s|ping)?)\b",
    re.I,
)
# Tools that only read or write the agent's own notes never count toward K.
NOTE_TOOLS = frozenset(
    {
        "read_guide",
        "write_guide",
        "read_working",
        "write_working",
        "save_compact_checkpoint",
    }
)


@dataclass(frozen=True)
class Cadence:
    """When decisions are required and when the agent is told to commit.

    `every` evidence or action calls are allowed between decisions. Commit
    advice needs confidence >= `commit_confidence` and sufficiency >=
    `commit_sufficiency`; both relax linearly toward `floor` as the task's
    tool budget is spent. With `every` or fewer calls left, the advice is a
    forced commitment to the leading option.
    """

    every: int
    commit_confidence: float = 0.8
    commit_sufficiency: float = 0.7
    floor: float = 0.5


@dataclass(frozen=True)
class Decider:
    provider: DeciderProvider
    decisions: int | None = None
    cadence: Cadence | None = None

    @property
    def mode(self) -> str:
        return "cadence" if self.cadence else "adviser"

    def manifest(self) -> dict[str, Any]:
        return {
            **self.provider.manifest(),
            "mode": self.mode,
            "cadence": asdict(self.cadence) if self.cadence else None,
            "decisions_budget": self.decisions,
            "prompt_version": PROMPT_VERSIONS[self.mode],
            "sufficiency_question": SUFFICIENCY_QUESTION if self.cadence else None,
            "max_findings": MAX_FINDINGS,
            "finding_chars": FINDING_CHARS,
        }

    def method(self) -> str:
        if self.cadence is None:
            return DECIDE_METHOD
        cadence = CADENCE_METHOD.format(every=self.cadence.every)
        return f"{DECIDE_METHOD} {cadence}"


def decide_spec(max_options: int) -> ToolSpec:
    return ToolSpec(
        "decide",
        "Ask a decision model to choose among options from your findings. It "
        "cannot see images; only `question`, the options and `findings` reach "
        "it. Each finding is one fact about one subject: 'task' (the figure, "
        "scene or game state) or a single option label. A finding about one "
        "option must not mention another. Returns the chosen option, a "
        "probability for every option, a confidence from 0 to 1, and any "
        "advice from the harness. Omit options to use the task's own answer "
        "choices.",
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["question", "findings"],
            "properties": {
                "question": {"type": "string", "minLength": 1, "maxLength": 1024},
                "findings": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_FINDINGS,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["subject", "fact"],
                        "properties": {
                            "subject": {
                                "type": "string",
                                "minLength": 1,
                                "description": "'task', or one option label.",
                            },
                            "fact": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": FINDING_CHARS,
                                "description": "One observed fact about that "
                                "subject alone: a count, position, colour, size "
                                "or shape. No comparison, no verdict.",
                            },
                        },
                    },
                },
                "hypothesis": {
                    "type": "string",
                    "maxLength": 2048,
                    "description": "Your current conclusion and why. Recorded "
                    "for analysis; never sent to the decision model.",
                },
                "leaning": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The option label you currently favour, if "
                    "any. Recorded; never sent.",
                },
                "options": {
                    "type": "object",
                    "minProperties": 2,
                    "maxProperties": max_options,
                    "additionalProperties": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 120,
                    },
                    "description": "For tasks without their own choices: option "
                    "label to a short neutral name (put facts in findings). With "
                    "the task's choices, a subset narrows them; their text is "
                    "the task's.",
                },
            },
        },
        # Reads evidence and changes nothing, like inspect.
        Capability.INSPECTION,
        Effect.NONE,
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )


class DecisionLayer:
    """One task's decide handler, shared by every harness path.

    `record(event, value)` stores events with the run's others. `budget()`
    returns (tool calls used, tool call limit), or None where the task has
    no tool budget; cadence advice relaxes with it.
    """

    def __init__(
        self,
        decider: Decider,
        objective: str,
        *,
        options: dict[str, str] | None = None,
        record: Callable[[str, dict[str, Any]], None],
        budget: Callable[[], tuple[int, int] | None] = lambda: None,
    ) -> None:
        self.decider = decider
        self.objective = objective
        self.options = options
        self.record = record
        self.budget = budget
        self.summary = {
            "provider": decider.provider.name,
            "mode": decider.mode,
            "calls": 0,
            "failures": 0,
            "refusals": 0,
            "latency_seconds": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    def __call__(self, arguments: dict[str, Any]) -> tuple[bool, str]:
        """(success, the text the agent receives)."""
        options = arguments.get("options") or self.options
        if self.options and arguments.get("options"):
            # The task's own text, never the agent's rewording of it.
            options = {
                label: self.options[label]
                for label in arguments["options"]
                if label in self.options
            }
        entry: dict[str, Any] = {
            "provider": self.decider.provider.manifest(),
            "question": arguments["question"],
            "options": options,
            "option_source": "agent" if arguments.get("options") else "task",
            # The agent's own view: recorded, never sent to the decider.
            "hypothesis": arguments.get("hypothesis"),
            "leaning": arguments.get("leaning"),
        }
        budget = self.decider.decisions
        if budget is not None and self.summary["calls"] >= budget:
            # Refused, not fatal: the task continues without the decision layer.
            self.summary["refusals"] += 1
            self.record("decision", {**entry, "outcome": "budget_exhausted"})
            return False, "The decision budget is exhausted; continue without decide."
        self.summary["calls"] += 1
        try:
            if not options:
                raise DeciderError(
                    "This task has no answer choices; supply options in the call."
                )
            if self.options and arguments.get("options"):
                if not set(arguments["options"]) <= set(self.options):
                    raise DeciderError(
                        f"Options must be among the task's choices: {sorted(self.options)}"
                    )
                if len(options) < 2:
                    raise DeciderError("Narrow the task's choices to at least two.")
            if entry["leaning"] is not None and entry["leaning"] not in options:
                raise DeciderError(
                    f"leaning must be one of the options: {sorted(options)}"
                )
            entry["findings"], entry["flags"] = self._check(
                arguments["findings"], options
            )
            entry["state"] = "\n".join(
                [
                    f"Task:\n{self.objective}",
                    "",
                    "Findings reported by the agent, one fact per line:",
                    *(
                        f"- {'task' if f['subject'] == 'task' else 'option ' + f['subject']}: {f['fact']}"
                        for f in entry["findings"]
                    ),
                ]
            )
            decision = decide(
                self.decider.provider,
                state=entry["state"],
                instructions=arguments["question"],
                options=options,
                sufficiency=SUFFICIENCY_QUESTION if self.decider.cadence else None,
            )
        except DeciderError as exc:
            self.summary["failures"] += 1
            self.record("decision", {**entry, "outcome": "failed", "error": str(exc)})
            return False, f"The decision layer failed: {exc}"
        usage = decision["usage"]
        self.summary["latency_seconds"] += decision["latency_seconds"]
        self.summary["input_tokens"] += int(usage.get("input_tokens", 0))
        self.summary["output_tokens"] += int(usage.get("output_tokens", 0))
        reply = {
            key: decision[key] for key in ("choice", "probabilities", "confidence")
        }
        if self.decider.cadence:
            reply["sufficient"] = decision["sufficient"]
            entry["policy"] = self.advise(decision)
            reply["advice"] = entry["policy"]["advice"]
        self.record("decision", {**entry, "outcome": "returned", **decision})
        return True, canonical(reply)

    @staticmethod
    def _check(
        findings: list[dict[str, str]], options: dict[str, str]
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        """Findings grouped task first, then by option; refuse cross-references.

        Returns the ordered findings and, per finding, any verdict words.
        """
        order = {"task": 0, **{label: i + 1 for i, label in enumerate(options)}}
        flags = []
        for index, finding in enumerate(findings):
            subject = finding["subject"]
            if subject not in order:
                raise DeciderError(
                    f"Finding {index}: subject must be 'task' or one of {sorted(options)}"
                )
            own = (
                set()
                if subject == "task"
                else {subject.lower(), options[subject].lower()}
            )
            for match in OPTION_REFERENCE.finditer(finding["fact"]):
                if match.group(1).lower() not in own:
                    raise DeciderError(
                        f"Finding {index} about {subject} mentions "
                        f"'{match.group(0)}'. Report each subject's facts on "
                        "their own; the decision model compares them."
                    )
            words = sorted({w.lower() for w in VERDICT_WORDS.findall(finding["fact"])})
            if words:
                flags.append({"finding": index, "words": words})
        ordered = sorted(findings, key=lambda f: order[f["subject"]])
        return ordered, flags

    def advise(self, decision: dict[str, Any]) -> dict[str, Any]:
        cadence = self.decider.cadence
        assert cadence is not None
        spent = self.budget()
        progress = min(1.0, spent[0] / spent[1]) if spent and spent[1] else 0.0
        tau = (
            cadence.commit_confidence
            - (cadence.commit_confidence - cadence.floor) * progress
        )
        sigma = (
            cadence.commit_sufficiency
            - (cadence.commit_sufficiency - cadence.floor) * progress
        )
        ranked = sorted(
            decision["probabilities"], key=decision["probabilities"].get, reverse=True
        )
        top, runner = ranked[0], ranked[1]
        if spent and spent[1] - spent[0] <= cadence.every:
            verdict = "final"
            advice = (
                f"Final decision: the tool budget is nearly spent. Commit to {top} now."
            )
        elif decision["confidence"] >= tau and decision["sufficient"] >= sigma:
            verdict = "commit"
            advice = f"The evidence supports {top}. Commit to it now."
        elif decision["confidence"] >= tau:
            verdict = "verify"
            advice = (
                f"Leaning {top}, but the evidence may be insufficient. Check what "
                f"distinguishes {top} from {runner}, then decide again."
            )
        else:
            verdict = "gather"
            advice = (
                f"Undecided between {top} and {runner}. Gather evidence that "
                "distinguishes them, then decide again."
            )
        return {
            "verdict": verdict,
            "advice": advice,
            "confidence_threshold": tau,
            "sufficiency_threshold": sigma,
            "budget": list(spent) if spent else None,
        }


class DecisionGate:
    """A tool dispatcher enforcing the decision cadence.

    It counts evidence and action calls (everything but `decide` and note
    tools) and refuses the next one once `cadence.every` have run since the
    last `decide`; a failed `decide` still resets the count, so a failing
    decision layer cannot deadlock the agent. With `answers_decide`, the gate
    also answers `decide` itself, for harnesses that dispatch tools directly
    (ARC-AGI-3) instead of through a TaskRuntime.
    """

    def __init__(
        self,
        dispatcher: Any,
        layer: DecisionLayer,
        *,
        answers_decide: bool = False,
    ) -> None:
        self._dispatcher = dispatcher
        self.layer = layer
        self.answers_decide = answers_decide
        self.since_decision = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dispatcher, name)

    def execute(self, tool: str, arguments: Any, call_id: str) -> Any:
        from vista.core.dispatcher import ToolExecution

        if tool == "decide":
            self.since_decision = 0
            if not self.answers_decide:
                return self._dispatcher.execute(tool, arguments, call_id)
            if (
                not isinstance(arguments, dict)
                or not all(
                    isinstance(arguments.get(key), str) and arguments[key].strip()
                    for key in ("question",)
                )
                or not isinstance(arguments.get("findings"), list)
                or not all(
                    isinstance(f, dict)
                    and isinstance(f.get("subject"), str)
                    and isinstance(f.get("fact"), str)
                    and f["fact"].strip()
                    for f in arguments["findings"]
                )
            ):
                return ToolExecution(
                    "decide needs a question and findings of subject and fact.",
                    False,
                )
            success, text = self.layer(arguments)
            return ToolExecution(text, success)
        cadence = self.layer.decider.cadence
        if tool not in NOTE_TOOLS and cadence is not None:
            if self.since_decision >= cadence.every:
                self.layer.summary["refusals"] += 1
                self.layer.record(
                    "decision_gate",
                    {"tool": tool, "call_id": call_id, "since": self.since_decision},
                )
                return ToolExecution(
                    f"Run `decide` with your current evidence before another "
                    f"{tool} call: a decision is required after every "
                    f"{cadence.every} evidence or action calls.",
                    False,
                )
            self.since_decision += 1
        return self._dispatcher.execute(tool, arguments, call_id)


class DecidingSession:
    """A task session plus the `decide` tool; everything else is delegated."""

    def __init__(
        self,
        session: Any,
        decider: Decider,
        *,
        options: dict[str, str] | None = None,
        objective: str | None = None,
    ) -> None:
        self._session = session
        self.runtime: Any = None
        self.layer = DecisionLayer(
            decider,
            objective or session.task.objective,
            options=options,
            record=lambda event, value: self.runtime.artifacts.record(event, value),
            budget=self._tool_budget,
        )
        self.tools = (
            *session.tools,
            ToolBinding(decide_spec(decider.provider.max_options), self._decide),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def gate(self, dispatcher: Any) -> Any:
        """The dispatcher the backend should use (see backends/binding.py)."""
        if self.layer.decider.cadence is None:
            return dispatcher
        return DecisionGate(dispatcher, self.layer)

    def _tool_budget(self) -> tuple[int, int] | None:
        limit = self.runtime.experiment.budgets.get("tool_calls")
        if limit is None:
            return None
        return self.runtime.usage.get("tool_calls", 0), limit

    def _decide(self, arguments: dict[str, Any]) -> ToolReply:
        if self.runtime is None:
            raise RuntimeError("The deciding session was not attached to its runtime")
        success, text = self.layer(arguments)
        return ToolReply(text, success=success)


def with_decider(experiment: Experiment, decider: Decider) -> Experiment:
    """The same experiment with `decide` enabled; identity changes with it."""
    profile = experiment.profile
    prompts = replace(
        profile.prompts, system=f"{profile.prompts.system}\n\n{decider.method()}"
    )
    suffix = f"+decide-{decider.provider.name}"
    if decider.cadence:
        suffix += f"-every{decider.cadence.every}"
    return replace(
        experiment,
        profile=AgentProfile(
            f"{profile.profile_id}{suffix}",
            (*profile.tools, "decide"),
            prompts,
            profile.policy,
        ),
        presentation={
            **thaw(experiment.presentation),
            "decider": decider.manifest(),
        },
    )


def enable(
    session: Any,
    experiment: Experiment,
    decider: Decider | None,
    *,
    options: dict[str, str] | None = None,
    objective: str | None = None,
) -> tuple[Any, Experiment]:
    """The session and experiment a benchmark should run, with or without
    the decision layer. Attach the runtime afterwards with `attach`.

    `objective` is what the decider is told the task is; it defaults to the
    task's objective. A benchmark whose objective carries instructions meant
    only for the agent (an answer format) passes the bare question instead.
    """
    if decider is None:
        return session, experiment
    return (
        DecidingSession(session, decider, options=options, objective=objective),
        with_decider(experiment, decider),
    )


def attach(session: Any, runtime: Any) -> dict[str, Any] | None:
    """Give a deciding session its runtime; returns its running summary."""
    if not isinstance(session, DecidingSession):
        return None
    session.runtime = runtime
    return session.layer.summary


def add_arguments(parser: Any) -> None:
    group = parser.add_argument_group("decision layer")
    group.add_argument("--decider", choices=tuple(PROVIDERS))
    group.add_argument("--decider-url", help="Provider base URL (required for jev)")
    group.add_argument("--decider-model", help="Provider model alias")
    group.add_argument("--decisions", type=int, help="Budget of decide calls")
    group.add_argument(
        "--decide-every",
        type=int,
        help="Cadence mode: require decide after every N evidence or action calls",
    )
    group.add_argument(
        "--commit-confidence", type=float, default=Cadence.commit_confidence
    )
    group.add_argument(
        "--commit-sufficiency", type=float, default=Cadence.commit_sufficiency
    )


def resolve(args: Any, error: Any) -> Decider | None:
    """The configured decision layer, checked ready; None when none is used."""
    if not args.decider:
        if (
            args.decider_url
            or args.decider_model
            or args.decisions is not None
            or args.decide_every is not None
        ):
            error("Decision-layer options need --decider")
        return None
    from dotenv import load_dotenv

    load_dotenv(os.getenv("ARC3_ENV_FILE", ".env"))
    if args.decisions is not None and args.decisions < 1:
        error("--decisions must be positive")
    cadence = None
    if args.decide_every is not None:
        if args.decide_every < 1:
            error("--decide-every must be positive")
        if not (
            Cadence.floor < args.commit_confidence <= 1
            and Cadence.floor < args.commit_sufficiency <= 1
        ):
            error(f"Commit thresholds must be in ({Cadence.floor}, 1]")
        cadence = Cadence(
            args.decide_every, args.commit_confidence, args.commit_sufficiency
        )
    provider = PROVIDERS[args.decider]
    provider = replace(
        provider,
        base_url=args.decider_url or provider.base_url,
        model=args.decider_model or provider.model,
    )
    try:
        provider = ready(provider)
    except (DeciderError, OSError) as exc:
        error(f"The {provider.name} decision layer is not ready: {exc}")
    return Decider(provider, args.decisions, cadence)
