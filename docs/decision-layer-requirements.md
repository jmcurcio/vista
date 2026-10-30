# Requirements: a calibrated decision layer for VISTA

Status: draft for review, 2026-10-18. Implementation status is tracked in §9.
Anything not listed there is not implemented yet unless it is marked
**existing**.

## 1. Purpose

VISTA (Han et al., arXiv 2610.02200) reaches state-of-the-art results by letting a
multimodal model reason at length, inspect its lossless visual memory, and act
through tools. That reasoning is slow and expensive: the paper reports 30.7M
tokens per ARC-AGI-3 game at a 200K context.

This project tests whether a fast, calibrated *decision model* (a "System One"
layer in the style of TypeSafe's Jev) can take over some of that work, reducing
time and cost **without reducing task success**. The decision layer is provided two
ways:

- **Local:** LocalJev (`githubnext/localjev`), a Jev-wire-compatible server backed
  by a model running in oMLX on the same machine.
- **Hosted:** the TypeSafe Jev API.

Both are reached through one client, because LocalJev implements the same wire
protocol (§4.1). The work must hold up to publication-level scrutiny: hypotheses
are fixed before data collection, every run is reproducible from recorded
provenance, and every comparison is paired and reported with uncertainty.

## 2. Hypotheses

The three primary hypotheses are evaluated per benchmark. VISTA is compared with
VISTA plus a decision layer (§5.1, C0 vs C1/C2), using the same main model, task
set, budgets and settings.

| ID | Hypothesis | Endpoint | Test |
|---|---|---|---|
| **H1** | The decision layer reduces time per task. | Wall-clock time per task, excluding thermal pauses (§4.8) | Paired ratio of geometric means, 95% CI below 1 |
| **H2** | The decision layer reduces cost per task. | USD per task, main model plus decider (§4.8) | Paired ratio of geometric means, 95% CI below 1 |
| **H3** | The decision layer does not reduce task success (non-inferiority). | The benchmark's official metric (§5.3) | One-sided 95% CI lower bound of the paired difference above −δ |

Secondary hypotheses:

- **H4:** The decider's confidence is calibrated on VISTA decisions (ECE, Brier; §5.5).
- **H5:** Gains increase as decisions are delegated more aggressively (fast-path
  threshold sweep, §4.6), up to a point where H3 fails. The goal is to locate that
  point.
- **H6:** The local decider (LocalJev) and the hosted decider (Jev API) do not
  differ in task success. This is reported as an estimate with a confidence
  interval, not a claim.

The non-inferiority margins δ are fixed before any experimental run (§5.4). H1–H3
must all be reported, including any that fail.

## 3. Verified background

These facts were checked against code and documentation on 2026-10-18.
Implementation must re-verify them at the pinned versions.

**VISTA (existing).**

- Tools are `ToolSpec`/`ToolBinding` pairs (`core/contracts.py`). Support tools
  such as `inspect` and `read_pixels` are built in `core/tools.py` and bound in
  `core/support.py`. Profiles enable tools by name (`core/profiles.py`).
- Tools reach the model in the same way for every backend: the MCP bridge for
  Claude Code and dynamic tools for Codex.
- ARC-AGI-3 keeps its own tool copies in `benchmarks/arc3/{claude,codex}/tools.py`.
- Each benchmark exposes its legal actions:
  - GameWorld's `act` schema lists `allowed_keys` as an `enum`
    (`benchmarks/gameworld/computer_use.py`).
  - ARC observations carry `available_actions`.
  - BabyVision questions carry `options` (`benchmarks/babyvision/dataset.py`).
- The answer key and the reference solution (`choiceAns`, `blankAns`, `coT`) are
  held host-side and never reach the model.

**LocalJev.** Pinned for this analysis at commit
`3f23e36e1a3bff46c7e83e8e3781d3512bc82021` (2026-09-18), version 0.2.0, MIT licence.

- TypeScript on Bun 1.2+ (the upstream evaluation used Bun 1.4.0). It serves
  `POST /v1/systemone` and `GET /ready` on `127.0.0.1:8080`.
- Request: `{model, state, questions}`.
  - `state` is a string, array or object. Text only; there is no image input.
  - Each question is one of three types:
    - `choice`: 2–128 labelled options.
    - `score`: 2–10 ordered levels.
    - `noul`: a probability that something is true.
- Response: `{model, answers, usage: {input_tokens, output_tokens}}`.
  - A `choice` answer has `choice`, `probabilities` and `confidence`.
  - Confidence is 1 minus the normalized entropy of the probabilities.
- It calls an OpenAI-compatible `/chat/completions` endpoint. Defaults:
  `LOCALJEV_UPSTREAM=http://127.0.0.1:8000`, which is oMLX. Settings are
  `temperature 0`, a per-request seed, `enable_thinking: false`, and a strict
  `json_schema` response format.
- It retries malformed JSON (`LOCALJEV_MALFORMED_RETRIES`, default 2).
- **Probabilities are self-reported by the model in JSON, not read from logits.**
  LocalJev's own documentation calls this "wire-compatible, but not mathematically
  equivalent". Calibration must therefore be measured (H4), not assumed.
- The upstream model is configurable (`LOCALJEV_UPSTREAM_MODEL`; default
  `diffusiongemma-26B-A4B-it-4bit`). LocalJev's 2026-09-18 bake-off on an M5 Max
  reports Gemma 4 26B-A4B and Qwen3.6-35B-A3B as the strongest of five models on
  short text classification, with p50 latency of 0.7–0.9 s.

**Jev API.**

- A hosted service from TypeSafe AI, released in early access on 2026-09-15.
- Typed answers only (choice, score, noul); no free-form text generation.
- No image input documented; no public weights; no fine-tuning.
- Up to 255 choice options. Vendor-reported price is $0.042 per million input
  tokens, with output free.
- The wire format and model aliases (`jev-latest`, `jev-preview`) are matched by
  LocalJev. Exact behaviour must be confirmed against TypeSafe's official
  documentation once API access exists.

**Consequence for VISTA.** Neither decider can see images. VISTA's evidence is
visual, so the main multimodal model must remain the perceiver. The decider
receives a text description of the evidence (§4.4). This is a structural property
of the design and must be stated as a limitation (§6).

## 4. System requirements

"Must" items are required for any reported experiment. "Should" items are required
before submission.

### 4.1 Decider client

- **R1.1** Provide one decider client that speaks the Jev SystemOne wire protocol
  (§3). LocalJev and the Jev API are two *providers* of that client, differing
  only in base URL, credential, model alias and limits. This follows the pattern
  of `ClaudeCodeProvider` in `backends/claude/provider.py`.
- **R1.2** Each provider must declare:
  - name;
  - base URL;
  - credential environment variable;
  - model alias;
  - maximum choice options (LocalJev 128, Jev 255);
  - whether images are accepted (both: no);
  - request timeout.
- **R1.3** Use only `choice` questions for decisions. `score` and `noul` may be
  used for analysis questions, such as "is the evidence sufficient?" (§4.6).
- **R1.4** Fail closed. A malformed response, a timeout, an HTTP 5xx, or a 529
  overload after a bounded number of retries must:
  - be reported to the main model as a failed tool result;
  - never be replaced by a silent default decision;
  - be counted (§4.7).
- **R1.5** Both providers must be callable with the same request bytes, so that a
  single decision can be replayed against either for comparison (§5.6).

### 4.2 Providers

- **R2.1 LocalJev.**
  - Installed by a setup script, `scripts/env_setup/localjev.sh`, following the
    existing benchmark setup scripts.
  - The script clones `githubnext/localjev` into `local/` at a pinned commit
    recorded in code.
  - Bun is pinned to an exact version in the project `mise.toml`.
  - The upstream oMLX model is pinned by id. Its oMLX settings are captured with
    the existing `omlx_settings_snapshot`.
- **R2.2 Jev API.**
  - Base URL and model alias come from configuration. The credential comes from
    `.env`, and its name is listed in `.env.example`.
  - The model version string returned in each response must be recorded. Hosted
    models change without notice.
- **R2.3** A readiness check must run before the first decision of a run:
  `GET /ready` for LocalJev, and one minimal decision for the Jev API. A failed
  check must stop the run before the main model starts. This mirrors the existing
  `READY` preflight.

### 4.3 Option discovery (task-agnostic)

- **R3.1** The options for a decision must come from the task, never from code
  written for one particular game. In order of precedence:
  1. **Enumerated parameters of the task's action tool schema.** `enum` and
     `const` values cover GameWorld keys and ARC actions with no
     benchmark-specific code.
  2. **The task's public answer choices.** BabyVision `options`, read through a
     session accessor that exposes public fields only.
  3. **Candidates proposed by the main model in the `decide` call,** for
     continuous parameters (ARC click coordinates) or open answers (BabyVision
     blank questions).
- **R3.2** Every option must map to exactly one legal tool call or answer, and the
  mapping must be validated before the decision is returned. An option that does
  not map is a failed tool result.
- **R3.3** If the option set exceeds the provider's limit, the call must be
  rejected with an explicit error and never silently truncated. ARC clicks must
  use candidates (R3.1, item 3) or a two-step decision (action, then target).
- **R3.4** Options must never be built from host-only data such as answer keys or
  evaluator state.

### 4.4 Evidence (decider input)

- **R4.1** The decider's `state` is text, assembled from two parts:
  - **The task's question or objective,** without instructions meant only for
    the agent. BabyVision passes the bare question: its choices travel as
    options, and its `\boxed{}` answer-format instruction is omitted.
  - **The main model's `findings`** (prompt `decide-v3`): a list of atomic
    facts, each `{subject, fact}`. The subject is `task` or one option label. The
    findings are sent one per line, grouped task first, then by option.
    Comparing options is the decider's job, so a finding about one option that
    names another ("option N", "candidate N", "choice N", "alternative N") is
    refused with an instruction to rewrite. Words that usually carry a verdict
    ("matches", "aligns", "only", "must", "fits" and others) are flagged per
    finding in the log, not refused, because many are also factual.
  - **Options.** With task choices, the options sent are the task's own text;
    the agent may narrow them but not reword them. Without task choices, the
    agent names options briefly (at most 120 characters), and facts about them
    go in findings.

  The exact assembled `state` string must be recorded.
- **R4.1a (no echo).** The main model's own conclusion must not reach the decider.
  - The `decide` call carries it separately, in two fields that are **recorded
    but never sent**:
    - `hypothesis`: free text;
    - `leaning`: an option label, validated against the options.
  - The tool description and prompt (`decide-v2`) tell the model to keep
    conclusions out of `observations` and to describe options neutrally.
  - Compliance cannot be enforced, so residual leakage is measured. The
    decider's agreement with `leaning` is reported beside its accuracy. A decider
    whose agreement far exceeds its accuracy is echoing the agent.
  - Found in smoke runs on 2026-10-18:
    - Under `decide-v1`, the agent wrote "Preliminary lean toward 2" into its
      evidence, and the decider chose 2.
    - Under `decide-v2`, the agent's free-text observations argued the case
      ("…matching the hole's wide right side… candidate 1 would overlap…"). The
      decider returned that option with confidence 1.0, and the harness advised
      committing to the wrong answer.

    `decide-v3`'s structured findings are the response to both.
- **R4.2** Images are never sent to a provider that declares no image support. If a
  future provider supports images, image inputs must be selected with the `inspect`
  view syntax (event, moment, region). They are then resolved from VISTA's
  lossless memory, and recorded by content hash.
- **R4.3** The size of `state` must be capped, and the cap recorded. A truncated
  state must be marked as truncated in the log.
- **R4.4** No hidden information may reach the decider. It sees nothing the main
  model could not see.

### 4.5 Harness integration

- **R5.1** `decide` is a support tool alongside `inspect` and `read_pixels`:
  - `Effect.NONE`, so it never changes task state;
  - defined in `core/tools.py` and bound in `core/support.py`;
  - registered in the ARC tool copies;
  - enabled per profile by name.
- **R5.2** `decide` has a CLI budget (`--decisions N`), enforced by the decision
  layer itself on every harness path. When the budget is exhausted, `decide` is
  refused and the task continues; it is never ended. Refusals are counted. This
  keeps the semantics identical on the TaskRuntime path and the ARC dispatcher
  path. A `TaskRuntime` meter would instead end the task.
- **R5.3** The prompt change for decide-enabled profiles must be minimal:
  - It consists of one added paragraph describing the tool, in the style of the
    paper's Appendix A.5 prompts.
  - It must be versioned and recorded in the profile manifest.
  - Baseline profiles must stay byte-identical to the existing VISTA profiles.
- **R5.4** The integration must work unchanged with every backend (`claude`,
  `codex`, `omlx`). No backend-specific code is allowed in the decision path.
- **R5.5** Existing behaviour must be unchanged when `decide` is not enabled,
  following the project rule of adding new code beside existing implementations
  rather than modifying them.

### 4.6 Modes

- **R6.1 Adviser mode (phase 1).**
  - The main model calls `decide` when it chooses to.
  - The result (probabilities, choice, confidence) is returned as a tool result.
  - The main model remains responsible for acting.
- **R6.2 Fast-path mode (phase 2).**
  - At each decision point, the harness calls the decider first.
  - If confidence ≥ τ, the harness executes the decided action and tells the main
    model what was executed and why.
  - Otherwise, normal reasoning proceeds.
  - τ must be a recorded experiment parameter.
  - This requires a policy hook at each backend's action boundary (the Claude and
    Codex lifecycles, and the ARC controllers). It must be designed so that a
    harness-executed action is indistinguishable from a model-executed one in the
    environment's records.
- **R6.3** *(Superseded by R6.4, which includes the sufficiency question.)*
- **R6.4 Decision cadence (condition C1-K).** The harness enforces decisions at
  intervals, and each decision says whether to commit or keep gathering evidence.
  - **Gate.** After every K evidence or action calls since the last `decide`, the
    next such call is refused with an instruction to run `decide`.
    - Note tools (GUIDE/WORKING read and write, checkpoint) are exempt.
    - A failed or refused `decide` still resets the count, so a failing decision
      layer cannot deadlock the agent.
    - Each refusal is logged as a `decision_gate` event.
  - **Two questions per call.** Each `decide` asks, in a single request, the
    `choice` question and a `noul` sufficiency question: "Does the reported
    evidence establish exactly one option, with nothing important left
    unverified?"
  - **Advice.** The reply includes advice computed by a fixed rule from
    confidence c, sufficiency s and thresholds τ, σ:
    - *commit* when c ≥ τ and s ≥ σ;
    - *verify* when c ≥ τ and s < σ: check what distinguishes the top two options;
    - *gather* otherwise: gather evidence that distinguishes the top two options;
    - *final* when K or fewer tool calls remain: commit to the leading option.
  - **Relaxation.** τ and σ relax linearly toward a floor of 0.5 as the task's tool
    budget is spent.
  - **Commit guard (`cadence-policy-v2`).** A decision that would advise *commit*
    while any finding is flagged for verdict words is downgraded to *restate*. The
    agent is told which findings and words, and asked to restate them as plain
    observations.
  - **Consistency.** Each decision is asked `--decider-orderings` times (default 3),
    and the answers are averaged.
    - Between askings, only the order of the findings changes (each option's
      findings first in turn). The labels, the options and their listed order
      stay fixed.
    - Confidence is recomputed from the averaged probabilities on LocalJev's
      entropy scale, and sufficiency is the mean. Every sample is logged.
    - The design is the result of two failed alternatives, found while replaying
      recorded decisions on 2026-10-18:
      1. **Reordering the listed options** produced spurious disagreement.
         LocalJev's model reports probabilities in label order whatever the
         listing order, so answers were read against the wrong options.
      2. **Reassigning options to labels** contradicted tasks whose option names
         carry meaning: BabyVision's "patterns numbered 1, 2, 3 left to right".
         The decider then hedged uniformly even on clear evidence.

      Findings-order rotation has neither problem. Clear evidence stays
      unanimous at confidence 1.0, and noisy evidence loses confidence.
  - **Logging.** The verdict, the advice, the effective thresholds and the budget
    state are logged with every decision.
  - **Pre-registration.** K, τ and σ are experiment parameters. They must be tuned
    on a development set disjoint from the evaluation set, then fixed in
    pre-registration (§5.4).
    - **BabyVision development set:** the 44 Visual Tracking questions outside
      the paper's 39 (Lines Observation 9, Metro map 12, Recognize numbers and
      letters 23). These are the same task type, disjoint by construction.
      Select them with `--task-set visual-tracking --subtype "Lines Observation"
      --subtype "Metro map" --subtype "Recognize numbers and letters"`.
    - **Confidence scale.** `confidence` is 1 minus normalized entropy, not the
      top probability. With three options, a top probability of 0.8 gives about
      0.42, so τ must be tuned on this scale.
    - Smoke runs on question 457 (2D Pattern Completion) are outside both sets.
      They count as development data and are never reported as results.
  - **Random control.** The random control (N1) under the cadence has uniform
    probabilities and sufficiency 0.5, so it never triggers *commit* before the
    budget-relaxed thresholds reach the floor. It isolates the effect of forced
    summarizing from the effect of the decisions themselves.

### 4.7 Provenance and logging

- **R7.1** Every decision is logged in the run's `events.jsonl` with:
  - call id;
  - provider and model alias;
  - the returned model version;
  - the LocalJev commit, or the API base URL;
  - the request hash and full `state`;
  - the options and their mapping to actions;
  - the answers;
  - latency;
  - token usage;
  - retries;
  - the outcome (used, overridden, or failed).
- **R7.2** The run manifest records the full decider configuration. That covers
  the provider declaration, mode, τ, the `decisions` budget and the state cap,
  plus the upstream oMLX model settings snapshot when LocalJev is used.
- **R7.3** Experiments must run from a clean, tagged commit. **As of 2026-10-18,
  the oMLX backend work is uncommitted.** It must be committed before any
  experimental run. The existing `source_fingerprint` stays in place as a
  secondary check.

### 4.8 Time and cost accounting

- **R8.1** For each task, record wall-clock time split into three components:
  - main-model inference;
  - decider inference;
  - harness and tools.

  Exclude thermal-gate pauses (existing `thermal.jsonl`), and report them
  separately.
- **R8.2** Record tokens per component: input, cached input and output for the
  main model; input and output for the decider.
- **R8.3** Compute USD cost from a versioned price table in the repository, giving
  model, input, cached-input and output prices, the source URL, and the date
  retrieved.
  - Subscription CLIs (Claude Code, Codex) report token usage, not marginal cost.
    Their cost is computed at the published API rates and **labelled as an
    estimate**.
  - Local inference is reported as time, with no dollar figure. Energy may be
    reported if it is measured, for example with `powermetrics`, which requires
    sudo.
- **R8.4** Time comparisons must use the same hardware state for every condition:
  - power source;
  - Low Power Mode;
  - the oMLX memory ceiling;
  - the thinking budget.

  Conditions must be interleaved in randomized order, not run in blocks (§5.2).

### 4.9 Reproducibility

- **R9.1** Pin and record everything that can change results:
  - VISTA commit;
  - LocalJev commit;
  - Bun version;
  - oMLX version and settings;
  - model ids;
  - Claude Code or Codex version;
  - benchmark revisions (existing pins);
  - the price table version.
- **R9.2** Provide scripts that regenerate every reported table and figure from
  raw run directories alone.
- **R9.3** Archive raw run directories with the paper artifacts, excluding
  credentials (the existing secret filter applies).
- **R9.4** Respect licences:
  - LocalJev: MIT.
  - Benchmarks: their own terms.
  - ARC-AGI-3: requires API terms acceptance.
  - Jev API: TypeSafe terms.

## 5. Experimental design

### 5.1 Conditions

| ID | Harness | Decider | Purpose |
|---|---|---|---|
| **C0** | VISTA (unchanged) | none | Baseline |
| **C1** | VISTA + `decide`, adviser | LocalJev | H1–H3, local |
| **C2** | VISTA + `decide`, adviser | Jev API | H1–H3, hosted; H6 |
| **C1-K** | VISTA + `decide`, cadence (R6.4), K from tuning | LocalJev | H1–H3 with enforced decisions |
| **N1-K** | VISTA + `decide`, cadence | random decider | Control for C1-K: forced summarizing versus decision quality |
| **C3(τ)** | VISTA + fast path, τ ∈ {0.6, 0.8, 0.9} | best of C1/C2 | H5 |
| **N1** | VISTA + `decide`, adviser | **random decider** (uniform probabilities) | Control: is any gain from *calibrated* decisions, or from the prompt and tool alone? |
| **N2** | VISTA + `decide` described, but always unavailable | none | Control: does the added prompt text alone change behaviour? |

Main models, run as separate experiment families:

- **M-frontier (paper parity):** Claude Opus 5.0 at xhigh effort through Claude
  Code, as in Table 1. On macOS this requires the host-process launch path
  (existing in the `omlx` backend) with the native Anthropic provider, or a Linux
  host with the original Docker path.
- **M-local:** Qwen3.8-27B through oMLX. This family tests the cheap end of the
  hypothesis.

### 5.2 Tasks, repetitions, order

| Benchmark | Task set | Budget (as in the paper) | Runs per condition |
|---|---|---|---|
| BabyVision | the paper's 39 (20 Maze, 19 Connect the lines) | tool calls capped equally across conditions (value fixed in pre-registration) | 3 |
| ARC-AGI-3 | the 25 public games, online scorecard | 2,000 actions per game | 3 |
| GameWorld | 170 tasks | 100 actions per task | 3 (phase 2) |
| AI GameStore | 10 games | 120 decisions | 3 (phase 2) |

- Three runs per condition match the paper's run-to-run protocol (Appendix B.1).
- Every condition runs on every task (paired design).
- The run order is randomized per task and recorded.
- BabyVision is the pilot and first full experiment. ARC-AGI-3 is the headline
  experiment.

### 5.3 Effectiveness metrics (official evaluators only)

| Benchmark | Metric |
|---|---|
| BabyVision | Accuracy. Exact match is primary; the official LLM-judge score uses one fixed judge model, recorded. |
| ARC-AGI-3 | RHAE from the ARC scorecard, plus total actions |
| GameWorld | Success rate and progress |
| AI GameStore | Geometric mean of human-normalized scores |

### 5.4 Non-inferiority margins (to be fixed before data collection)

δ is set per benchmark from the paper's reported run-to-run standard deviation of
VISTA itself (Appendix B.1). A difference smaller than VISTA's own noise is not
treated as a loss.

| Benchmark | Proposed δ | Basis |
|---|---|---|
| BabyVision | 3.0 percentage points | SD over 3 runs: ±3.00 |
| ARC-AGI-3 | 2.0 RHAE | SD ±0.17 at RHAE 99; widened, because local models will sit far from the ceiling |
| GameWorld | 2.4 percentage points (SR) | SD ±2.37 |
| AI GameStore | 15 points | spread of the 3 reported aggregates (120–149) |

These values are proposals. The final values, with written justification, must be
committed (pre-registration) before the first C0 run.

### 5.5 Analysis

- **H1/H2:** the paired ratio of geometric means of time and cost per task, with a
  95% CI from 10,000 paired bootstrap resamples over tasks. Runs are averaged
  within a task first.
- **H3:** the paired difference in the benchmark metric, as a one-sided 95%
  bootstrap CI. Non-inferior if the lower bound is above −δ.
- **Multiplicity:** Holm correction across benchmarks within each hypothesis.
- **Echo:** the decider's agreement with the agent's logged `leaning`, overall and
  when the leaning is wrong (R4.1a). It is reported for every decide condition.
- **H4:**
  - Expected calibration error (15 equal-mass bins), Brier score and reliability
    diagrams on BabyVision decisions, where the correct option is known.
  - For games, report agreement between the decider's choice and the main model's
    eventual action, labelled as agreement, not correctness.
- **Report:**
  - all hypotheses, all conditions and all failed runs;
  - infrastructure failures, which are re-run and reported, kept separate from
    task failures, which are counted as failures;
  - effect sizes with CIs, not only significance;
  - per-task tables in an appendix.
- **Power:** 39 BabyVision questions can detect only large accuracy differences.
  The design states this, and treats BabyVision as a pilot for effect size. ARC (25
  games × 3 runs) carries the main claim.

### 5.6 Decider-only evaluations

These isolate the decider from the harness:

- Replay recorded `decide` requests from C1/C2 runs against both providers
  (R1.5). This measures agreement between the providers and the calibration of
  each, at zero main-model cost.
- Run LocalJev's own evaluation (`bun run eval`) with the chosen upstream model,
  to anchor its calibration on standard text benchmarks.

## 6. Threats to validity (to be addressed in the paper)

1. **Text-only deciders.** The decider judges the main model's *description* of
   visual evidence, so its accuracy is bounded by the perceiver. This is
   structural, and reported as such.
2. **Self-reported probabilities (LocalJev).** They are not logits, and
   calibration may not transfer to VISTA decisions. This is measured by H4.
3. **Hosted model drift (Jev, frontier models).** Record returned model versions,
   and keep each experiment family's runs within a short time window.
4. **Subscription CLIs.** Cost is estimated from list prices, and caching
   behaviour may differ from the raw API.
5. **Local hardware.** Thermal throttling, memory pressure and cache state affect
   time. This is mitigated by R8.4, randomized order and the thermal log.
6. **Contamination.** As the original paper notes (§6), the models may have seen
   the public benchmarks. This applies equally to all conditions, so paired
   comparisons remain valid, but absolute scores carry the caveat.
7. **Small BabyVision sample** (§5.5, power).
8. **Prompt confound.** Adding a tool changes the prompt. Controlled by N1 and N2.

## 7. Phased plan and exit criteria

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **P0: environment** | Bun pinned via mise; LocalJev cloned at its pin; decider model in oMLX; memory ceiling sized for two models; oMLX work committed (R7.3) | `GET /ready` returns ready; one manual `choice` decision returns valid probabilities |
| **P1: LocalJev pilot** | 20 recorded VISTA-style decisions from BabyVision text descriptions | Valid responses for all 20; latency and calibration recorded |
| **P2: adviser integration** | Decider client and LocalJev provider; `decide` tool (core and ARC copies); option discovery; logging; budget; profiles C1, N1, N2 | One full BabyVision question and one ARC game complete under C1, with every decision logged per R7.1 |
| **P3: Jev API provider** | Jev provider; replay tool (R1.5, §5.6) | The same recorded requests replay on both providers |
| **P4: fast path** | Policy hook; τ parameter; C3 | One ARC game completes under C3 with harness-executed actions logged |
| **P5: pre-registration and experiments** | Committed δ values, task lists, budgets, analysis code; then runs | All planned runs complete or failures documented |
| **P6: paper artifacts** | Table and figure scripts, archived runs, threats-to-validity text | Every number in the paper regenerates from archived runs |

## 8. Open decisions

1. **Decider model for LocalJev.** Options:
   - Qwen3.8-27B: already loaded, adds no memory, but is not a fast System One model.
   - Gemma 4 26B-A4B: LocalJev's strongest default candidate; about 4B active
     parameters, so fast; needs installing.
   - DiffusionGemma 26B-A4B: LocalJev's default.
2. **Memory.** Two resident models need the oMLX ceiling raised. On this Mac the
   macOS GPU cap is 51.8 GB.
3. **Frontier main model.** Paper parity needs Claude Opus 5.0 (Claude Code
   subscription) or GPT-5.6 Sol (Codex), at their usage cost.
4. **Jev API access** and key.
5. **Final δ values and the BabyVision tool-call cap**, fixed in pre-registration.

## 9. Implementation status (2026-10-18)

| Requirement | Status |
|---|---|
| P0 environment | **Done.** Bun 1.4.0 pinned (`mise.toml`); LocalJev at `3f23e36` via `scripts/env_setup/localjev.sh`; upstream model `gemma-4-26b-a4b-it-UD-MLX-4bit`; oMLX ceiling 44 GB with both models resident. |
| R1, R2 (client, providers) | **Done**: `src/vista/decide/client.py`. Providers: `localjev`, `jev` (needs `--decider-url` and `JEV_API_KEY`; not yet run live), `random`, `unavailable`. The readiness check records the LocalJev checkout, upstream model and oMLX settings. |
| R3 (options) | **Partial.** Public task choices (BabyVision) and options supplied by the agent are implemented. Enumerating options from action tool schemas, and validating that agent options are legal actions, are **not** implemented; R3.2 matters only for the fast path (R6.2). |
| R4, R5 (evidence, tool) | **Done**, with prompt `decide-v3`: only structured findings reach the decider, and cross-option references are refused; `hypothesis` and `leaning` are logged, never sent (R4.1a); BabyVision sends its bare question. Code: `src/vista/decide/integration.py`. `DecidingSession` (general path) and `DecisionGate` (also answers `decide` on the ARC path). Hooks: `backends/binding.py` (gate) and the `run.py`/`cli.py` of each benchmark; ARC in `benchmarks/arc3/omlx/harness.py`. |
| R6.1 adviser, R6.4 cadence | **Implemented**, with `cadence-policy-v2` (commit guard) and 3 findings orderings averaged per decision. A replay of the recorded `decide-cadence-v3` run on 457 shows the wrong confidence-1.0 commit becoming *restate*. Verified in-process with the random provider and live LocalJev decisions. **No complete benchmark run yet.** |
| R6.2 fast path | Not started. |
| R7 (provenance) | Decisions and gate refusals are logged per event. The decider configuration is in the experiment manifest and changes the experiment identity. The oMLX backend is committed (`omlx-backend`); the decision layer is **not yet committed**. |
| R8, R9.2 (accounting, analysis scripts) | Not started. Per-task decider summaries (calls, failures, refusals, latency, tokens) are written to `result.json`. |
