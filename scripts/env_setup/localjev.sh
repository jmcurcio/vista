#!/usr/bin/env bash
# Prepare LocalJev, the local Jev-compatible decision server: the pinned
# checkout and its dependencies. Bun comes from the project's mise.toml.
#
#   ./scripts/env_setup/localjev.sh [--dest DIR] [--no-install]
set -euo pipefail
usage() { sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
parse_common_args "$@"

# Pinned upstream checkout (v0.2.0); see docs/decision-layer-requirements.md §3.
LOCALJEV_REVISION=3f23e36e1a3bff46c7e83e8e3781d3512bc82021

clone_pinned https://github.com/githubnext/localjev.git \
  "$DEST/localjev" "$LOCALJEV_REVISION"

if [[ ${SKIP_INSTALL:-0} != 1 ]]; then
  say "installing LocalJev dependencies from its lockfile"
  mise exec --cd "$ROOT" -- bun install --cwd "$DEST/localjev" --frozen-lockfile
fi

cat <<EOF

LocalJev is ready. Configure $DEST/localjev/.env (upstream model and oMLX key),
then start it:

  mise exec --cd $ROOT -- bun run --cwd $DEST/localjev start
EOF
