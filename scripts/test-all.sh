#!/usr/bin/env bash
# Run every repository's test suite from the flat side-by-side checkout.
#
#   work/
#     jwstflow/            <- this script lives in jwstflow/scripts/
#     jwstflow-midas/
#     jwstflow-joys/       (optional: skipped when absent)
#     jwstflow-reducer/    (optional; its venv hosts the shared environment)
#
# One environment is synced in jwstflow-reducer (whose tool.uv.sources point at
# the sibling checkouts, editable), then each repo's pytest runs with it.
# Extra arguments go to pytest, e.g.:  ./test-all.sh -m "not integration" -q
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
repos=(jwstflow jwstflow-midas jwstflow-joys jwstflow-reducer)

env_repo="$root/jwstflow-reducer"
if [[ ! -d "$env_repo" ]]; then
    # no reducer checkout: fall back to an env in jwstflow itself
    env_repo="$root/jwstflow"
fi
echo "==> uv sync in ${env_repo#"$root"/}"
(cd "$env_repo" && uv sync --quiet)
python="$env_repo/.venv/bin/python"

failed=()
for repo in "${repos[@]}"; do
    dir="$root/$repo"
    [[ -d "$dir/tests" ]] || { echo "==> $repo: skipped (no checkout or no tests/)"; continue; }
    echo
    echo "==> $repo"
    if ! (cd "$dir" && "$python" -m pytest "$@"); then
        failed+=("$repo")
    fi
done

echo
if ((${#failed[@]})); then
    echo "FAILED: ${failed[*]}"
    exit 1
fi
echo "all suites passed"
