#!/bin/bash
# PostToolUse: lint the edited file, and run the invariant tests when core maths changes.
input=$(cat)
path=$(jq -r '.tool_input.file_path // ""' <<<"$input")
[[ "$path" == *.py ]] || exit 0

out=""
if command -v ruff >/dev/null 2>&1; then
  out=$(ruff check --quiet "$path" 2>&1)
fi

case "$path" in
  */promo/baseline.py|*/promo/transfer.py|*/promo/validate.py|*/promo/prices.py)
    tests=$(pytest -q tests/test_invariants.py 2>&1 | tail -5)
    out="${out}
Invariant tests:
${tests}"
    ;;
esac

if [[ -n "${out// /}" ]]; then
  jq -n --arg ctx "$out" '{
    hookSpecificOutput: {
      hookEventName: "PostToolUse",
      additionalContext: $ctx
    }
  }'
fi
exit 0
