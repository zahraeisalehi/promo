#!/bin/bash
# PreToolUse: raw data is immutable evidence. Block any write into it.
input=$(cat)
path=$(jq -r '.tool_input.file_path // .tool_input.notebook_path // ""' <<<"$input")

case "$path" in
  */data/raw/*)
    jq -n '{
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "deny",
        permissionDecisionReason: "data/raw is immutable. Write derived output to data/interim/ instead."
      }
    }'
    exit 0
    ;;
esac
exit 0
