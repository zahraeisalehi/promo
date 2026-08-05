#!/bin/bash
# SessionStart: state of the pipeline, so Claude knows which stages have run.
ctx="Pipeline artifacts present:"
for f in data/interim/*.parquet data/out/*.parquet; do
  [ -e "$f" ] && ctx="${ctx}
  $f ($(stat -c %y "$f" 2>/dev/null | cut -d. -f1))"
done
[ -e data/out/gates.json ] && ctx="${ctx}
Latest gate statuses: $(jq -c '[.[] | {gate, status}]' data/out/gates.json 2>/dev/null)"
ctx="${ctx}
Git branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)
Uncommitted: $(git diff --name-only 2>/dev/null | tr '\n' ' ')"

jq -n --arg c "$ctx" '{hookSpecificOutput: {hookEventName: "SessionStart", additionalContext: $c}}'
