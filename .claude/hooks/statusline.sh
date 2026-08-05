#!/bin/bash
input=$(cat)
model=$(jq -r '.model.display_name // "claude"' <<<"$input")
dir=$(basename "$(jq -r '.workspace.current_dir // "."' <<<"$input")")
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "-")
gates=$(jq -r '[.[] | select(.status=="refuse")] | length' data/out/gates.json 2>/dev/null || echo "-")
echo "$model | $dir | $branch | refusals: $gates"
