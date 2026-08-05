# Promotional Intelligence — setup

Everything here is already in the right place. Run the commands below from the folder that contains this README.

## 1. System packages

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip jq git
npm install -g @anthropic-ai/claude-code
```

`jq` is required — the hooks parse JSON with it. On macOS use `brew install python jq git`.

## 2. Folders and data

```bash
mkdir -p promo app tests data/raw data/interim data/out scratch
touch promo/__init__.py tests/__init__.py
git init
chmod +x .claude/hooks/*.sh
```

Copy the eight CSVs into `data/raw/`.

## 3. Python

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy pyarrow duckdb lightgbm scikit-learn scipy statsmodels cvxpy \
            streamlit plotly pydantic pytest ruff anthropic python-dotenv
pip freeze > requirements.txt
```

Activate the venv before every session. Debian-derived systems block installs outside one.

No API key is needed. Claude Code uses your account login. The `anthropic` package is only for Phase 8's chatbot, which needs separate API credit — ignore it until then.

## 4. Start Claude Code

```bash
claude
```

First run opens a browser to log in.

## 5. Verify (type these at the Claude Code prompt, not in your shell)

| Type | Expect |
|---|---|
| `/memory` | CLAUDE.md listed as Project memory |
| `/hooks` | Five hooks, all labelled Project Settings |
| `/agents` | `causal-reviewer`, `eval-runner`, `data-detective` |
| `/` | Menu containing `task`, `phase`, `gate`, `placebo` |
| `Write a test file to data/raw/scratch.txt` | **Denied.** If it succeeds, rerun the chmod and restart. |

Path-scoped rules in `.claude/rules/` will not appear in `/memory` until Claude reads a matching file. That is correct.

## 6. Work

```
/task 1.1     run one task from docs/plan.md, then stop
/phase 1      check whether a phase is genuinely finished
```

Do one task, read the output, then run the next. `docs/plan.md` is the plan; `docs/runbook.md` is the technical reference it points at.
