# HANDBOOK — everything you need, offline

This is the only file you need. Setup, then how to drive Claude Code through all nine phases. Keep it open in a text editor.

---

# PART 1 — SETUP (do once)

## 1.1 Unpack

```bash
mkdir ~/promo && cd ~/promo
unzip -o ~/Downloads/promo-starter.zip -d .
```

## 1.2 System packages

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip jq git
npm install -g @anthropic-ai/claude-code
```

macOS instead: `brew install python jq git`

## 1.3 Folders

```bash
cd ~/promo
mkdir -p promo app tests data/raw data/interim data/out scratch
touch promo/__init__.py tests/__init__.py
git init
chmod +x .claude/hooks/*.sh
```

Copy the eight CSVs into `~/promo/data/raw/`.

## 1.4 Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy pyarrow duckdb lightgbm scikit-learn scipy statsmodels cvxpy \
            streamlit plotly pydantic pytest ruff anthropic python-dotenv
pip freeze > requirements.txt
```

**You must run `source .venv/bin/activate` every time you open a new terminal.** If you forget, installs and scripts fail.

No API key needed. Claude Code uses your account login. The `anthropic` package is only for Phase 8's chatbot, which needs separate paid API credit — ignore it until then.

## 1.5 Start

```bash
claude
```

First run opens a browser to log in. After that you get a prompt where you type.

## 1.6 Verify

Type each of these **at the Claude Code prompt** and press Enter. Escape closes any menu.

- `/memory` — shows CLAUDE.md as Project memory
- `/hooks` — top line says "4 hooks configured". The long list of 31 events is normal; it shows every event Claude Code supports, not just yours
- `/agents` — shows `causal-reviewer`, `eval-runner`, `data-detective`
- `/` — menu includes `task`, `phase`, `gate`, `placebo`

Then type this sentence:

```
Write a test file to data/raw/scratch.txt
```

It must refuse. If it writes the file, quit, run `chmod +x .claude/hooks/*.sh`, restart `claude`.

Setup is done.

---

# PART 2 — HOW TO DRIVE IT

## The loop

1. Type `/task X.Y`
2. Claude asks permission to create or edit files — press Enter to allow
3. Read the output
4. Type `/task X.Z` for the next one

At the end of a phase, type `/phase N` to check the phase is genuinely finished before moving on.

## Keys

| Key | Does |
|---|---|
| Enter | Send, or accept a permission prompt |
| Escape | Close a menu, or interrupt Claude mid-work |
| Shift+Tab twice | Plan mode — Claude plans before editing. Use for Phase 4 |
| `/clear` | Wipe context. Use between phases to stay fast |
| Ctrl+C twice | Quit |

## When something goes wrong

- **Claude drifts into later work:** Escape, then type `Stop. Do only Task X.Y from docs/plan.md.`
- **Memory crash / machine freezes:** type `Use DuckDB to query the CSV directly instead of loading it into pandas.` The memory rule should prevent this, but say it if you see a plain `read_csv` on transaction data.
- **Slow or confused after long work:** `/clear`, then `Read docs/plan.md and continue from Task X.Y.`
- **A number looks wrong:** `@causal-reviewer` — reviews the estimation code for identification errors.
- **Wondering if the model works:** `@eval-runner` — runs the synthetic and placebo harnesses.

---

# PART 3 — THE PHASES

Every task below is a `/task` command. The full specification for each lives in `docs/plan.md`, which Claude reads. You just type the command.

## Phase 1 — Know the data

Nothing is cleaned or modelled here. You are finding out what is true.

```
/task 1.1     shape, coverage, ID ranges, date spans
/task 1.2     the QUANTITY column — find the volume-measured goods
/task 1.3     the three discount columns, and reconstruct the regular price
/task 1.4     which treatment actually varies (this is the important one)
/task 1.5     structural zeros and repurchase cycles
```

Then:

```
Write docs/data_findings.md recording every decision from Phase 1: which treatment
varies and on which axis, which regular-price reconstruction I chose and why, the
share of units that are volume-measured, the no-trip share, and the median
repurchase cycle for the top categories.
```

```
/phase 1
```

**Expect:** display and mailer vary across product, store, and week. Campaign membership does not. That finding is what makes the rest possible.

## Phase 2 — Build the data layer

The biggest phase. Everything downstream reads its output.

```
/task 2.1     ingest with a schema contract
/task 2.2     transaction cleaning, with recorded effects
/task 2.3     price decomposition and depth
/task 2.4     price index and deflation
/task 2.5     the treatment panel from causal_data
/task 2.6     derived model variables (lags, seasonality, controls)
/task 2.7     availability flags and repurchase cycles
/phase 2
```

**Done when:** `data/interim/panel.parquet` exists at product × store × week with all features, and `data/interim/quality.json` records every exclusion.

## Phase 3 — The feasibility gate

Small in code, large in value. This is your first demoable checkpoint.

```
/task 3.1     variation across axes
/task 3.2     overlap via propensity classifier
/task 3.3     treatment collisions and horizon checks
/task 3.4     break-even margin sweep
/task 3.5     the refusal engine with all ten reason codes
/phase 3
```

Then try it:

```
/gate display
```

**If you ran out of time here, you would still have a real product.** The audit alone answers the lecture's closing line.

## Phase 4 — Baseline and incremental lift (MVP 01, 02)

Use plan mode here: press Shift+Tab twice before `/task 4.1`.

```
/task 4.1     train on untreated rows only
/task 4.2     recursive rollout (the critical one)
/task 4.3     lift with an extended horizon
/task 4.4     synthetic truth harness
/task 4.5     placebo distribution
/phase 4
```

Then:

```
@eval-runner
/placebo
```

**Gate to pass:** the τ=0 synthetic case must recover approximately zero. If it doesn't, stop and fix it before Phase 5. Everything downstream is built on this number.

## Phase 5 — Accounting (MVP 03)

```
/task 5.1
/phase 5
```

There is no COGS in this data, so true ROI is not computable. The output is the break-even margin plus a sensitivity table across assumed margins. **That is the correct answer, not a shortcoming** — say so in the demo.

## Phase 6 — Cannibalisation (MVP 04)

You have BASKET_ID, so this is identified. Most teams can't do it at all.

```
/task 6.1     transfer matrix from baskets
/task 6.2     decomposition: delta_q = s + (g - l)
/phase 6
```

Then verify the invariant:

```
@causal-reviewer
```

## Phase 7 — Ranking and recommendation (MVP 05, 06)

```
/task 7.1
/phase 7
```

Shrinkage before ranking, response curves, and the MDE holdout calculator.

## Phase 8 — Agent, UI, chatbot

Only now. Building this earlier is the commonest way hackathon projects fail.

```
/task 8.1     orchestration (deterministic, no LLM)
/task 8.2     Streamlit UI, four pages
/task 8.3     narration layer
```

For 8.3 you need an API key with separate credit:

```bash
echo "ANTHROPIC_API_KEY=sk-..." > .env
```

If you don't have credit, skip 8.3. Pre-write the refusal messages as templates — the deterministic fallbacks are already specified in `promo/gates.py` and the demo works fine without live generation.

Run the app:

```bash
source .venv/bin/activate
streamlit run app/app.py
```

---

# PART 4 — IF YOU RUN OUT OF TIME

Cut in exactly this order:

1. Planner optimisation → a top-3 heuristic list
2. Chat panel → keep only pre-written verdict paragraphs
3. Transfer matrix → report gains and losses without cell-level flows
4. Intervals on the recommender → keep them on the estimates

**Never cut:** the placebo harness, the refusal engine, the recursive rollout, the data-honesty report. Those four are the argument. Everything else decorates it.

---

# PART 5 — THE DEMO

Four minutes, hits all six MVPs in order:

1. **Audit page.** Open on a campaign the system refuses to score, and read why.
2. **Data honesty.** Show the volume-measured units finding — a tiny share of rows carrying most of the raw units.
3. **Counterfactual.** An accepted campaign, actual versus baseline, with the rollout band.
4. **Placebo.** The distribution with your estimate sitting outside it. This is where you prove the number means something.
5. **Decomposition.** Expansion and redistribution side by side — and say why they are added, not netted.
6. **Break-even.** The margin table, and the fact that the system refuses to invent a margin it wasn't given.
7. **Close on the holdout designer.** Hold out 20% next time and most of this machinery becomes unnecessary.

## The three lines to say out loud

- "A number returned without identification is worse than no number."
- "A system that invents the missing input is more dangerous than one that reports the gap."
- "Every hard thing in this project exists to compensate for one decision nobody made: holding out a control group."

## What differentiates you

- The refusal engine — most teams always return a number
- The placebo band — evidence rather than assertion
- The recursive rollout — most teams' multi-week estimates silently shrink toward zero
- Post-promotion dip correction — most teams bank the peak and never see the trough

---

# PART 6 — QUICK REFERENCE

```bash
cd ~/promo && source .venv/bin/activate && claude    # start a session
pytest -q                                            # run tests
streamlit run app/app.py                             # run the UI
```

| At the Claude Code prompt | Does |
|---|---|
| `/task 2.3` | Run one task from the plan, then stop |
| `/phase 2` | Check a phase is genuinely finished |
| `/gate display` | Feasibility audit on a treatment |
| `/placebo` | Locate where the estimator's zero sits |
| `@causal-reviewer` | Review estimation code for identification errors |
| `@eval-runner` | Run synthetic truth and placebo harnesses |
| `@data-detective` | Investigate a data question without cluttering context |
| `/clear` | Wipe context between phases |
| `/memory` `/hooks` `/agents` | Check the setup loaded |

**The one rule that matters:** one task, read the output, next task. Do not let it run ahead.
