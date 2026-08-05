---
paths:
  - promo/**/*.py
  - app/**/*.py
  - tests/**/*.py
---

# Python conventions

- Type hints on every public function. Pydantic models for anything crossing a stage boundary.
- Stage functions return `(DataFrame, dict)`. Diagnostics are returned, never printed.
- No bare `except`. No `try/except` that swallows a numerical failure — surface it as a diagnostic field.
- Use `assert` for mathematical invariants (mass conservation, monotonicity). Use `GateResult` for data conditions.
- Vectorise. If a loop iterates over rows of a DataFrame, it needs a comment explaining why.
- Randomness takes an explicit `seed` argument and uses `np.random.default_rng`. No global seeding.
- Plotting belongs in `app/`, never in `promo/`.
