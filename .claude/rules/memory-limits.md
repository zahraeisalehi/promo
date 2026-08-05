---
paths:
  - promo/**/*.py
  - scratch/**/*.py
  - tests/**/*.py
---

# Memory discipline

`transaction_data.csv` has millions of rows. This machine has limited RAM and has already crashed once from a naive load. Treat memory as a hard constraint, not an optimisation.

## Never

- `pd.read_csv("data/raw/transaction_data.csv")` with no arguments.
- Loading the full transaction table to compute a summary statistic.
- Merging the full transaction table against another frame before aggregating.
- Holding more than one copy of a large frame alive at once.

## Instead

Use DuckDB to query the CSVs directly, and only materialise aggregates:

```python
import duckdb
con = duckdb.connect()
df = con.execute("""
    SELECT PRODUCT_ID, STORE_ID, WEEK_NO,
           SUM(QUANTITY) AS units,
           SUM(SALES_VALUE) AS sales
    FROM 'data/raw/transaction_data.csv'
    GROUP BY 1,2,3
""").df()
```

DuckDB streams from disk and never loads the whole file. This is the default approach for anything touching transactions.

When pandas is genuinely needed, pass `usecols` and explicit `dtype`, and read in chunks:

```python
pd.read_csv(path, usecols=[...], dtype={...}, chunksize=1_000_000)
```

## Rules

- Exploration writes aggregates to `data/interim/` as parquet, then works from those. Never re-read the raw CSV twice in one script.
- IDs are int64 or string, never float. `BASKET_ID` loses precision as float64.
- Sample before profiling: a 200k-row random sample answers most distribution questions. Say in the output that it was a sample and how large.
- If a step genuinely needs the full table, aggregate inside the SQL query, not after it.
