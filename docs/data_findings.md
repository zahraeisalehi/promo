# Data findings

What is actually in the Dunnhumby files, established in Phase 1. Sections are
added as tasks complete; a heading absent here means the question has not been
asked yet, not that it has no answer.

Every figure below was computed on `data/interim/transactions.parquet` (the
mirror of `transaction_data.csv`, which CLAUDE.md forbids reading directly) and
`data/raw/product.csv`, by full scan in DuckDB. Each section names the script
that reproduces it.

---

## Task 1.2 — The quantity column

Reproduce: `python scratch/explore_02.py`. Full scan, no sampling, so the shares
are exact for the mirror.

**Finding: `QUANTITY` is not one unit of measure. 1.07% of rows carry 98.71% of
all raw units, and those rows are petrol and miscellaneous kiosk transactions
priced at roughly a quarter of a cent per unit. Summing `QUANTITY` across the
file measures fuel volume, not grocery demand.**

### The column as a whole

| | |
|---|---|
| rows | 2,595,732 |
| total `QUANTITY` | 260,685,622 |
| total `SALES_VALUE` | 8,057,463.08 |
| mean `QUANTITY` per row | 100.43 |
| min / max `QUANTITY` | 0 / 89,638 |
| median `QUANTITY` | 1 |
| p99 `QUANTITY` | 10 |
| p99.9 `QUANTITY` | 16,963 |

The median row buys one of something and the mean row buys a hundred. That gap
between p99 (10) and p99.9 (16,963) is the whole finding in one line.

Degenerate rows, which matter because they make unit value undefined or zero:

| case | rows | share |
|---|---:|---:|
| `QUANTITY = 0` | 14,466 | 0.56% |
| `QUANTITY < 0` | 0 | 0.00% |
| `SALES_VALUE = 0` | 18,850 | 0.73% |
| `SALES_VALUE < 0` | 0 | 0.00% |

There are **no returns** in this file — neither quantity nor sales value is ever
negative. The refunds live in the discount columns, not here.

### Where the units are

Ranking rows by `QUANTITY` descending:

| slice | share of all units |
|---|---:|
| top 0.01% of rows | 2.91% |
| top 0.1% | 20.39% |
| top 1% | 98.73% |
| top 10% | 99.00% |

### Unit value, `SALES_VALUE / QUANTITY`

Defined on the 2,581,266 rows with `QUANTITY > 0`. Range 0.000000 to 499.99,
mean 2.44, median 1.92.

| decade band | rows | row % | units | unit % |
|---|---:|---:|---:|---:|
| `SALES_VALUE <= 0` | 4,451 | 0.17% | 4,544 | 0.00% |
| 1e-3 .. 1e-2 | 23,153 | 0.90% | 257,330,170 | **98.71%** |
| 1e-2 .. 1e-1 | 281 | 0.01% | 538 | 0.00% |
| 1e-1 .. 1e+0 | 492,147 | 19.07% | 841,748 | 0.32% |
| 1e+0 .. 1e+1 | 2,022,688 | 78.36% | 2,466,522 | 0.95% |
| 1e+1 .. 1e+2 | 38,464 | 1.49% | 42,007 | 0.02% |
| 1e+2 .. 1e+3 | 82 | 0.00% | 93 | 0.00% |

Two populations, not a tail. Counted groceries sit in the 1e-1 to 1e+1 decades —
97% of rows, under 1.3% of units. A separate population sits two decades below
at a fraction of a cent and carries almost everything.

### Below the 0.05 line

| | below line | all rows | share |
|---|---:|---:|---:|
| rows | 27,714 | 2,595,732 | **1.07%** |
| `QUANTITY` | 257,334,912 | 260,685,622 | **98.71%** |
| `SALES_VALUE` | 604,322.40 | 8,057,463.08 | 7.50% |

Mean `QUANTITY` per row below the line: 9,285. Above it: about 1.3.

### What those rows are

By `DEPARTMENT`, ranked by units carried below the line:

| DEPARTMENT | rows | row % | units | unit % | sales | mean unit value |
|---|---:|---:|---:|---:|---:|---:|
| KIOSK-GAS | 20,277 | 73.17% | 221,253,105 | 85.98% | 514,974.01 | 0.0023 |
| MISC SALES TRAN | 2,884 | 10.41% | 36,077,082 | 14.02% | 89,342.26 | 0.0025 |
| GROCERY | 1,246 | 4.50% | 1,329 | 0.00% | 4.06 | 0.0009 |
| MEAT-PCKGD | 939 | 3.39% | 955 | 0.00% | 0.02 | 0.0000 |
| DRUG GM | 681 | 2.46% | 719 | 0.00% | 1.01 | 0.0007 |
| COUP/STR & MFG | 448 | 1.62% | 460 | 0.00% | 0.07 | 0.0002 |

Those two departments carry **99.998%** of the units below the line between them.
Everything after them is a long list of ordinary grocery rows that happen to be
priced at zero — giveaways and coupon lines — contributing rows but no units.

Measured the other way, against each department's own totals:

| DEPARTMENT | all rows | all units | % of its units below line | % of its rows below line |
|---|---:|---:|---:|---:|
| KIOSK-GAS | 22,059 | 221,254,887 | 100.00% | 91.92% |
| MISC SALES TRAN | 6,050 | 36,080,860 | 99.99% | 47.67% |
| COUP/STR & MFG | 817 | 1,011 | 45.50% | 54.83% |
| GROCERY | 1,646,076 | 2,194,762 | 0.06% | 0.08% |
| DRUG GM | 277,232 | 353,844 | 0.20% | 0.25% |

KIOSK-GAS and MISC SALES TRAN are wholly volume-measured. No real merchandise
department loses more than about 1% of its units to the line.

**`COMMODITY_DESC` does not identify these rows.** 99.95% of below-line units are
labelled `COUPON/MISC ITEMS`; the commodity literally named `FUEL` accounts for
6 rows and 0.05% of them. The single largest row in the file is `PRODUCT_ID`
6534178, `KIOSK-GAS` / `COUPON/MISC ITEMS`, 89,638 units for 250.00 — a tank of
petrol recorded in hundredths of a gallon under a catalogue label that says
nothing. `DEPARTMENT` is the usable discriminator here, not `COMMODITY_DESC`.

### The 0.05 threshold is not load-bearing

| threshold | rows | row % | units | unit % | sales % |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 4,451 | 0.17% | 4,544 | 0.00% | 0.00% |
| 0.005 | 27,604 | 1.06% | 257,334,714 | 98.71% | 7.50% |
| 0.010 | 27,604 | 1.06% | 257,334,714 | 98.71% | 7.50% |
| 0.050 | 27,714 | 1.07% | 257,334,912 | 98.71% | 7.50% |
| 0.100 | 28,301 | 1.09% | 257,337,464 | 98.72% | 7.50% |
| 0.250 | 48,630 | 1.87% | 257,391,890 | 98.74% | 7.62% |
| 0.500 | 154,734 | 5.96% | 257,612,825 | 98.82% | 8.59% |

Anywhere from 0.005 to 0.10 returns the same 98.71% of units. The two
populations are separated by a genuinely empty band, so the answer is a property
of the data rather than of the cutoff. Below 0.005 the filter stops catching
volume rows and catches only the 4,451 zero-priced rows, which are a different
phenomenon (free goods) carrying 4,544 units.

### Consequences for the pipeline

1. **Units are not comparable across departments.** Any `sum(QUANTITY)` at the
   PRODUCT_ID × STORE_ID × WEEK_NO grain that includes KIOSK-GAS or
   MISC SALES TRAN is a mixture of gallons and tins. A baseline trained on raw
   units in a panel that includes them fits petrol volume.
2. **The identifying column is `DEPARTMENT`, not unit value and not
   `COMMODITY_DESC`.** A unit-value threshold also sweeps in several thousand
   zero-priced grocery rows, which are a separate issue; `COMMODITY_DESC`
   labels petrol as `COUPON/MISC ITEMS`.
3. **Sales value is much less distorted than units.** These rows are 7.50% of
   `SALES_VALUE` against 98.71% of `QUANTITY`, so a revenue-denominated
   quantity is far more robust to this problem than a unit-denominated one.
4. **Nothing was dropped.** Per the plan, exclusion happens in Phase 2 with a
   recorded before/after on rows, units, and sales value; per CLAUDE.md, these
   rows are flagged, never silently removed. The candidate flag —
   `DEPARTMENT IN ('KIOSK-GAS', 'MISC SALES TRAN')`, with the zero-price rows
   flagged separately — is proposed here as evidence, and is Phase 2's decision
   to make and record.

---

## Task 1.3 — The discount columns

Reproduce: `python scratch/explore_03.py`. Part 1 is a full scan; part 2 scans
the 200 sampled products only.

**The decision is recorded at the end of this section.** Everything before it is
the evidence the decision was made on.

### The three columns

| column | nonzero rows | nonzero % | sum | as % of `SALES_VALUE` |
|---|---:|---:|---:|---:|
| `RETAIL_DISC` | 1,303,062 | 50.20% | -1,398,334.84 | 17.35% |
| `COUPON_DISC` | 36,422 | 1.40% | -42,611.54 | 0.53% |
| `COUPON_MATCH_DISC` | 17,449 | 0.67% | -7,575.81 | 0.09% |

Magnitude over the rows where each fires:

| column | min | p1 | p25 | median | p75 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `RETAIL_DISC` | -180.00 | -6.98 | -1.22 | -0.69 | -0.32 | -0.04 | **+3.99** |
| `COUPON_DISC` | -55.93 | -6.73 | -1.00 | -1.00 | -0.50 | -0.25 | -0.08 |
| `COUPON_MATCH_DISC` | -7.70 | -1.00 | -0.50 | -0.45 | -0.30 | -0.25 | -0.01 |

**The retailer's own markdown is the dominant mechanic by two orders of
magnitude.** It touches half of all rows and 17.35% of revenue; the two coupon
mechanics together touch under 2% of rows and 0.62% of revenue.

**`RETAIL_DISC` is not uniformly negative.** 36 rows are positive, summing
+8.53, max +3.99; 17 of them have `SALES_VALUE = 0`. Both candidate
reconstructions use `abs()`, which silently converts those surcharges into
discounts. Sign-flip count is small, but the rule "all three arrive negative"
has exceptions and code that assumes otherwise will not notice. `COUPON_DISC`
and `COUPON_MATCH_DISC` have no positive rows.

### How the mechanics co-occur

| `RETAIL_DISC` | `COUPON_DISC` | `COUPON_MATCH_DISC` | rows | row % | sales % |
|---|---|---|---:|---:|---:|
| yes | – | – | 1,283,013 | 49.43% | 49.41% |
| – | – | – | 1,276,297 | 49.17% | 49.00% |
| yes | yes | yes | 10,846 | 0.42% | 0.36% |
| – | yes | – | 9,770 | 0.38% | 0.42% |
| yes | yes | – | 9,203 | 0.35% | 0.57% |
| – | yes | yes | 6,603 | 0.25% | 0.24% |

`COUPON_MATCH_DISC` fires on **zero** rows where `COUPON_DISC` is zero. The
match is strictly nested inside the manufacturer coupon, which is what the
mechanic's name claims, and means it can never be reconstructed independently.

Discount depth where each mechanic fires, `|disc| / (SALES_VALUE + |disc|)`:

| column | rows | p25 | median | p75 | p95 | at 100% |
|---|---:|---:|---:|---:|---:|---:|
| `RETAIL_DISC` | 1,303,026 | 15.67% | 24.62% | 37.11% | 50.13% | 0.29% |
| `COUPON_DISC` | 36,422 | 15.81% | 24.01% | 37.50% | 100.00% | **13.72%** |
| `COUPON_MATCH_DISC` | 17,449 | 10.03% | 15.08% | 22.50% | 40.00% | 0.07% |

13.72% of coupon rows are 100% discounts — the shopper paid nothing. Under
reconstruction B those rows price at the coupon face value; under A they price
at `|RETAIL_DISC|` alone, which is often zero, giving a reconstructed regular
price of 0.00. Neither is obviously right and both are wrong for some rows.

### The two reconstructions

```
A = (SALES_VALUE + |RETAIL_DISC| + |COUPON_MATCH_DISC|) / QUANTITY
B = (SALES_VALUE + |RETAIL_DISC| + |COUPON_DISC| + |COUPON_MATCH_DISC|) / QUANTITY
```

Sample: the 200 products with the most transactions that appear in at least 8
weeks — 501,918 rows, 249.7M units, 1,661,358.26 of sales, spanning all 102
weeks. Departments: GROCERY (106), PRODUCE (55), DELI (10), MEAT (9), DRUG GM
(8), PASTRY (3), MEAT-PCKGD (3), MISC SALES TRAN (2), SALAD BAR (2), KIOSK-GAS
(2). The four volume-measured products flagged in Task 1.2 are scored, then
scored again with them removed.

**A and B differ on 1,417 rows — 0.28% of the sample.** They are algebraically
identical everywhere `COUPON_DISC = 0`. That share is the ceiling on any
difference between them, and it is why most comparisons below are ties rather
than close calls.

Stability of the modal price, scored per product and compared pairwise on the
same products, weeks and rows. Medians across the 200 products:

| metric | A | B | better |
|---|---:|---:|:--|
| within-week support of the modal price | 0.9507 | 0.9495 | **A** |
| distinct modal prices across weeks | 4.50 | 4.50 | tie |
| share of weeks at the dominant price | 0.5514 | 0.5514 | tie |
| coefficient of variation of weekly modal price | 0.1048 | 0.1048 | tie |
| median week-over-week move of modal price | 0.0000 | 0.0000 | tie |

Paired per-product wins (tolerance 1e-9; see *Determinism* below):

| metric | A wins | B wins | tie |
|---|---:|---:|---:|
| within-week support of the modal price | **77** | **0** | 123 |
| distinct modal prices across weeks | 2 | 1 | 197 |
| share of weeks at the dominant price | 2 | 0 | 198 |
| coefficient of variation of modal price | 2 | 2 | 196 |
| median week-over-week move of modal price | 1 | 0 | 199 |

Excluding the four volume-measured products (196 products) the picture is
unchanged: within-week support 77–0 to A, everything else 2–1, 2–0, 2–2, 1–0.

**The one metric that separates them separates them unanimously.** On 77 of 200
products, adding `COUPON_DISC` back scatters the within-week price distribution;
on none of them does it tighten it. The other four metrics see essentially
nothing, because they read the modal price itself rather than its support, and
0.28% of rows almost never move a mode.

Where the two disagree most on the count of distinct modal prices:

| PRODUCT_ID | COMMODITY | weeks | modes A | modes B | dominant A | dominant B |
|---:|---|---:|---:|---:|---:|---:|
| 971922 | CHEESES | 101 | 38 | 37 | 3.05 | 3.00 |
| 1132771 | CHEESES | 102 | 27 | 28 | 3.11 | 3.05 |
| 870547 | MARGARINES | 102 | 7 | 8 | 1.99 | 1.99 |
| 820165 | CITRUS | 91 | 9 | 9 | 0.50 | 0.50 |

Even the worst disagreement is one mode in thirty-eight, and the two dominant
prices differ by five cents. For PRODUCT_ID 870547 the two weekly modal series
are identical over its first 26 weeks (3.19 for nine weeks, then 1.99).

### Reading of the evidence

The direction is consistent with the mechanics: a manufacturer coupon is
deducted at the till and does not move the shelf price, so adding it back pushes
those rows above the price everyone else in that store-week paid, which is
exactly the scatter the support metric picks up. Reconstruction A never loses on
that metric. But the effect is small, confined to 0.28% of rows, and measured on
a metric already saturated at 95%.

### What this evidence cannot settle

- **Stability is not correctness.** The criterion rewards any transformation
  that varies less; a constant would score perfectly. It says A's prices are
  more concentrated, not that A's level is the true shelf price. A level test —
  comparing each reconstruction against the modal price on rows where all three
  discounts are zero — would test that, and has not been run.
- **The 100%-coupon rows** (13.72% of coupon rows) are mispriced by both
  candidates and need their own handling either way.
- **Neither candidate addresses the 36 positive `RETAIL_DISC` rows.**
- **Cost-bearer, not just price level.** A and B answer "what would the shopper
  have paid"; the accounting in Phase 5 needs "what did the retailer forgo",
  which is a different combination of the same three columns. Choosing a shelf
  price does not choose that.

### Determinism

The paired win counts are computed at a relative tolerance of 1e-9, not by exact
equality. DuckDB sums doubles in thread-arrival order, so scores that are
mathematically identical differ in the last bits between runs. With exact `=`,
the within-week-support count moved between 104–28–68 and 97–36–67 on two
consecutive runs of the same script — noise, entirely from ties being
reclassified as wins. At tolerance it is a stable 77–0–123. Any downstream
comparison of two float aggregates needs the same treatment.

---

### DECISION — reconstruction A

**The regular price is reconstructed as A:**

```
regular_price_per_unit = (SALES_VALUE + |RETAIL_DISC| + |COUPON_MATCH_DISC|) / QUANTITY
```

`COUPON_DISC` is **excluded** from the shelf price.

**Rationale.** `COUPON_DISC` is manufacturer-funded and reimbursed to the
retailer, so the retailer's revenue is unaffected by it. Adding it back would
double-count that reimbursement and invent a price that was never charged to
anyone. `RETAIL_DISC` and `COUPON_MATCH_DISC` are both borne by the retailer and
both move the price the shopper faces, so both belong in the reconstruction.

**Empirical support.** Within-week support of the modal price favours A **77–0**
with 123 ties across the 200 sampled products, and 77–0 again with the
volume-measured products removed. No metric favours B.

**Scope of this decision.** It fixes the *shelf price* only. What the retailer
forgave — the Phase 5 accounting quantity — is a different combination of the
same three columns and is not decided here. All three columns stay separate
through the pipeline.

### Implementation requirements for Phase 2

These are conditions on the code that implements A, not suggestions.

1. **Do not call `abs()` on `RETAIL_DISC`.** 36 rows are positive (max +3.99);
   `abs()` would silently convert surcharges into discounts. Flag those rows as
   surcharges and exclude them, recording the exclusion's effect on row count,
   units, and sales value like any other exclusion.
2. **Guard the divide.** 14,466 rows have `QUANTITY = 0`, so the per-unit price
   is undefined for 0.56% of rows. The reconstruction must produce a null there,
   never an infinity and never a silent zero.
3. **Compare prices at a tolerance of 1e-9, never with `==`.** Exact float
   equality was not reproducible across runs of `explore_03.py` — DuckDB sums
   doubles in thread-arrival order — and the same hazard applies to any price
   equality test, modal-price grouping, or "did the price change" check.

### Outstanding — to be run in Task 2.3

**The level test has not been run.** The evidence above establishes that A's
prices are more *concentrated* than B's; it does not establish that A's *level*
is the true shelf price, because the stability criterion would also reward a
constant. The test: compare the reconstructed price against the observed paid
price on rows where all three discount columns are zero, where the paid price
*is* the regular price by construction. If A is right, the two should agree
within a cent for the same product-store-week. Run it in Task 2.3 as a validity
check on the price decomposition, and record the result here. Should it fail,
this decision is reopened.
