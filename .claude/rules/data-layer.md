---
paths:
  - promo/io.py
  - promo/clean.py
  - promo/prices.py
  - promo/treatment.py
  - promo/features.py
  - promo/quality.py
  - scratch/*.py
---

# Dunnhumby data layer rules

Eight CSVs in `data/raw/`. These facts are settled — do not re-derive or contradict them.

## Grain and joins

The panel is PRODUCT_ID x STORE_ID x WEEK_NO. Household-level analysis is a separate optional axis, not the default.

- `transaction_data.csv` is the spine. DAY runs roughly 1-711, WEEK_NO 1-102.
- `causal_data.csv` joins on PRODUCT_ID + STORE_ID + WEEK_NO and carries the treatment.
- `product.csv` joins on PRODUCT_ID for DEPARTMENT, COMMODITY_DESC, BRAND.
- `hh_demographic.csv` covers well under half of households. Anything using it runs on a subset and must say so in its diagnostics.
- IDs are identifiers, not numbers. Read them as int64 or string, never float, or BASKET_ID loses precision.

## The treatment

`display` and `mailer` from `causal_data.csv` are the treatment, because they vary across product, store, and week. Both are categorical codes, not booleans: `display` uses digits including 0 for none, `mailer` uses letters including 0 for none. Preserve the raw codes alongside any derived boolean.

Campaign membership (`campaign_table.csv`) and coupon redemption (`coupon_redempt.csv`) are household-targeted and mostly fail the variation test. They are secondary treatments at best.

Product-store-weeks absent from `causal_data.csv` are untreated, not missing. Record what share of the panel that assumption covers.

## Prices

`SALES_VALUE` is what the shopper paid. The regular price is reconstructed, not read:

- `RETAIL_DISC` is the retailer's loyalty-card discount.
- `COUPON_DISC` is manufacturer-funded.
- `COUPON_MATCH_DISC` is the retailer matching a manufacturer coupon.

All three arrive negative. Keep them as separate components through the whole pipeline — they have different cost bearers and are not the same treatment. Which components belong in the reconstructed shelf price is a recorded decision in `docs/data_findings.md`, not a default.

Deflate before any price comparison. Expect near-flat drift on this dataset; the module exists because the pitch targets high-inflation retail.

## Known traps

- `QUANTITY` mixes counted goods with volume-measured goods sold at fractions of a cent. A tiny share of rows carries most of the raw units. Flag with a boolean column; never silently drop.
- A household-week with no transactions is a structural zero (no shopping trip), not a decision not to buy. Carry an availability flag.
- Stockouts are unobservable here. State the bias direction rather than pretending: stockouts correlate with promotions, so effects are understated.
- There is no COGS or margin column anywhere. Never impute one.

## Diagnostics discipline

Every exclusion records row count, share of total units, and share of total sales value, before and after. A filter whose effect is not recorded is indistinguishable from a bug.
