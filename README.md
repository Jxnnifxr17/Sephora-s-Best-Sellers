# Sephora Best Sellers: Popularity, Pricing and Brand Positioning

Exploratory machine learning project on 200 products from a Sephora best-seller listing. It asks three questions:

1. What drives a product's review count (a proxy for popularity)?
2. Can price be predicted from brand, category, name keywords and tags?
3. How do brands group together by price and volume?

## Key findings

- **Popularity:** Random Forest reached R² = 0.58 (RMSE 1.28 in log units) under leakage-safe grouped cross-validation. The strongest drivers were the *New* tag, the *Sets & Gift Sets* category and the word "set" in the name, all associated with fewer reviews. A high/low popularity classifier reached ROC AUC = 0.79.
- **Price:** Ridge was best (R² = 0.26, MAE about $11.36). Price is only weakly predictable from the available fields; product size is likely the biggest missing feature.
- **Brands:** k-means (k = 2) split 27 brands into a premium, lower-volume group (avg price about $51) and an accessible, higher-volume group (avg price about $29). Silhouette is about 0.27, so the groups are loose.

## Repository contents

| File | Description |
|---|---|
| `Sephora_Best_Sellers.xlsx` | Cleaned dataset (`Best Sellers` sheet) and formula-driven `Summary` sheet |
| `Sephora_ML_Project.ipynb` | Full analysis: Sankey diagram, popularity modeling, price prediction, clustering |
| `sankey_brand_category.html` | Interactive Sankey diagram (generated when the notebook runs) |
| `Sephora Best Sellers Popularity, Pricing and Brand Positioning Analysis.docx.pdf` | Full report |

## Data

Copied from a Sephora product listing page and organized into one row per product.

| Column | Description |
|---|---|
| Brand, Product | As listed |
| Category | Assigned by keyword rules on the product name (13 categories); not a Sephora field |
| Review Count | Parsed from rounded display values (3.1K = 3,100) |
| Price Low / High / Avg | Ends of the listed range; average is the midpoint |
| New, Limited Edition, Online Only | Tags shown on the listing |
| Store Pickup | Whether a pickup store was listed |

## Methods

- **Leakage control:** minis and full sizes share the same review count, so products are grouped into product lines and evaluated with 5-fold `GroupKFold`.
- **Features:** standardized numeric inputs, one-hot category and brand (rare brands pooled), and TF-IDF of the product name.
- **Models:** mean baseline, Ridge, Random Forest, XGBoost; plain linear baseline for price. All scores are out-of-fold.
- **Explainability:** SHAP on the XGBoost popularity model.
- **Residual analysis:** out-of-fold price residuals flag items priced above or below expectation.
- **Clustering:** brand-level aggregates (brands with at least 3 listings), standardized log features, k-means chosen by silhouette, PCA map, Ward dendrogram as a check.

## Results

**Popularity (target: log review count)**

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Baseline | -0.004 | 1.962 | 1.553 |
| Ridge | 0.556 | 1.306 | 1.038 |
| Random Forest | 0.576 | 1.275 | 1.011 |
| XGBoost | 0.530 | 1.342 | 1.046 |

**Price (target: log average price)**

| Model | R² | RMSE | MAE ($) |
|---|---|---|---|
| Baseline | -0.011 | 0.472 | 13.84 |
| Linear (category + tags) | 0.080 | 0.450 | 12.92 |
| Ridge | 0.264 | 0.403 | 11.36 |
| Random Forest | 0.217 | 0.415 | 11.28 |
| XGBoost | 0.175 | 0.426 | 11.39 |

## Limitations

- Small sample (200 rows) from a single snapshot, so results are exploratory and high variance.
- **Time-on-shelf confound:** New products have had less time to collect reviews, so the negative *New* effect largely reflects age, not weaker demand.
- Review counts are rounded; prices for multi-size products are range midpoints; categories come from keyword rules.
- The sort order of the source page was not captured, so listing position is not a sales rank.
- No star rating, size, ingredients or launch date.
- SHAP values describe patterns in this sample, not out-of-sample evidence. The Online Only tag covers only 7 products.

## Next steps

1. Collect more pages or snapshots over time, plus ratings, size (price per ounce), launch dates and ingredients.
2. Model reviews per month on shelf to control for product age.
3. Compare TF-IDF to sentence embeddings for the product names.
4. Use nested cross-validation and repeated grouped splits for confidence intervals.
5. Build a Tableau or Streamlit dashboard combining the Sankey, positioning map and over/underpriced table.
