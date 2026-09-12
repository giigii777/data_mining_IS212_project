# ✈ Airline Passenger Satisfaction Analysis

A dark, SOC-style **Streamlit** dashboard built from
`data_mining_airline_updated.ipynb` (IS-212 Data & Knowledge Mining,
University of Computer Studies, Yangon, 2025–2026), running on the
**original `airline.csv` dataset**.

## Run it

```bash
pip install streamlit pandas numpy plotly scikit-learn mlxtend
cd airline-satisfaction-app
streamlit run app.py
```

The app opens on `http://localhost:8501`.

## Pages

| Page | What you get |
|------|--------------|
| 🏠 **Overview** | Hero status bar, KPI cards, satisfaction donut, segment bars, `head(10)` preview, data-quality audit (missing / duplicates / IQR outliers / rating validity), column inventory |
| 📊 **EDA Explorer** | Per-field histograms (log toggle for delays), IQR outlier table + box plots, skewness panel, satisfied-rate by segment, rating gaps (satisfied vs neutral), 18×18 correlation heatmap |
| 🔍 **Descriptive Mining** | K-Means: elbow + silhouette scan (K=2–8), selected K, cluster sizes, satisfaction split, z-colored cluster profile · Apriori: frequent itemsets and rules with support/confidence sliders |
| 🤖 **Model Lab** | Full predictive pipeline — 80/20 stratified split, median impute + one-hot + scale, SelectKBest (mutual information, top 20), Logistic Regression vs Decision Tree vs Random Forest with accuracy/precision/recall/F1 bars, train-vs-test overfit check, 5-fold CV, confusion matrices, ROC curves, MI scores, RF feature importance, actual-vs-predicted sample |
| 🎯 **Predict** | Live scoring form (demographics + flight + 14 service ratings): SATISFIED / NEUTRAL badge, probability bars, confidence ring, top-3 contributing factors (neutral perturbation), similar-passenger benchmark |
| ℹ **About** | Notebook → app mapping, methodology, dataset disclosure |

## Data

`data/airline.csv` is the **original project dataset**: 103,904 passenger
survey records × 23 analysis columns (index/id columns dropped
automatically), 310 missing *Arrival Delay in Minutes* values
(median-imputed during cleaning), and the real class split —
45,025 satisfied (43.3%) vs 58,879 neutral or dissatisfied (56.7%).
All charts, clusters, association rules and models in the app run on
this data.

**Use your own data:** upload any CSV with the same 23-column
schema via the sidebar (extra index/id columns are dropped
automatically) — every chart, cluster, rule and model retrains
instantly against your file.

## Notebook reference results

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Logistic Regression | 0.8743 | 0.8652 | 0.8409 | 0.8528 |
| Decision Tree | 0.9424 | 0.9267 | 0.9417 | 0.9341 |
| **Random Forest** | **0.9624** | **0.9705** | **0.9418** | **0.9559** |

Clustering: K = 2 (silhouette 0.1526) · clusters 46,594 / 57,310 ·
satisfaction split 17% / 65%.
Top drivers: Online boarding · Inflight wifi service · Type of Travel · Class.
