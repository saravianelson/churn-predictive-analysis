# 🛒 Predictive Churn Analysis — E-commerce (Online Retail II)

**Author:** Nelson Saravia  
**Dataset:** [Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) — Kaggle  
**Stack:** Python · pandas · scikit-learn · matplotlib · seaborn

---

## 🧭 Business Context

Customer churn is one of the most expensive problems in e-commerce. Acquiring a new customer costs 5–7x more than retaining an existing one. The challenge: **identifying who is about to leave before they do** — and acting on it with precision.

This project builds a predictive churn model on a real UK-based online retail dataset, with a focus on **translating model output into actionable business decisions**. The goal is not just accuracy — it's a model that actually works in production.

---

## ⚠️ The Core Problem: Data Leakage from Recency

During the initial model build, I identified a critical flaw that would have made the model **useless in a real business context**:

**Recency** (days since last purchase) was included as a training feature alongside the churn label. But Recency is *derived from the same time window used to define churn*. A customer labeled as "churned" (inactive for 90+ days) will, by definition, have a high Recency value. Including it as a feature means the model learns a tautology — not a predictive signal.

### What this means in practice
A model trained with Recency leakage would achieve near-perfect accuracy on training data, but **fail completely when deployed** — because at prediction time, you don't yet know if someone is churning. The model would be predicting the past, not the future.

### The fix
Recency was **excluded from the feature set**. The model was retrained using only:

| Feature | Description |
|---|---|
| `Frequency_Log` | Log-transformed number of unique orders |
| `Monetary_Log` | Log-transformed total spend |
| `Avg_Ticket` | Average spend per transaction |

This produces a model that reflects genuine behavioral patterns — not a data artifact.

---

## 📊 Dataset & Preprocessing

- **Source:** Online Retail II (UK e-commerce, 2009–2011)
- **Raw records:** ~1M transactions
- **After cleaning:** 805,549 valid records
  - Removed rows without `Customer ID` (anonymous sessions)
  - Removed returns (`Quantity < 0`) and price anomalies (`Price ≤ 0`)
- **Churn definition:** Customer inactive for **90+ days** from last purchase
- **RFM table:** Built per customer — Recency, Frequency, Monetary + engineered features

---

## 🔧 Model

**Algorithm:** Random Forest Classifier  
**Key parameters:**
- `n_estimators=100`
- `max_depth=12`
- `class_weight='balanced'` ← critical for imbalanced churn datasets
- `random_state=42`

The `class_weight='balanced'` adjustment ensures the model doesn't default to predicting "no churn" for everyone — which would achieve high accuracy but zero business value.

**Train/test split:** 80/20

---

## 📈 Results

### Feature Importance
The model confirms that **purchase frequency and monetary value** are the strongest predictors of churn — validating the feature selection after removing Recency.


### Risk Segmentation
Each customer receives a churn probability score (0–1), segmented into three risk tiers:


| Risk Level | Churn Probability |
|---|---|
| Low | 0 – 0.40 |
| Medium | 0.40 – 0.70 |
| High | 0.70 – 1.00 |

### Recommended Actions by Segment

| Segment | Customers | Action |
|---|---|---|
| 🔴 VIP Rescue | 93 | Direct contact + exclusive offer |
| 🟠 Churn Alert | 2,487 | Re-engagement campaign |
| 🟢 Loyalty VIP | 1,486 | VIP rewards program |
| ⚪ Standard Monitor | 1,812 | Routine follow-up |

The **VIP Rescue** segment is the highest-value intervention: high-spend customers showing strong churn signals. These 93 customers represent a disproportionate share of revenue risk.

---

## 💡 Business Impact

This model enables the business to:

1. **Stop wasting budget on mass retention campaigns** — target only the customers who actually need intervention
2. **Protect high-value customers first** — the VIP Rescue segment prioritizes revenue concentration
3. **Deploy at scale** — output integrates directly with Supabase or Power BI for operational use

---

## 📁 Project Structure

```
churn-analysis/
│
├── churn_analysis.ipynb     # Main notebook (EDA → Model → Segmentation)
├── ecommerce_churn_final.csv  # Output: scored customer base
└── README.md
```

---

## 🚀 How to Run

1. Clone this repo
2. Install dependencies: `pip install kagglehub pandas scikit-learn matplotlib seaborn`
3. Run `churn_analysis.ipynb` end-to-end in Google Colab or Jupyter
4. The notebook auto-downloads the dataset from Kaggle via `kagglehub`

---

## 📬 Contact

**Nelson Saravia** — Data Analyst  
[LinkedIn](https://linkedin.com/in/saravianelson) · [Portfolio](https://nelsonsaravia.netlify.app)
