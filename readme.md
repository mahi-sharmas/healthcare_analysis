## Healthcare Data Analysis & Length of Stay Prediction

An end-to-end exploratory data analysis of 55,500 patient records covering demographics, billing, and clinical outcomes — with a Random Forest regression model to predict hospital length of stay.

### Highlights

- Analyzed 55,500 patient records across 15 features covering demographics, billing, medical conditions, and outcomes
- Performed comprehensive EDA with 9 visualizations uncovering patterns in age distribution, gender balance, billing trends, and condition prevalence
- Built a predictive model identifying Billing Amount (46.7%) and Age (22.1%) as the top drivers of hospital length of stay
- Engineered features including Length of Stay calculation from dates and Age Group segmentation into 5 bins for deeper cohort analysis

### Problem Statement

Understanding patient demographics, billing patterns, and factors that influence hospital stay duration is essential for healthcare operations and resource planning. This project performs a thorough exploratory analysis of a large healthcare dataset to uncover actionable trends, then builds a Random Forest model to predict length of stay — a key metric for hospital capacity management, staffing, and cost forecasting.

### Dataset

- **Source:** [Kaggle Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)
- **Size:** 55,500 patient records × 15 columns (16 after feature engineering)
- **Engineered target:** `Length of Stay` (days) = Discharge Date − Date of Admission
- **Key features:** Age (mean 51.5, range 13–89), Gender (50/50 split), Blood Type (8 types), Medical Condition (6 conditions, top: Arthritis with 9,308 cases), Billing Amount (mean $25,539), Admission Type (3 types), Insurance Provider (5 providers, top: Cigna with 11,249), Medication, Test Results
- **Data quality:** Zero missing values across all 15 columns

### Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data-green)
![NumPy](https://img.shields.io/badge/NumPy-Compute-yellow)
![Seaborn](https://img.shields.io/badge/Seaborn-Viz-teal)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plots-red)

### Methodology

1. **Data Loading & Profiling** — Imported 55,500 records, verified shape (15 columns), inspected data types and completeness (zero missing values)
2. **Data Cleaning** — Converted Date of Admission and Discharge Date to datetime, cleaned Billing Amount by stripping non-numeric characters, validated Age (0–120 range) and Length of Stay (>0)
3. **Feature Engineering** — Calculated `Length of Stay` from admission/discharge dates, created 5 Age Group bins (0–18, 18–35, 35–50, 50–65, 65+)
4. **Exploratory Data Analysis** — Generated 9 visualizations: histograms with KDE (Age, Billing Amount, Length of Stay), countplots (Gender), bar charts (Top 10 Medical Conditions), boxplots (Billing by Condition), grouped analysis (Age Group × Gender), and missing values heatmap
5. **Preprocessing Pipeline** — Built a ColumnTransformer with StandardScaler for numerical features (Age, Billing Amount) and OneHotEncoder for categorical features (Gender, Medical Condition, Admission Type, Insurance Provider)
6. **Modeling** — Trained Random Forest Regressor (100 estimators, `random_state=42`) on 80/20 train-test split
7. **Evaluation** — Computed MAE, RMSE, R² metrics and extracted top 10 feature importances

### Key Results

**Descriptive Insights:**

| Insight | Value |
|---|---|
| Average patient age | 51.5 years (std: 19.6, range: 13–89) |
| Average billing amount | $25,539 (std: $14,211, range: −$2,008 to $52,764) |
| Average length of stay | 15.5 days (std: 8.66, range: 1–30) |
| Gender distribution | Near-even — Male: 27,774 (50.04%) / Female: 27,726 (49.96%) |
| Most common condition | Arthritis (9,308 cases) |
| Top insurer | Cigna (11,249 patients) |
| Most common admission type | Elective (18,655) |

**Predictive Model (Random Forest Regressor):**

| Metric | Value |
|---|---|
| Mean Absolute Error (MAE) | 7.28 days |
| Root Mean Squared Error (RMSE) | 8.60 days |
| R² Score | 0.006 |

**Top 5 Feature Importances:** Billing Amount (46.7%), Age (22.1%), Admission Type — Elective (2.2%), Admission Type — Emergency (2.1%), Medical Condition — Arthritis (2.1%)

The low R² (0.006) indicates that length of stay in this dataset has a near-uniform distribution and is not well-predicted by the available features alone — suggesting that clinical severity or other unmeasured factors are the true drivers. The EDA insights and end-to-end ML pipeline architecture remain the primary value of this project.

### How to Run

```bash
git clone https://github.com/mahi-sharmas/healthcare_analysis.git
cd healthcare_analysis
pip install -r requirements.txt
jupyter notebook eda_healthcare.ipynb
```

### Project Structure

```
healthcare_analysis/
├── eda_healthcare.ipynb    # Full analysis — EDA (9 visualizations), preprocessing, modeling
├── archive.zip             # Compressed healthcare dataset (55,500 records)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

### Future Improvements

- Incorporate clinical severity scores (e.g., APACHE, Charlson Comorbidity Index) as features to improve predictive power beyond the available demographics
- Apply unsupervised clustering (K-Means, DBSCAN) to identify distinct patient segments with different stay and billing patterns
- Build an interactive Streamlit dashboard for hospital administrators to explore trends by condition, insurer, age group, and admission type

### Author

**Mahi Sharma** — B.Tech CSE (Data Science), Manipal University Jaipur (2023–2027)

GitHub: [github.com/mahi-sharmas](https://github.com/mahi-sharmas) | Email: mahi.sh4rma7@gmail.com
