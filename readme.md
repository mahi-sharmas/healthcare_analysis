# Healthcare Data Analysis & Length-of-Stay Prediction

An end-to-end EDA and predictive modeling project on 55,500 patient records. Explores billing patterns, medical conditions, and demographics, then builds a Random Forest model to predict hospital length of stay.

## Problem Statement

Hospitals need to forecast patient length of stay (LOS) for resource planning, staffing, and cost management. This project performs comprehensive exploratory analysis on healthcare data and builds a predictive model to estimate LOS based on patient demographics, medical conditions, and billing information.

## Dataset

- **Source:** [Kaggle Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)
- **Records:** 55,500 patients
- **Features (15):** `Name`, `Age`, `Gender`, `Blood Type`, `Medical Condition`, `Date of Admission`, `Doctor`, `Hospital`, `Insurance Provider`, `Billing Amount`, `Room Number`, `Admission Type`, `Discharge Date`, `Medication`, `Test Results`
- **Engineered Feature:** `Length of Stay (days)` — computed from admission and discharge dates

## Approach

### Part 1: Exploratory Data Analysis

1. **Data Profiling** — 55,500 records, 15 features, zero missing values
2. **Data Cleaning** — Parsed dates, validated ages (13–89 range), computed LOS
3. **Distribution Analysis** — Age (mean 51.5, range 13–89), billing (mean $25,539), LOS (mean 15.5 days)
4. **Categorical Breakdown** — Gender split (50/50), 6 medical conditions, 3 admission types, 5 insurance providers
5. **Cross-Analysis** — Billing by condition, age groups by gender, medical condition frequencies

### Part 2: Predictive Modeling

1. **Feature Selection** — `Age`, `Gender`, `Medical Condition`, `Admission Type`, `Insurance Provider`, `Billing Amount`
2. **Preprocessing Pipeline** — StandardScaler for numeric, OneHotEncoder for categorical via ColumnTransformer
3. **Model:** Random Forest Regressor (100 estimators)
4. **Train/Test Split:** 80/20

## Key Results

### EDA Insights

| Insight | Value |
|---|---|
| Most Expensive Condition (avg billing) | Obesity |
| Longest Average Stay | Asthma |
| Average Patient Age | 51.5 years |
| Gender Distribution | 50/50 (Male: 27,774 / Female: 27,726) |
| Average Billing Amount | $25,539 |
| Average Length of Stay | 15.5 days |

### Predictive Model Performance

| Metric | Value |
|---|---|
| MAE | 7.28 days |
| RMSE | 8.60 days |
| R² Score | 0.006 |

### Top Feature Importances

| Feature | Importance |
|---|---|
| Billing Amount | 0.4667 |
| Age | 0.2212 |
| Admission Type (Elective) | 0.0218 |
| Admission Type (Emergency) | 0.0211 |
| Medical Condition (Arthritis) | 0.0207 |

**Note:** The low R² (0.006) indicates that LOS in this synthetic dataset has minimal predictable signal from the available features. The EDA and pipeline architecture remain valuable as a demonstration of end-to-end ML workflow.

## Tech Stack

- **Language:** Python
- **Libraries:** Scikit-learn, Pandas, NumPy, Seaborn, Matplotlib
- **Techniques:** EDA, Random Forest Regression, OneHotEncoding, StandardScaler, Pipeline, Feature Importance Analysis

## Project Structure

```
├── eda_healthcare.ipynb   # Full EDA + modeling notebook
├── archive.zip            # Dataset
├── requirements.txt       # Dependencies
└── README.md
```

## How to Run

```bash
unzip archive.zip
jupyter notebook eda_healthcare.ipynb
```

## Author

**Mahi Sharma**
B.Tech CSE (Data Science) — Manipal University Jaipur
[GitHub](https://github.com/mahi-sharmas)
