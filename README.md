## Overview

This project leverages machine learning to predict customer conversions in digital marketing campaigns. By analyzing demographics, campaign engagement metrics, and purchase history, the model identifies high-probability converters, enabling marketing teams to target effectively and optimize campaign ROI. The project also includes a **Streamlit web application** for real-time predictions.

---

## Problem Statement

The goal is to build a predictive model that identifies which customers are most likely to convert. This allows marketing teams to prioritize high-value customers, reduce wasted ad spend, and improve overall campaign performance.

---

## Project Workflow

1. **Data Preprocessing:** Handling nulls, encoding categorical variables, treating outliers.  
2. **Exploratory Data Analysis (EDA):** Generating visuals and insights from campaign and customer data.  
3. **Feature Engineering:** Creating new features to enhance model performance.  
4. **Model Training:** Training Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting classifiers.  
5. **Model Evaluation:** Assessing models using Accuracy, Precision, Recall, F1 Score, ROC-AUC Score, Confusion Matrix, and Classification Report.

---

## Technical Stack

- **Programming Language:** Python 3.11  
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Streamlit  
- **IDE/Tools:** VS Code, Jupyter Notebook, Power BI for dashboards  
- **Model File:** `conversion_pipeline.joblib`  

---

## Model Evaluation Summary

Four classification models were trained and compared using:  

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC Score  
- Confusion Matrix  
- Classification Report  

> **Note:** Given the class imbalance (many more converters than non-converters), the focus was on **recall** and **F1-score**, especially for the minority class (non-converters).

---

## Key Observations

- **Accuracy alone can be misleading** — many models predicted converters well but failed to detect non-converters.  
- Logistic Regression and Random Forest showed **high recall for Class 1** but very low recall for Class 0.  
- **Gradient Boosting** delivered the best balance:  

| Metric | Value |
|--------|-------|
| Accuracy | 92% |
| Precision (Class 1) | 0.92 |
| Recall (Class 1) | 0.99 |
| Recall (Class 0) | 0.40 |
| F1 Score (Class 0) | 0.55 |
| ROC-AUC | 0.84 |

---

## Final Conclusion

Gradient Boosting was the most effective model, providing balanced performance across both customer segments. Its superior ability to detect both likely and unlikely converters makes it the best choice for deployment in marketing campaigns.  

**Benefits:**  
- More efficient campaign targeting  
- Increased ROI  
- Improved customer segmentation strategies  

---

**Live Demo:** [Streamlit App](https://digital-marketing-campaign-conversion-prediction-e8vymcou3wpqm.streamlit.app/)
