# 🏡 Boston House Price Predictor

A Streamlit-based web application that predicts median housing prices in Boston using Polynomial Regression, built from the Boston Housing Dataset. It supports interactive model tuning, residual visualization, and user-defined prediction.

---

## 📆 Project Overview

This project builds a machine learning web app that predicts housing prices using polynomial regression. The interface is built with **Streamlit**, and the model uses **scikit-learn** for preprocessing and training. The app features:

* Polynomial regression model
* Model evaluation metrics: MSE and R²
* Residual plot for model diagnostic
* Feature importance (top coefficients)
* Live prediction from user inputs
* Polynomial degree selector
* CI/CD-ready structure for deployment to **Streamlit Cloud**

---

## 📊 Dataset: Boston Housing Data

This dataset contains information collected by the U.S. Census Service concerning housing in the area of Boston, Massachusetts. It includes 506 instances and 14 attributes (13 features + target).

**Target Variable:**

* `MEDV`: Median value of owner-occupied homes in \$1000s

**Features Used:**

| Feature | Description                                            |
| ------- | ------------------------------------------------------ |
| CRIM    | Crime rate per capita                                  |
| ZN      | Proportion of residential land zoned                   |
| INDUS   | Non-retail business acres per town                     |
| CHAS    | Charles River dummy variable (1 if tract bounds river) |
| NOX     | Nitric oxides concentration (parts per 10 million)     |
| RM      | Average number of rooms per dwelling                   |
| AGE     | Proportion of owner-occupied units built prior to 1940 |
| DIS     | Weighted distances to employment centers               |
| RAD     | Index of accessibility to radial highways              |
| TAX     | Full-value property-tax rate per \$10,000              |
| PTRATIO | Pupil-teacher ratio by town                            |
| B       | 1000(Bk - 0.63)^2 (where Bk is % of Black residents)   |
| LSTAT   | % lower status population                              |

---

## 🚀 Model and Features

### Preprocessing

* StandardScaler is used to normalize all features to zero mean and unit variance.
* PolynomialFeatures is used to expand the feature space to include higher-order and interaction terms.

### Model

* `sklearn.linear_model.LinearRegression` is used.
* Model is trained on a split of 80% training and 20% test data.
* Model evaluation uses:

  * **Mean Squared Error (MSE)**
  * **R² Score (Variance Explained)**

---

## 🌍 Web App UI (Streamlit)

### Sidebar / Controls

* Slider to select polynomial degree (1 to 5)
* Expanders for:

  * Model evaluation
  * Residual plot
  * Feature importance (top 10 terms by coefficient)
  * User-defined prediction inputs

### Output

* Real-time metrics update
* Matplotlib residual plot
* Top coefficients table
* House price prediction from user input

---

## 🛋️ Files in the Project

```
.
├── app.py                  # Streamlit frontend UI
├── model.py                # Model training and return function
├── requirements.txt        # Python dependencies
├── Boston-house-price-data.csv # Dataset file (13 features + MEDV)
└── .github
    └── workflows
        └── deploy.yml      # GitHub Actions for Streamlit Cloud deploy
```

---

## 🎓 How to Run Locally

1. **Install Python 3.10+**
2. Clone the repository:

   ```bash
   git clone https://github.com/your-username/boston-house-price-predictor.git
   cd boston-house-price-predictor
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run Streamlit app:

   ```bash
   streamlit run app.py
   ```

---
## 🚫 Limitations

* Assumes feature relationships are polynomial in nature.
* Does not include regularization (e.g., Ridge/Lasso).
* Dataset is static and not updated with real-world values.

---

## 🤝 Credits

* Dataset: UCI Machine Learning Repository
* UI: [Streamlit](https://streamlit.io/)
* ML: [scikit-learn](https://scikit-learn.org/)

---

## 📊 Future Improvements

* Add Ridge/Lasso models
* Include SHAP explainability
* Add dataset uploader for user data
* Export predictions as CSV

---

Made with ❤️ by Mansoob E Zehra
