import streamlit as st
import pandas as pd
import numpy as np
from model import train_polynomial_model
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt

#  Page setup
st.set_page_config(page_title="🏡 House Price Predictor", layout="wide", page_icon="📈")

#  Custom CSS
st.markdown("""
<style>
html, body, [class*="st-"] {
    font-family: 'Segoe UI', sans-serif;
    color: #f0f0f0;  /* Light gray text */
    background-color: #0e1117;  /* Dark background */
}

h1, h2, h3 {
    color: #9cd1ff;  /* Sky blue headers */
}

.stApp {
    background-color: #0e1117 !important;
}

.metric-container {
    background-color: #161a24;
    padding: 1rem;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


#  Title
st.title("🏡 Boston House Price Predictor")
st.markdown("""
Welcome to the **Boston House Price Predictor** powered by **Polynomial Regression**!  
Use the sliders and inputs to:
- Train a regression model
- Explore residuals and top features
- Predict custom house prices  
""")

#  Degree slider
degree = st.slider("🎚️ Select Polynomial Degree", 1, 5, value=2)

#  Train model and fetch results
model, mse, r2, feature_names = train_polynomial_model(degree)

#  Load the dataset and preprocess for residuals and prediction
df = pd.read_csv("Boston-house-price-data.csv").drop(columns=["AGE_BIN", "LSTAT_BIN"], errors='ignore')

# Add simple engineered features
df['TAX_PER_ROOM'] = df['TAX'] / df['RM']
df['DIS_NOX_RATIO'] = df['DIS'] / df['NOX']
df['IS_HIGH_END'] = (df['MEDV'] > 35).astype(int)

X = df.drop(columns=["MEDV"])
y = df["MEDV"]

# 🔬 Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X_scaled)

#  Predict on entire dataset
y_pred = model.predict(X_poly)
residuals = y - y_pred

#  Evaluation
with st.expander("📊 Model Evaluation", expanded=True):
    col1, col2 = st.columns(2)
    col1.metric("📉 Mean Squared Error", f"{mse:.2f}")
    col2.metric("📈 R² Score", f"{r2:.3f}")

#  Residuals
with st.expander("🧯 Residual Plot"):
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Predicted MEDV")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Predicted Price")
    st.pyplot(fig)

#  Feature importance (Top polynomial terms)
with st.expander("🔥 Top 10 Feature Importances (by Coefficient)"):
    coefs = model.coef_
    terms = poly.get_feature_names_out(feature_names)

    coef_df = pd.DataFrame({
        "Feature": terms[1:],  # Skip intercept
        "Coefficient": coefs[1:]
    }).sort_values(by="Coefficient", key=abs, ascending=False).head(10)

    st.dataframe(coef_df)

#  User Prediction
with st.expander("🎯 Predict Your Own House Price", expanded=True):
    st.markdown("Enter custom values for each feature below:")

    user_inputs = []
    input_cols = st.columns(4)
    for idx, feature in enumerate(feature_names):
        val = float(input_cols[idx % 4].number_input(
            f"{feature}", value=float(X[feature].mean())))
        user_inputs.append(val)

   
    # Transform user input
    user_scaled = scaler.transform([user_inputs[:len(feature_names)]])
    user_poly = poly.transform(user_scaled)
    user_prediction = model.predict(user_poly)[0]

    st.success(f"🏠 **Predicted House Price:** ${user_prediction * 1000:.2f}")
