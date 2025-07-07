import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_polynomial_model(degree=2):
    # Load cleaned dataset with engineered features
    df = pd.read_csv("Boston-house-price-data.csv")

    # Drop categorical columns only if they exist not suitable for regression directly
    for col in ["AGE_BIN", "LSTAT_BIN"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Features and target
    df['TAX_PER_ROOM'] = df['TAX'] / df['RM']
    df['DIS_NOX_RATIO'] = df['DIS'] / df['NOX']
    df['IS_HIGH_END'] = (df['MEDV'] > 35).astype(int)
    X = df.drop(columns=["MEDV"])
    y = df["MEDV"]


    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Polynomial features
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X_scaled)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2, X.columns.tolist()
