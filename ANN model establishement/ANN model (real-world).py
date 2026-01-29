import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
X_train, X_test, y_train, y_test = train_test_split(Top_plus_PCs_df, Top_plus_PCs[targets], test_size=0.2, random_state=42)
# Standardize inputs
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)

mlp = MLPRegressor(
    hidden_layer_sizes= (128, 64, 32),
    activation="relu",
    learning_rate_init= 0.0001,
    solver="adam",
    max_iter=2000,
    alpha=0.0001,
    random_state=42,
    early_stopping=True
)

# Train and evaluate
mlp.fit(X_train_scaled, y_train_scaled)

y_pred_val = mlp.predict(X_test_scaled)
# Generate customized ANN for real-world scenario 
joblib.dump(mlp,"Methanol_real_ANN.pkl")
