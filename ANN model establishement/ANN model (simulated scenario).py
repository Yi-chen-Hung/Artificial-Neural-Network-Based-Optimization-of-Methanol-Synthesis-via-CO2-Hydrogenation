import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
# Simulated Target parameters
targets = ["Methanol_lag", "CO2_lag", "CO_lag"]  #y
# Top-originals+PCs optimal choice
Top_PC=Aspen_top_per_pc.get('PC1')+['Methanol_lag','CO2_lag','CO_lag']
Top_plus_PCs=pd.concat([pd.DataFrame(Aspen_X_pca,columns=["PCA1", "PCA2", "PCA3"]), Aspen_data[Top_PC].reset_index(drop=True)],axis=1)
Top_plus_PCs_df=Top_plus_PCs.drop(columns=[col for col in Top_plus_PCs.columns if col in targets])  #X
#--------------------------------------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(Top_plus_PCs_df, Top_plus_PCs[targets], test_size=0.2, random_state=42)
# Standardize inputs
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)

mlp = MLPRegressor(
    hidden_layer_sizes= (64, 32),
    activation="relu",
    learning_rate_init= 0.0005,
    solver="adam",
    max_iter=2000,
    alpha=0.0001,
    random_state=42,
    early_stopping=True
)

# Train and evaluate
mlp.fit(X_train_scaled, y_train_scaled)

y_pred_val = mlp.predict(X_test_scaled)
# Generate customized ANN for simulated scenario 
joblib.dump(mlp,"Methanol_simulated_ANN.pkl")
