# Artifical Neural Network 
#(ANN with given hidden layer, initial learning rate, alpha, Multi-Layer Perceptron for regression, train/test sets, activation function: ReLU)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Target parameters
targets = ["Methanol_lag", "CO2_lag", "CO_lag"]  #y

def evaluate_mlp(X, y, alpha, hidden_layer_sizes, learning_rate_init, name):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )
    # Standardize inputs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_scaled = scaler.fit_transform(y_train)
    y_test_scaled = scaler.transform(y_test)

    # Build comparable model
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        learning_rate_init=learning_rate_init,
        solver="adam",
        max_iter=2000,
        alpha=alpha,
        random_state=42,
        early_stopping=True
    )

    # Train and evaluate
    mlp.fit(X_train_scaled, y_train_scaled)
    
    y_pred_val = mlp.predict(X_test_scaled)
    mape_list = []  # optional, if you want percentage
    r2_list = []
    
    n_targets = y_test_scaled.shape[1]
    
    for i in range(n_targets):
        mae = mean_absolute_error(y_test_scaled[:, i], y_pred_val[:, i])
        r2  = r2_score(y_test_scaled[:, i], y_pred_val[:, i])
        mape = (mae / np.mean(np.abs(y_test_scaled[:, i]))) * 100  # optional
        mape_list.append(mape)
        r2_list.append(r2)
        
    return {'Model': name, 'MAPE': mape_list, 'R²': r2_list}

#----------------------------------------------------------------------------------------------------------------------------------
# PCs only
PCs_only=pd.concat([pd.DataFrame(X_pca,columns=["PCA1", "PCA2", "PCA3"]),Fdata_total[['Methanol_lag','CO2_lag','CO_lag']].reset_index(drop=True)] ,axis=1) 
PCs_only_df=PCs_only.drop(columns=[col for col in PCs_only.columns if col in targets]) #X

# Top-originals+PCs
Top_PC=top_per_pc.get('PC1')+['Methanol_lag','CO2_lag','CO_lag']
Top_plus_PCs=pd.concat([pd.DataFrame(X_pca,columns=["PCA1", "PCA2", "PCA3"]), Fdata_total[Top_PC].reset_index(drop=True)],axis=1)
Top_plus_PCs_df=Top_plus_PCs.drop(columns=[col for col in Top_plus_PCs.columns if col in targets])  #X

# Unions of Top-originals
Top_PC=list(set(top_per_pc.get('PC1')).union(top_per_pc.get('PC2')).union(top_per_pc.get('PC3')))+['Methanol_lag','CO2_lag','CO_lag']
Top_union=Fdata_total[Top_PC].reset_index(drop=True)
Top_union_df=Top_union.drop(columns=[col for col in Top_union.columns if col in targets])  #X

#Top_original only
Top_PC=top_per_pc.get('PC1')+['Methanol_lag','CO2_lag','CO_lag']
Top_only=Fdata_total[Top_PC].reset_index(drop=True)
Top_only_df=Top_only.drop(columns=[col for col in Top_only.columns if col in targets])  #X
