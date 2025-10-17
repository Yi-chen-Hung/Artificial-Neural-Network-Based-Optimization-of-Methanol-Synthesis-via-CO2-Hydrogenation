# Artifical Neural Network 
#(ANN with 3 hidden layer, Multi-Layer Perceptron for regression, train/test sets, activation function: ReLU)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Target parameters
targets = ["Methanol_lag", "CO2_lag", "CO_lag"]  #y

def evaluate_mlp(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )
    # Standardize inputs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build comparable model
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),  # 3 hidden layers
        activation="relu",
        learning_rate_init=1e-3,
        solver="adam",
        max_iter=2000,
        random_state=42,
        early_stopping=True
    )

    # Train and evaluate
    mlp.fit(X_train_scaled, y_train)
    
    y_pred_val = mlp.predict(X_test_scaled)
    mse_list=[]
    r2_list=[]
    for i in range(len(targets)):
        mse= mean_squared_error(y_test.iloc[:,i], y_pred_val[:,i])
        r2= r2_score(y_test.iloc[:,i], y_pred_val[:,i])
        mse_list.append(mse)
        r2_list.append(r2)
        # print(f"{name:5s} -> Type:  {targets[i]} | Val MSE: {mse:.4f} | Val R²: {r2:.4f}")
        
    return {'Model': name, 'MSE': mse_list, 'R²': r2_list}

#----------------------------------------------------------------------------------------------------------------------------------
# PCs only
PCs_only=pd.concat([pd.DataFrame(X_pca,columns=["PCA1", "PCA2", "PCA3"]),Fdata_profil[['Methanol_lag','CO2_lag','CO_lag']].loc[2:,:].reset_index(drop=True)] ,axis=1) 
PCs_only_df=PCs_only.drop(columns=[col for col in PCs_only.columns if col in targets]) #X

# Top-originals+PCs
Top_PC=top_per_pc.get('PC1')+['Methanol_lag','CO2_lag','CO_lag']
Top_plus_PCs=pd.concat([pd.DataFrame(X_pca,columns=["PCA1", "PCA2", "PCA3"]), Fdata_profil[Top_PC].loc[2:,:].reset_index(drop=True)],axis=1)
Top_plus_PCs_df=Top_plus_PCs.drop(columns=[col for col in Top_plus_PCs.columns if col in targets])  #X

# Unions of Top-originals
Top_PC=list(set(top_per_pc.get('PC1')).union(top_per_pc.get('PC2')).union(top_per_pc.get('PC3')))+['Methanol_lag','CO2_lag','CO_lag']
Top_union=pd.concat([pd.DataFrame(X_pca,columns=["PCA1", "PCA2", "PCA3"]), Fdata_profil[Top_PC].loc[2:,:].reset_index(drop=True)],axis=1)
Top_union_df=Top_union.drop(columns=[col for col in Top_union.columns if col in targets])  #X

#Top_original only
Top_PC=top_per_pc.get('PC1')+['Methanol_lag','CO2_lag','CO_lag']
Top_only=Fdata_profil[Top_PC].loc[2:,:].reset_index(drop=True)
Top_only_df=Top_only.drop(columns=[col for col in Top_only.columns if col in targets])  #X

# Apply different input combination into define function
# ML_methanol(X : Target_df, y : Type[targets])
