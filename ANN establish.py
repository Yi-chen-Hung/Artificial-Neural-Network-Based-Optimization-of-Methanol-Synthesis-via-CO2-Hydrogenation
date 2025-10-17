# Artifical Neural Network 
#(ANN with 3 hidden layer, Multi-Layer Perceptron for regression, train/test sets, activation function: ReLU)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def ML_methanol(X,y):
    # =====================
    # 1. Train/Test split
    # =====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # =====================
    # 2. Scaling
    # =====================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # =====================
    # 3. Build and Train MLPRegressor
    # =====================
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),  # 3 hidden layers
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42,
        early_stopping=True
    )
    
    mlp.fit(X_train_scaled, y_train)
    
    # =====================
    # 4. Evaluate
    # =====================
    y_pred = mlp.predict(X_test_scaled)
    Analysis_df=pd.DataFrame(index=targets, columns=['RMSE','R²'])
    for i, target in enumerate(targets):
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        Analysis_df.loc[target,'RMSE']=round(rmse,3)
        Analysis_df.loc[target, 'R²']=round(r2,3)
    
    
    # =====================
    # 5. Parity plots
    # =====================
    for i, target in enumerate(targets):
        plt.figure(figsize=(5,5))
        plt.scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.7)
        plt.plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
                 [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
                 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Parity plot for {target}")
    
    return Analysis_df, plt.show()

#----------------------------------------------------------------------------------------------------------------------------------
# Target parameters
targets = ["Methanol_lag", "CO2_lag", "CO_lag"]  #y

# PCs only
Target_df=pd.concat([pd.DataFrame(X_pca,columns=["PCA1", "PCA2", "PCA3"]),Fdata_profil[['Methanol_lag','CO2_lag','CO_lag']].loc[2:,:].reset_index(drop=True)] ,axis=1) 
Target_df.drop(columns=[col for col in Target_df.columns if col in targets])  #X

# Top-originals+PCs
Top_PC=top_per_pc.get('PC1')+['Methanol_lag','CO2_lag','CO_lag']
Target_df=pd.concat([pd.DataFrame(X_pca,columns=["PCA1", "PCA2", "PCA3"]), Fdata_profil[Top_PC].loc[2:,:].reset_index(drop=True)],axis=1)
Target_df.drop(columns=[col for col in Target_df.columns if col in targets])  #X

# Unions of Top-originals
Top_PC=list(set(top_per_pc.get('PC1')).union(top_per_pc.get('PC2')).union(top_per_pc.get('PC3')))+['Methanol_lag','CO2_lag','CO_lag']
Target_df=pd.concat([pd.DataFrame(X_pca,columns=["PCA1", "PCA2", "PCA3"]), Fdata_profil[Top_PC].loc[2:,:].reset_index(drop=True)],axis=1)
Target_df.drop(columns=[col for col in Target_df.columns if col in targets])  #X

#Top_original only
Top_PC=top_per_pc.get('PC1')+['Methanol_lag','CO2_lag','CO_lag']
Target_df=Fdata_profil[Top_PC].loc[2:,:].reset_index(drop=True)
Target_df.drop(columns=[col for col in Target_df.columns if col in targets])  #X

# Apply different input combination to define function
ML_methanol(X : Target_df.drop(columns=[col for col in Target_df.columns if col in targets]), y : Target_df[targets])
