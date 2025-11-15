# Deciding the most feasible Hyperparameter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

for Typ in ('PCs_only','Top_plus_PCs','Top_union','Top_only'):

    X_train, X_test, y_train, y_test = train_test_split(
        globals()[f"{Typ}_df"], globals()[f"{Typ}"][targets], test_size=0.2, random_state=42)
    # Standardize inputs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_scaled = scaler.fit_transform(y_train)
    y_test_scaled = scaler.transform(y_test)
    
    param_grid = {
        'hidden_layer_sizes': [(128, 64, 32), (64, 32), (64,)],
        'alpha': [1e-5, 1e-4, 1e-3],
        'learning_rate_init': [1e-3, 5e-4, 1e-4]
    }
    
    grid = GridSearchCV(
        estimator=MLPRegressor(
            activation='relu', solver='adam', early_stopping=True,
            validation_fraction=0.2, random_state=42, max_iter=500
        ),
        param_grid=param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train_scaled, y_train_scaled)
    print(f"Best parameters of {Typ}:", grid.best_params_)
#------------------------------------------------------------------------------------------------------------------------------------
# Example for multiple input sets (profil + osci)
results = []
results.append(evaluate_mlp(PCs_only_df, PCs_only[targets], 0.001, (128, 64, 32), 0.0005, "PCs only"))
results.append(evaluate_mlp(Top_plus_PCs_df, Top_plus_PCs[targets], 0.0001, (128, 64, 32), 0.0005, "PCs + top originals"))
results.append(evaluate_mlp(Top_union_df, Top_union[targets], 0.001, (128, 64, 32), 0.001, "Unions of Top-originals"))
results.append(evaluate_mlp(Top_only_df, Top_only[targets], 1e-05, (128, 64, 32), 0.001, "Top originals only"))
results=pd.DataFrame(results)
MAPE=pd.DataFrame(results['MAPE'].tolist(), columns=["Methanol_selectivity_MAPE", "CO2_conversion_rate_MAPE", "CO_selectivity_MAPE"])
R2=pd.DataFrame(results['R²'].tolist(), columns=["Methanol_selectivity_R²", "CO2_conversion_rate_R²", "CO_selectivity_R²"])
display(pd.concat([results, MAPE], axis=1).drop(columns=['MAPE','R²']))
display(pd.concat([results, R2], axis=1).drop(columns=['MAPE','R²']))
