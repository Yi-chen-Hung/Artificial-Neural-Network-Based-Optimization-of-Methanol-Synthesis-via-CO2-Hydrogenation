import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, make_scorer
import matplotlib.pyplot as plt
import joblib
mlp_simulated = joblib.load("Methanol_simulated_ANN.pkl")
mlp_real= joblib.load("Methanol_real_ANN.pkl")
#----------------------------------------------------------------
# Simulated Target parameters
targets = ["Methanol_lag", "CO2_lag", "CO_lag"]  #y
# Top-originals+PCs
Top_PC=Aspen_top_per_pc.get('PC1')+['Methanol_lag','CO2_lag','CO_lag']
Top_plus_PCs=pd.concat([pd.DataFrame(Aspen_X_pca,columns=["PCA1", "PCA2", "PCA3"]), Aspen_data[Top_PC].reset_index(drop=True)],axis=1)
Top_plus_PCs_df=Top_plus_PCs.drop(columns=[col for col in Top_plus_PCs.columns if col in targets])  #X
#Simulated
from sklearn.inspection import permutation_importance
X_train, X_test, y_train, y_test = train_test_split(Top_plus_PCs_df, Top_plus_PCs[targets], test_size=0.2, random_state=42)
# Standardize inputs
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)
mlp_simulated.fit(X_train_scaled, y_train_scaled)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Real Target parameters
targets = ["Methanol_lag", "CO2_lag", "CO_lag"]  #y
# Top-originals+PCs
Top_PC=Real_top_per_pc.get('PC1')+['Methanol_lag','CO2_lag','CO_lag']
Top_plus_PCs=pd.concat([pd.DataFrame(Real_X_pca,columns=["PCA1", "PCA2", "PCA3"]), Real_data[Top_PC].reset_index(drop=True)],axis=1)
Top_plus_PCs_df=Top_plus_PCs.drop(columns=[col for col in Top_plus_PCs.columns if col in targets])  #X
#Real-world
X_train, X_test, y_train, y_test = train_test_split(Top_plus_PCs_df, Top_plus_PCs[targets], test_size=0.2, random_state=42)
# Standardize inputs
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)
mlp_real.fit(X_train_scaled, y_train_scaled)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Method 1 Permutation importance 
def multioutput_mape(y_true, y_pred):
    """
    Vector-valued MAPE: returns one MAPE per output dimension.
    Shape: (n_outputs,)
    """
    # avoid division by zero
    y_true = np.where(y_true == 0, 1e-8, y_true)

    # compute MAPE per output (axis=0 means: mean across samples)
    return np.mean(np.abs((y_true - y_pred) / y_true), axis=0)

mape_scorer = make_scorer(
    multioutput_mape,
    greater_is_better=False
)

perm_importance = permutation_importance(mlp_real, X_test_scaled, y_test_scaled, n_repeats=40, scoring=mape_scorer, random_state=42)

output_names = ['Methanol_selectivity', 'CO2_conversion_rate', 'CO_selectivity']

for i, name in enumerate(output_names):
    globals()[f"importance_{name}"]=pd.DataFrame({
        f"{name}": Top_plus_PCs_df.columns,
        'Importance Mean': perm_importance.importances_mean[:,i],
        'Importance Std': perm_importance.importances_std[:,i]
    })
    globals()[f"importance_{name}"]=globals()[f"importance_{name}"].sort_values('Importance Mean', ascending=False).reset_index(drop=True)
    display(globals()[f"importance_{name}"].head(13))
# Visualization
for name in output_names:
    globals()[f"importance_{name}"].set_index(f'{name}')['Importance Mean'].sort_values().plot.barh(figsize=(8,6), color='teal')
    plt.xlabel('Importance Mean')
    plt.title(f'Feature importance for {name}')
    plt.tight_layout()
    plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Scoreing = R2
import matplotlib.pyplot as plt
feature_counts = [2, 4, 6, 8, 10, 12, 13]
for name in output_names:
    globals()[f"{name}_scores"]=[]
    for n in feature_counts:
        X_sel = pd.DataFrame(X_train_scaled,columns=[Top_plus_PCs_df.columns])[globals()[f"importance_{name}"][name][:n]]
        mlp_simulated.fit(X_sel, y_train_scaled)
        score = mlp_simulated.score(pd.DataFrame(X_test_scaled,columns=[Top_plus_PCs_df.columns])[globals()[f"importance_{name}"][name][:n]], y_test_scaled)
        globals()[f"{name}_scores"].append(score)

for i, name in enumerate(output_names): 
    fig, ax = plt.subplots(figsize=(8,6))
    plt.plot(feature_counts, globals()[f"{name}_scores"], marker='o')
    plt.xlabel('Number of features')
    plt.ylabel('R2 score')
    plt.title(f'{name}_Performance vs feature count')
    for index in range(len(feature_counts)):
      ax.text(feature_counts[index], globals()[f"{name}_scores"][index], round(globals()[f"{name}_scores"][index],4), size=12)
    plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Scoreing = RMSE
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
feature_counts = [3, 6, 9, 12, 13]
for name in output_names:
    globals()[f"{name}_RMSE"]=[]
    for n in feature_counts:
        X_sel = pd.DataFrame(X_train_scaled,columns=[Top_plus_PCs_df.columns])[globals()[f"importance_{name}"][name][:n]]
        mlp_simulated.fit(X_sel, y_train_scaled)
        y_pred_val = mlp_simulated.predict(pd.DataFrame(X_test_scaled,columns=[Top_plus_PCs_df.columns])[globals()[f"importance_{name}"][name][:n]])
        
        y_pred=scaler.inverse_transform(y_pred_val)
        rmse = root_mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i]) 
        globals()[f"{name}_RMSE"].append(rmse)
    fig, ax = plt.subplots(figsize=(8,6))
    plt.plot(feature_counts, globals()[f"{name}_RMSE"], marker='o')
    plt.xlabel('Number of features')
    plt.ylabel('RMSE score')
    plt.title(f'{name} Performance vs feature count')
    for index in range(len(feature_counts)):
      ax.text(feature_counts[index], globals()[f"{name}_RMSE"][index], round(globals()[f"{name}_RMSE"][index],4), size=12)
    plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Scoreing = MAPE
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, mean_squared_error
feature_counts = [2, 4, 6, 8, 10, 12, 13]
for name in output_names:
    globals()[f"{name}_MAPE"]=[]
    for n in feature_counts:
        X_sel = pd.DataFrame(X_train_scaled,columns=[Top_plus_PCs_df.columns])[globals()[f"importance_{name}"][name][:n]]
        mlp_simulated.fit(X_sel, y_train_scaled)
        y_pred_val = mlp_simulated.predict(pd.DataFrame(X_test_scaled,columns=[Top_plus_PCs_df.columns])[globals()[f"importance_{name}"][name][:n]])
        
        y_pred=scaler.inverse_transform(y_pred_val)
        mape = mean_absolute_percentage_error(y_test.iloc[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i]) 
        globals()[f"{name}_MAPE"].append(mape)
    fig, ax = plt.subplots(figsize=(8,6))
    plt.plot(feature_counts, globals()[f"{name}_MAPE"], marker='o')
    plt.xlabel('Number of features')
    plt.ylabel('MAPE score')
    plt.title(f'{name} Performance vs feature count')
    for index in range(len(feature_counts)):
      ax.text(feature_counts[index], globals()[f"{name}_MAPE"][index], round(globals()[f"{name}_MAPE"][index],4), size=12)
    plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Scoreing = MAE
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error
feature_counts = [2, 4, 6, 8, 10, 12, 13]
for name in output_names:
    globals()[f"{name}_MAE"]=[]
    for n in feature_counts:
        X_sel = pd.DataFrame(X_train_scaled,columns=[Top_plus_PCs_df.columns])[globals()[f"importance_{name}"][name][:n]]
        mlp_simulated.fit(X_sel, y_train_scaled)
        y_pred_val = mlp_simulated.predict(pd.DataFrame(X_test_scaled,columns=[Top_plus_PCs_df.columns])[globals()[f"importance_{name}"][name][:n]])
        
        y_pred=scaler.inverse_transform(y_pred_val)
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i]) 
        globals()[f"{name}_MAE"].append(mae)
    fig, ax = plt.subplots(figsize=(8,6))
    plt.plot(feature_counts, globals()[f"{name}_MAE"], marker='o')
    plt.xlabel('Number of features')
    plt.ylabel('MAE score')
    plt.title(f'{name} Performance vs feature count')
    for index in range(len(feature_counts)):
      ax.text(feature_counts[index], globals()[f"{name}_MAE"][index], round(globals()[f"{name}_MAE"][index],3), size=12)
    plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Method 2 SHAP
# Simulated scenario
import shap 
explainer = shap.Explainer(mlp_simulated.predict, pd.DataFrame(X_train_scaled,columns=[Top_plus_PCs_df.columns]))
shap_values = explainer(pd.DataFrame(X_test_scaled,columns=[Top_plus_PCs_df.columns]))
shap_values.values.shape
#real-world scenario
import shap
explainer = shap.Explainer(mlp_real.predict, pd.DataFrame(X_train_scaled,columns=[Top_plus_PCs_df.columns]))
shap_values = explainer(pd.DataFrame(X_test_scaled,columns=[Top_plus_PCs_df.columns]))
shap_values.values.shape
#-----------------------------------------------------------------------------------------------------------------------------------------------------
import shap
import matplotlib.pyplot as plt

output_names = ['Methanol selectivity', 'CO₂ conversion rate', 'CO selectivity']

for i, name in enumerate(output_names):
    print(f"\n--- SHAP beeswarm for {name} ---")
    shap.plots.beeswarm(
        shap_values[..., i],  # select one output
        show=False,
        max_display=13,
    )
    plt.title(f"SHAP Beeswarm – {name}")
    plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
for i, name in enumerate(output_names):
    mean_abs_shap = np.abs(shap_values.values[..., i]).mean(axis=0)
    rank_df = pd.DataFrame({
        'Feature': pd.DataFrame(X_test_scaled,columns=[Top_plus_PCs_df.columns]).columns,
        'Mean |SHAP|': mean_abs_shap
    }).sort_values('Mean |SHAP|', ascending=False).head(12)
    rank_df.set_index('Feature')['Mean |SHAP|'].sort_values().plot.barh(
        figsize=(8,6),
        color='teal'
    )
    plt.xlabel('Mean |SHAP value|')
    plt.title(f'Feature importance for {name}')
    plt.tight_layout()
    plt.show()
    print(f"\nTop 6 features for {name}:")
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------
