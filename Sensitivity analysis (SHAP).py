# Read ANN machine learning model
import joblib
mlp = joblib.load("Methanol_synthesis_ANN.pkl")
#---------------------------------------------------------------------------------------------------------------------------------------------------
# Target parameters
targets = ["Methanol_lag", "CO2_lag", "CO_lag"]  #y
# Unions of Top-originals
Top_PC=list(set(top_per_pc.get('PC1')).union(top_per_pc.get('PC2')).union(top_per_pc.get('PC3')))+['Methanol_lag','CO2_lag','CO_lag']
Top_union=Fdata_total[Top_PC].reset_index(drop=True)
Top_union_df=Top_union.drop(columns=[col for col in Top_union.columns if col in targets])  #X
#---------------------------------------------------------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(Top_union_df, Top_union[targets], test_size=0.2, random_state=42)
# Standardize inputs
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)
#---------------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.inspection import permutation_importance
mlp.fit(X_train_scaled, y_train_scaled)  # <-- not X_train.values
result = permutation_importance(mlp, X_test_scaled, y_test_scaled, n_repeats=10, random_state=0)
importance_df = pd.DataFrame({'feature': Top_union_df.columns, 'importance': result.importances_mean})
importance_df=importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
#---------------------------------------------------------------------------------------------------------------------------------------------------
# Scoreing = R2
import matplotlib.pyplot as plt

feature_counts = [3, 6, 9, 12, 15, 18, 21, 24, 26]
scores = []

for n in feature_counts:
    X_sel = pd.DataFrame(X_train_scaled,columns=[Top_union_df.columns])[importance_df['feature'][:n]]
    mlp.fit(X_sel, y_train_scaled)
    score = mlp.score(pd.DataFrame(X_test_scaled,columns=[Top_union_df.columns])[importance_df['feature'][:n]], y_test_scaled)
    scores.append(score)

fig, ax = plt.subplots(figsize=(8,6))
plt.plot(feature_counts, scores, marker='o')
plt.xlabel('Number of features')
plt.ylabel('MAPE score')
plt.title('Model performance vs feature count')
for index in range(len(feature_counts)):
  ax.text(feature_counts[index], scores[index], round(scores[index],4), size=12)
plt.show()
#---------------------------------------------------------------------------------------------------------------------------------------------------
# Scoreing = MAPE and plot 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
feature_counts = [3, 6, 9, 12, 15, 18, 21, 24, 26]
scores_Methanol_lag= []
scores_CO2_lag=[]
scores_CO_lag=[]

for n in feature_counts:
    X_sel = pd.DataFrame(X_train_scaled,columns=[Top_union_df.columns])[importance_df['feature'][:n]]
    mlp.fit(X_sel, y_train_scaled)
    y_pred_val = mlp.predict(pd.DataFrame(X_test_scaled,columns=[Top_union_df.columns])[importance_df['feature'][:n]])
    n_targets = y_test_scaled.shape[1]
    
    for i in range(n_targets):
        mae = mean_absolute_error(y_test_scaled[:, i], y_pred_val[:, i])
        mape = (mae / np.mean(np.abs(y_test_scaled[:, i]))) * 100  
        globals()[f"scores_{targets[i]}"].append(mape)
for ty in targets:
    fig, ax = plt.subplots(figsize=(8,6))
    plt.plot(feature_counts, globals()[f"scores_{ty}"], marker='o')
    plt.xlabel('Number of features')
    plt.ylabel(f'{ty}_MAPE score')
    plt.title('Model performance vs feature count')
    for index in range(len(feature_counts)):
      ax.text(feature_counts[index], globals()[f"scores_{ty}"][index], round(globals()[f"scores_{ty}"][index],2), size=12)
    plt.show()
#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------
import shap
explainer = shap.Explainer(mlp.predict, pd.DataFrame(X_train_scaled,columns=[Top_union_df.columns]))
shap_values = explainer(pd.DataFrame(X_test_scaled,columns=[Top_union_df.columns]))
shap_values.values.shape
#---------------------------------------------------------------------------------------------------------------------------------------------------
# Beeswarm plot
import shap
import matplotlib.pyplot as plt

output_names = ['Methanol selectivity', 'CO₂ conversion', 'CO selectivity']

for i, name in enumerate(output_names):
    print(f"\n--- SHAP beeswarm for {name} ---")
    shap.plots.beeswarm(
        shap_values[..., i],  # select one output
        show=False,
    )
    plt.title(f"SHAP Beeswarm – {name}")
    plt.show()
#---------------------------------------------------------------------------------------------------------------------------------------------------
# Mean SHAP plot
import numpy as np
import pandas as pd

for i, name in enumerate(output_names):
    mean_abs_shap = np.abs(shap_values.values[..., i]).mean(axis=0)
    rank_df = pd.DataFrame({
        'Feature': pd.DataFrame(X_test_scaled,columns=[Top_union_df.columns]).columns,
        'Mean |SHAP|': mean_abs_shap
    }).sort_values('Mean |SHAP|', ascending=False).head(10)
    rank_df.set_index('Feature')['Mean |SHAP|'].sort_values().plot.barh(
        figsize=(8,6),
        color='teal'
    )
    plt.xlabel('Mean |SHAP value|')
    plt.title(f'Feature importance for {name}')
    plt.tight_layout()
    plt.show()
    print(f"\nTop 10 features for {name}:")
