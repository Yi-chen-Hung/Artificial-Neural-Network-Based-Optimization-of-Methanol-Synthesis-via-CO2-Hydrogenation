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
