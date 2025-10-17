# Example for multiple input sets
results = []
results.append(evaluate_mlp(PCs_only_df, PCs_only[targets], "PCs only"))
results.append(evaluate_mlp(Top_plus_PCs_df, Top_plus_PCs[targets], "PCs + top originals"))
results.append(evaluate_mlp(Top_union_df, Top_union[targets], "Unions of Top-originals"))
results.append(evaluate_mlp(Top_only_df, Top_only[targets], "Top originals only"))
results=pd.DataFrame(results)
MSE=pd.DataFrame(results['MSE'].tolist(), columns=["Methanol_selectivity_MSE", "CO2_conversion_rate_MSE", "CO_selectivity_MSE"])
R2=pd.DataFrame(results['R²'].tolist(), columns=["Methanol_selectivity_R²", "CO2_conversion_rate_R²", "CO_selectivity_R²"])
display(pd.concat([results, MSE], axis=1).drop(columns=['MSE','R²']))
display(pd.concat([results, R2], axis=1).drop(columns=['MSE','R²']))
