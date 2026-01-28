import pandas as pd
import numpy as np
# Read all five types different hydrogen input type raw data
for typ in ['insert_up','osci','profil','rauf','runter']:
    globals()[f"Fdata_{typ}"]=pd.read_excel(f'Fdata_{typ}.xlsx').dropna() 
# Connect two differen pattern of Hydrogen input data 
Fdata_total=pd.DataFrame()
for typ in ['profil','osci']:
    Fdata_total=pd.concat([Fdata_total,globals()[f"Fdata_{typ}"]],axis=0)
#Establish two different scenario by input type (Simulated & real-world)
Aspen_data=Fdata_total[Fdata_total.columns.drop(list(Fdata_total.filter(regex='%')))].drop(columns=['Time']) # Aspen+ data type
Aspen_correlation=Fdata_total[Fdata_total.columns.drop(list(Fdata_total.filter(regex='%')))].drop(columns=['Time']) # for Aspen+ correlation
Real_data=Fdata_total[Fdata_total.columns.drop(list(Fdata_total.filter(regex='%|Zn|Fcn')))].drop(columns=['Time'])  # Simulate Real life accessible data type
Real_correlation=Fdata_total[Fdata_total.columns.drop(list(Fdata_total.filter(regex='%|Zn|Fcn')))].drop(columns=['Time'])  # for Real life correlation
#--------------------------------------------------------------------------------------------------------------------------------------------
# calculate correlationship between target parameter and every other columns for two scenario
def Correlation(df,parameter):
    list1=list(df.sort_values(parameter, ascending=True)[parameter][np.abs(df[parameter])>0.5].index)
    return list1
targets = ['Methanol_lag', 'CO2_lag', 'CO_lag']
Aspen_corr = Aspen_correlation.corr()[targets]
Real_corr = Real_correlation.corr()[targets]
corr1=Aspen_corr[~Aspen_corr.index.str.contains('lag')].dropna()
corr2=Real_corr[~Real_corr.index.str.contains('lag')].dropna()

Aspen_selected_features=list(set(Correlation(corr1,'Methanol_lag')).union(Correlation(corr1,'CO2_lag')).union(Correlation(corr1,'CO_lag')))
Real_selected_features=list(set(Correlation(corr2,'Methanol_lag')).union(Correlation(corr2,'CO2_lag')).union(Correlation(corr2,'CO_lag')))

print("Aspen Selected_feature:",len(Aspen_selected_features))
print("Real life Selected_feature:",len(Real_selected_features))
#--------------------------------------------------------------------------------------------------------------------------------------------
# Target parameters PCA analysis in simulated scenario
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
PCA_df=Aspen_data[Aspen_selected_features]

X_scaled = scaler.fit_transform(PCA_df)
pca = PCA()
Aspen_X_pca = pca.fit_transform(X_scaled)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Simulated envrionment - PCA")
plt.show()

X_scaled = scaler.fit_transform(PCA_df)
pca = PCA()
pca = PCA(n_components=0.95) # explain 95% variance
Aspen_X_pca = pca.fit_transform(X_scaled)

print("Explained amount of variance:", "95%")
print("Number of components chosen:", pca.n_components_)
print("Explained variance ratio:", pca.explained_variance_ratio_)

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=PCA_df.columns
)
Aspen_top_per_pc = {f'PC{i+1}': loadings[f'PC{i+1}'].abs().sort_values(ascending=False).head(10).index.tolist()
              for i in range(pca.n_components_)}
pd.DataFrame(Aspen_top_per_pc)
#--------------------------------------------------------------------------------------------------------------------------------------------
# Target parameters PCA analysis in Real world environment
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
PCA_df=Real_data[Real_selected_features]

X_scaled = scaler.fit_transform(PCA_df)
pca = PCA()
Real_X_pca = pca.fit_transform(X_scaled)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Real world envrionment - PCA")
plt.show()

pca = PCA()
pca = PCA(n_components=0.95) # explain 95% variance
Real_X_pca = pca.fit_transform(X_scaled)

print("Explained amount of variance:", "95%")
print("Number of components chosen:", pca.n_components_)
print("Explained variance ratio:", pca.explained_variance_ratio_)

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=PCA_df.columns
)
Real_top_per_pc = {f'PC{i+1}': loadings[f'PC{i+1}'].abs().sort_values(ascending=False).head(10).index.tolist()
              for i in range(pca.n_components_)}
pd.DataFrame(Real_top_per_pc)
#------------------------
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(PCA_df)
pca = PCA(n_components=0.95) # explain 95% variance
X_pca = pca.fit_transform(X_scaled)

print("Explained amount of variance:", "95%")
print("Number of components chosen:", pca.n_components_)
print("Explained variance ratio:", pca.explained_variance_ratio_)
#--------------------------------------------------------------------------------------------------------------------------------------------
# Simulated environment Target parameters
targets = ["Methanol_lag", "CO2_lag", "CO_lag"]  #y
# PCs only
PCs_only=pd.concat([pd.DataFrame(Aspen_X_pca,columns=["PCA1", "PCA2", "PCA3"]),Aspen_data[['Methanol_lag','CO2_lag','CO_lag']].reset_index(drop=True)] ,axis=1) 
PCs_only_df=PCs_only.drop(columns=[col for col in PCs_only.columns if col in targets]) #X

# Top-originals+PCs
Top_PC=Aspen_top_per_pc.get('PC1')+['Methanol_lag','CO2_lag','CO_lag']
Top_plus_PCs=pd.concat([pd.DataFrame(Aspen_X_pca,columns=["PCA1", "PCA2", "PCA3"]), Aspen_data[Top_PC].reset_index(drop=True)],axis=1)
Top_plus_PCs_df=Top_plus_PCs.drop(columns=[col for col in Top_plus_PCs.columns if col in targets])  #X

# Unions of Top-originals
Top_PC=list(set(Aspen_top_per_pc.get('PC1')).union(Aspen_top_per_pc.get('PC2')).union(Aspen_top_per_pc.get('PC3')))+['Methanol_lag','CO2_lag','CO_lag']
Top_union=Aspen_data[Top_PC].reset_index(drop=True)
Top_union_df=Top_union.drop(columns=[col for col in Top_union.columns if col in targets])  #X

#Top_original only
Top_PC=Aspen_top_per_pc.get('PC1')+['Methanol_lag','CO2_lag','CO_lag']
Top_only=Aspen_data[Top_PC].reset_index(drop=True)
Top_only_df=Top_only.drop(columns=[col for col in Top_only.columns if col in targets])  #X
#--------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

def evaluate_mlp(X, y, alpha, hidden_layer_sizes, learning_rate_init, name):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )
    # Standardize inputs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build comparable model
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,  # 3 hidden layers
        activation="relu",
        learning_rate_init=learning_rate_init,
        solver="adam",
        max_iter=2000,
        alpha=alpha,
        random_state=42,
        early_stopping=True
    )

    # Train and evaluate
    mlp.fit(X_train_scaled, y_train)
    
    y_pred_val = mlp.predict(X_test_scaled)    
    mse_list=[]
    mape_list=[]
    r2_list=[]
    for i in range(len(targets)):
        mse= mean_squared_error(y_test.iloc[:,i], y_pred_val[:,i])
        mape= mean_absolute_percentage_error(y_test.iloc[:,i], y_pred_val[:,i])
        r2= r2_score(y_test.iloc[:,i], y_pred_val[:,i])
        mse_list.append(mse)
        mape_list.append(mape)
        r2_list.append(r2)
        # print(f"{name:5s} -> Type:  {targets[i]} | Val MSE: {mse:.4f} | Val R²: {r2:.4f}")
        
    return {'Model': name, 'MAPE': mape_list, 'R²': r2_list}
#------------------------------------------------------------------------------------------------------------------------------------------------------
# Deciding the most feasible Hyperparameter in simulated envrionment
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

for Typ in ('PCs_only', 'Top_plus_PCs','Top_union','Top_only'):

    X_train, X_test, y_train, y_test = train_test_split(
        globals()[f"{Typ}_df"], globals()[f"{Typ}"][targets], test_size=0.2, random_state=42)
    # Standardize inputs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
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
        scoring='neg_mean_absolute_percentage_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train_scaled, y_train)
    print(f"Best parameters of {Typ}:", grid.best_params_)
#-------------------------------------------------
# Performance for multiple input sets with hyperparameter tunning in simulated scenario
results = []
results.append(evaluate_mlp(PCs_only_df, PCs_only[targets], 0.001, (64, 32), 0.0005, "PCs only"))
results.append(evaluate_mlp(Top_plus_PCs_df, Top_plus_PCs[targets], 0.0001, (64, 32), 0.0005, "PCs + top originals"))
results.append(evaluate_mlp(Top_union_df, Top_union[targets], 1e-05, (128, 64, 32), 0.0001, "Unions of Top-originals"))
results.append(evaluate_mlp(Top_only_df, Top_only[targets], 1e-05, (128, 64, 32), 0.001, "Top originals only"))
results=pd.DataFrame(results)
MAPE=pd.DataFrame(results['MAPE'].tolist(), columns=["Methanol_selectivity_MAPE", "CO2_conversion_rate_MAPE", "CO_selectivity_MAPE"])
R2=pd.DataFrame(results['R²'].tolist(), columns=["Methanol_selectivity_R²", "CO2_conversion_rate_R²", "CO_selectivity_R²"])
display(pd.concat([results, MAPE], axis=1).drop(columns=['MAPE','R²']))
display(pd.concat([results, R2], axis=1).drop(columns=['MAPE','R²']))
#------------------------------------------------
#Visualized MAPE comparison result in barplot
import seaborn as sns
kk=pd.concat([results, MAPE], axis=1).drop(columns=['MAPE','R²'])
df_melt = kk.melt(id_vars='Model', 
                  value_vars=['Methanol_selectivity_MAPE', 'CO2_conversion_rate_MAPE', 'CO_selectivity_MAPE'],
                  var_name='MAPE_Type', 
                  value_name='Value')

plt.figure(figsize=(8, 5))
sns.barplot(data=df_melt, x='Model', y='Value', hue='MAPE_Type')
plt.title('Model Comparison of MAPEs')
plt.xlabel('Model_Type')
plt.ylabel('Mean Absolute Percentage Error')
plt.legend(title='MAPE_Type')
plt.show()
#-----------------------------------------------
#Visualized R2 comparison result in barplot
qq=pd.concat([results, R2], axis=1).drop(columns=['MAPE','R²'])
df_melt = qq.melt(id_vars='Model', 
                  value_vars=['Methanol_selectivity_R²', 'CO2_conversion_rate_R²', 'CO_selectivity_R²'],
                  var_name='R²_Type', 
                  value_name='Value')

plt.figure(figsize=(8, 5))
sns.barplot(data=df_melt, x='Model', y='Value', hue='R²_Type')

plt.title('Model Comparison of R²')
plt.xlabel('Model_Type')
plt.ylabel('R²')
plt.ylim(0.5, 1)
plt.legend(title='R²_Type',bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
#--------------------------------------------------------------------------------------------------------------------------------------------
# Real-world environment Target parameters
targets = ["Methanol_lag", "CO2_lag", "CO_lag"]  #y
# PCs only
PCs_only=pd.concat([pd.DataFrame(Real_X_pca,columns=["PCA1", "PCA2", "PCA3"]),Real_data[['Methanol_lag','CO2_lag','CO_lag']].reset_index(drop=True)] ,axis=1) 
PCs_only_df=PCs_only.drop(columns=[col for col in PCs_only.columns if col in targets]) #X

# Top-originals+PCs
Top_PC=Real_top_per_pc.get('PC1')+['Methanol_lag','CO2_lag','CO_lag']
Top_plus_PCs=pd.concat([pd.DataFrame(Real_X_pca,columns=["PCA1", "PCA2", "PCA3"]), Real_data[Top_PC].reset_index(drop=True)],axis=1)
Top_plus_PCs_df=Top_plus_PCs.drop(columns=[col for col in Top_plus_PCs.columns if col in targets])  #X

# Unions of Top-originals
Top_PC=list(set(Real_top_per_pc.get('PC1')).union(Real_top_per_pc.get('PC2')).union(Real_top_per_pc.get('PC3')))+['Methanol_lag','CO2_lag','CO_lag']
Top_union=Real_data[Top_PC].reset_index(drop=True)
Top_union_df=Top_union.drop(columns=[col for col in Top_union.columns if col in targets])  #X

#Top_original only
Top_PC=Real_top_per_pc.get('PC1')+['Methanol_lag','CO2_lag','CO_lag']
Top_only=Real_data[Top_PC].reset_index(drop=True)
Top_only_df=Top_only.drop(columns=[col for col in Top_only.columns if col in targets])  #X
#--------------------------------------------
# Deciding the most feasible Hyperparameter in Real-world envrionment
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

for Typ in ('PCs_only', 'Top_plus_PCs','Top_union','Top_only'):

    X_train, X_test, y_train, y_test = train_test_split(
        globals()[f"{Typ}_df"], globals()[f"{Typ}"][targets], test_size=0.2, random_state=42)
    # Standardize inputs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
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
        scoring='neg_mean_absolute_percentage_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train_scaled, y_train)
    print(f"Best parameters of {Typ}:", grid.best_params_)
#--------------------------------------------------------------------------------
# Performance for multiple input sets with hyperparameter tunning in real-world scenario
results = []
results.append(evaluate_mlp(PCs_only_df, PCs_only[targets], 1e-05, (128, 64, 32), 0.0005, "PCs only"))
results.append(evaluate_mlp(Top_plus_PCs_df, Top_plus_PCs[targets], 0.0001, (128, 64, 32), 0.0001, "PCs + top originals"))
results.append(evaluate_mlp(Top_union_df, Top_union[targets], 1e-05, (64, ), 0.001, "Unions of Top-originals"))
results.append(evaluate_mlp(Top_only_df, Top_only[targets], 0.0001, (128, 64, 32), 0.001, "Top originals only"))
results=pd.DataFrame(results)
MAPE=pd.DataFrame(results['MAPE'].tolist(), columns=["Methanol_selectivity_MAPE", "CO2_conversion_rate_MAPE", "CO_selectivity_MAPE"])
R2=pd.DataFrame(results['R²'].tolist(), columns=["Methanol_selectivity_R²", "CO2_conversion_rate_R²", "CO_selectivity_R²"])
display(pd.concat([results, MAPE], axis=1).drop(columns=['MAPE','R²']))
display(pd.concat([results, R2], axis=1).drop(columns=['MAPE','R²']))
#-----------------------------------------
kk=pd.concat([results, MAPE], axis=1).drop(columns=['MAPE','R²'])
df_melt = kk.melt(id_vars='Model', 
                  value_vars=['Methanol_selectivity_MAPE', 'CO2_conversion_rate_MAPE', 'CO_selectivity_MAPE'],
                  var_name='MAPE_Type', 
                  value_name='Value')

plt.figure(figsize=(8, 5))
sns.barplot(data=df_melt, x='Model', y='Value', hue='MAPE_Type')
plt.title('Model Comparison of MAPEs')
plt.xlabel('Model_Type')
plt.ylabel('Mean Absolute Percentage Error')
plt.legend(title='MAPE_Type')
plt.show()
#-----------------------------------------
qq=pd.concat([results, R2], axis=1).drop(columns=['MAPE','R²'])
df_melt = qq.melt(id_vars='Model', 
                  value_vars=['Methanol_selectivity_R²', 'CO2_conversion_rate_R²', 'CO_selectivity_R²'],
                  var_name='R²_Type', 
                  value_name='Value')

plt.figure(figsize=(8, 5))
sns.barplot(data=df_melt, x='Model', y='Value', hue='R²_Type')

plt.title('Model Comparison of R²')
plt.xlabel('Model_Type')
plt.ylabel('R²')
plt.ylim(0.2, 1)
plt.legend(title='R²_Type',bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
