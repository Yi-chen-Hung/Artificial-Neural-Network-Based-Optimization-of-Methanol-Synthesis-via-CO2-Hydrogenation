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

