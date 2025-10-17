import pandas as pd
import numpy as np
Fdata_profil=pd.read_excel('Fdata_profile.xlsx').dropna().drop(columns=['Unnamed: 0'])
#--------------------------------------------------------------------------------------------------------------------------------------------
targets = ['Methanol_lag', 'CO2_lag', 'CO_lag']
corr = Fdata_profil[Fdata_profil.columns.drop(list(Fdata_profil.filter(regex='%')))].drop(columns=['Time']).corr()[targets]
corr1=corr[~corr.index.str.contains('lag')].dropna()

def Correlation_05(df,parameter):
    list1=list(df.sort_values(parameter, ascending=True)[parameter][np.abs(df[parameter])>0.5].index)
    return list1

selected_features=list(set(Correlation_05(corr1,'Methanol_lag')).union(Correlation_05(corr1,'CO2_lag')).union(Correlation_05(corr1,'CO_lag')))
len(selected_features)
#--------------------------------------------------------------------------------------------------------------------------------------------
# Target parameters PCA analysis
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
PCA_df=Fdata_profil[selected_features].loc[2:,:].dropna()

X_scaled = scaler.fit_transform(PCA_df)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Explained Variance")
plt.show()
#--------------------------------------------------------------------------------------------------------------------------------------------
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
import pandas as pd
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=PCA_df.columns
)
top_per_pc = {f'PC{i+1}': loadings[f'PC{i+1}'].abs().sort_values(ascending=False).head(10).index.tolist()
              for i in range(pca.n_components_)}
pd.DataFrame(top_per_pc)
