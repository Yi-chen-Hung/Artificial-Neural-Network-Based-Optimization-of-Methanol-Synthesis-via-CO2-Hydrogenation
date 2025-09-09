import matplotlib.pyplot as plt
import seaborn as sns
# Define your targets
targets = ['Methanol selectivity %', 'CO2 conversion rate %', 'CO selectivity %']

# Compute correlation matrix
corr = df_new.drop(columns=['Time']).corr()

# Show correlations of all features with targets
n=0
corr_comp=pd.DataFrame()
for target in targets:
    globals()[f"corr_{n}"]=pd.DataFrame(corr[target][np.abs(corr[target])>0.5])
    corr_comp=pd.concat([corr_comp, globals()[f"corr_{n}"]], axis=1)
    n+=1
Possible_para=corr_comp[~corr_comp.index.str.contains("STREAMS")].fillna(0) #remove the mass/mole flow which is certainly related with target parameter
Possible_para

-------------------------------------------------------------------------------------------------------------------------------------
# Optionally, visualize correlations
# Desired correlation dataframe
plt.figure(figsize=(10,6))
sns.heatmap(       , annot=True, cmap="coolwarm", center=0)
plt.title("Correlation of Features with Targets")
plt.show()
