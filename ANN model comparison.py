import joblib
mlp_simulated = joblib.load("Methanol_simulated_ANN.pkl")
mlp_real= joblib.load("Methanol_real_ANN.pkl")
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Compare Simulated model and Real-world model performance with different feed feature numbers
# Aspen+ data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
feature_counts = [2,4,6,8,10,12,13]

R2_Aspen = {key: [] for key in feature_counts}
RMSE_Aspen = {key: [] for key in feature_counts}
MAE_Aspen = {key: [] for key in feature_counts}
MAPE_Aspen = {key: [] for key in feature_counts}

for n in feature_counts:
    X_train, X_test, y_train, y_test = train_test_split(Aspen_Top_plus_PCs_df.iloc[:,1:n+1], Aspen_correlation[targets], test_size=0.2, random_state=42)
    # Standardize inputs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_scaled = scaler.fit_transform(y_train)
    y_test_scaled = scaler.transform(y_test)
    
    mlp_simulated.fit(X_train_scaled, y_train_scaled)
    y_pred = mlp_simulated.predict(X_test_scaled)
    y_pred_mlp_simulated=scaler.inverse_transform(y_pred)
    
    n_targets = y_test_scaled.shape[1]
    
    for i in range(n_targets):
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred_mlp_simulated[:, i])
        rmse = root_mean_squared_error(y_test.iloc[:, i], y_pred_mlp_simulated[:, i])
        r2  = r2_score(y_test.iloc[:, i], y_pred_mlp_simulated[:, i])
        mape = (mae / np.mean(np.abs(y_test.iloc[:, i]))) * 100 
        R2_Aspen[n].append(r2)
        RMSE_Aspen[n].append(rmse)
        MAE_Aspen[n].append(mae)
        MAPE_Aspen[n].append(mape)
#----------------------------------------------------------------------------------------
Performance_Aspen= None
for performance in ['R2','RMSE', 'MAE','MAPE']:
    kk=pd.DataFrame.from_dict(globals()[f"{performance}_Aspen"], orient='index')
    kk.columns = ["Methanol_selectivity", "CO2_conversion_rate", "CO_selectivity"]
    kk=kk.reset_index().melt(
        id_vars="index",
        var_name="Target_type",
        value_name=performance
    ).rename(columns={"index": "Feature_number"})
    if Performance_Aspen is None:
        Performance_Aspen=kk
    else:
        Performance_Aspen = Performance_Aspen.merge(kk, how="inner")
Performance_Aspen.head(7)
#----------------------------------------------------------------------------------------
# Real life data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
feature_counts = [2,4,6,8,10,12,13]

R2_Real = {key: [] for key in feature_counts}
RMSE_Real = {key: [] for key in feature_counts}
MAE_Real = {key: [] for key in feature_counts}
MAPE_Real = {key: [] for key in feature_counts}

for n in feature_counts:
    X_train, X_test, y_train, y_test = train_test_split(Real_Top_plus_PCs_df.iloc[:,1:n+1], Real_correlation[targets], test_size=0.2, random_state=42)
    # Standardize inputs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_scaled = scaler.fit_transform(y_train)
    y_test_scaled = scaler.transform(y_test)
    
    mlp_real.fit(X_train_scaled, y_train_scaled)
    y_pred = mlp_real.predict(X_test_scaled)
    y_pred_mlp_real=scaler.inverse_transform(y_pred)
    
    n_targets = y_test_scaled.shape[1]
    
    for i in range(n_targets):
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred_mlp_real[:, i])
        rmse = root_mean_squared_error(y_test.iloc[:, i], y_pred_mlp_real[:, i])
        r2  = r2_score(y_test.iloc[:, i], y_pred_mlp_real[:, i])
        mape = (mae / np.mean(np.abs(y_test.iloc[:, i]))) * 100 
        R2_Real[n].append(r2)
        RMSE_Real[n].append(rmse)
        MAE_Real[n].append(mae)
        MAPE_Real[n].append(mape)
#-------------------------------------------------------------------------------------------
Performance_Real= None
for performance in ['R2','RMSE', 'MAE','MAPE']:
    kk=pd.DataFrame.from_dict(globals()[f"{performance}_Real"], orient='index')
    kk.columns = ["Methanol_selectivity", "CO2_conversion_rate", "CO_selectivity"]
    kk=kk.reset_index().melt(
        id_vars="index",
        var_name="Target_type",
        value_name=performance
    ).rename(columns={"index": "Feature_number"})
    if Performance_Real is None:
        Performance_Real=kk
    else:
        Performance_Real = Performance_Real.merge(kk, how="inner")
Performance_Real.head(7)
#-------------------------------------------------------------------------------------------
Performance_Aspen['Source_type']='Simulated'
Performance_Real['Source_type']='Real-world'
Performance_Total=pd.concat([Performance_Aspen,Performance_Real], axis=0)
Performance_Total=Performance_Total.round({
    "R2": 4,
    "RMSE": 3,
    "MAE": 3,
    "MAPE": 3
})
Performance_Total_melt=Performance_Total.melt(
    id_vars=['Feature_number','Source_type','Target_type'],
    value_vars=['R2', 'RMSE', 'MAE','MAPE'],
    var_name='Performance_type',
    value_name='value'
)
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
g=sns.catplot(
    data=Performance_Total_melt,
    x="Feature_number",
    y='value',
    hue="Source_type",
    col='Performance_type',
    row="Target_type",
    kind="bar",
    dodge=True,
    sharey=False,
    height=3,
    aspect=1
)
g.set_titles("{col_name}")
for i, perf in enumerate(g.col_names):
    sub = Performance_Total_melt[Performance_Total_melt["Performance_type"] == perf]
    ymin, ymax = sub["value"].min(), sub["value"].max()
    margin = (ymax - ymin) * 0.12 if ymax > ymin else 0.1
    y0, y1 = max(0, ymin - margin), ymax + margin
    for j in range(len(g.row_names)):
        g.axes[j, i].set_ylim((y0, y1))

plt.show()
#------------------------------------------------------------------------------------------------------------
# Compare Simulated_ANN model and Real-world ANN model performance derivation under different time frequency
# Simulated environment with simulated trained ANN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
# Time frequency 10s, 30s, 60s, 3min, 5min, 10min
Time_frequency=[1, 3, 6, 18, 30, 60]

R2_Aspen = {key: [] for key in Time_frequency}
RMSE_Aspen = {key: [] for key in Time_frequency}
MAE_Aspen = {key: [] for key in Time_frequency}
MAPE_Aspen = {key: [] for key in Time_frequency}

for n in Time_frequency:
    X_train, X_test, y_train, y_test = train_test_split(Aspen_Top_plus_PCs_df.iloc[::n], Aspen_correlation[targets].iloc[::n], test_size=0.2, random_state=42)
    # Standardize inputs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_scaled = scaler.fit_transform(y_train)
    y_test_scaled = scaler.transform(y_test)
    
    mlp_simulated.fit(X_train_scaled, y_train_scaled)
    y_pred = mlp_simulated.predict(X_test_scaled)
    y_pred_mlp_simulated=scaler.inverse_transform(y_pred)
    
    n_targets = y_test_scaled.shape[1]
    
    for i in range(n_targets):
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred_mlp_simulated[:, i])
        rmse = root_mean_squared_error(y_test.iloc[:, i], y_pred_mlp_simulated[:, i])
        r2  = r2_score(y_test.iloc[:, i], y_pred_mlp_simulated[:, i])
        mape = (mae / np.mean(np.abs(y_test.iloc[:, i]))) * 100 
        R2_Aspen[n].append(r2)
        RMSE_Aspen[n].append(rmse)
        MAE_Aspen[n].append(mae)
        MAPE_Aspen[n].append(mape)
#-------------------------------------------------------------------------------------------
Performance_Aspen= None
for performance in ['R2','RMSE', 'MAE','MAPE']:
    kk=pd.DataFrame.from_dict(globals()[f"{performance}_Aspen"], orient='index')
    kk.columns = ["Methanol_selectivity", "CO2_conversion_rate", "CO_selectivity"]
    kk=kk.reset_index().melt(
        id_vars="index",
        var_name="Target_type",
        value_name=performance
    ).rename(columns={"index": "Time_frequency"})
    if Performance_Aspen is None:
        Performance_Aspen=kk
    else:
        Performance_Aspen = Performance_Aspen.merge(kk, how="inner")
Performance_Aspen.head(6)
#-------------------------------------------------------------------------------------
# Real-world environment with Real-world_trained_ANN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
# Time frequency 10s, 30s, 60s, 3min, 5min, 10min
Time_frequency=[1, 3, 6, 18, 30, 60]

R2_Real = {key: [] for key in Time_frequency}
RMSE_Real = {key: [] for key in Time_frequency}
MAE_Real = {key: [] for key in Time_frequency}
MAPE_Real = {key: [] for key in Time_frequency}

for n in Time_frequency:
    X_train, X_test, y_train, y_test = train_test_split(Real_Top_plus_PCs_df.iloc[::n], Real_correlation[targets].iloc[::n], test_size=0.2, random_state=42)
    # Standardize inputs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_scaled = scaler.fit_transform(y_train)
    y_test_scaled = scaler.transform(y_test)
    
    mlp_real.fit(X_train_scaled, y_train_scaled)
    y_pred = mlp_real.predict(X_test_scaled)
    y_pred_mlp_real=scaler.inverse_transform(y_pred)
    
    n_targets = y_test_scaled.shape[1]
    
    for i in range(n_targets):
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred_mlp_real[:, i])
        rmse = root_mean_squared_error(y_test.iloc[:, i], y_pred_mlp_real[:, i])
        r2  = r2_score(y_test.iloc[:, i], y_pred_mlp_real[:, i])
        mape = (mae / np.mean(np.abs(y_test.iloc[:, i]))) * 100 
        R2_Real[n].append(r2)
        RMSE_Real[n].append(rmse)
        MAE_Real[n].append(mae)
        MAPE_Real[n].append(mape)
#-----------------------------------------------------------------------------------------
Performance_Real= None
for performance in ['R2','RMSE', 'MAE','MAPE']:
    kk=pd.DataFrame.from_dict(globals()[f"{performance}_Real"], orient='index')
    kk.columns = ["Methanol_selectivity", "CO2_conversion_rate", "CO_selectivity"]
    kk=kk.reset_index().melt(
        id_vars="index",
        var_name="Target_type",
        value_name=performance
    ).rename(columns={"index": "Time_frequency"})
    if Performance_Real is None:
        Performance_Real=kk
    else:
        Performance_Real = Performance_Real.merge(kk, how="inner")
Performance_Real.head(6)
#---------------------------------------------------------------------------------------
Performance_Aspen['Data_type']='Simulated'
Performance_Real['Data_type']='Real-world'
Performance_Total_2=pd.concat([Performance_Aspen,Performance_Real], axis=0)
Performance_Total_2=Performance_Total_2.round({
    "R2": 4,
    "RMSE": 3,
    "MAE": 3,
    "MAPE": 3
})
mapping = {
    1: "10 s",
    3: "30 s",
    6: "1 min",
    18: "3 min",
    30: "5 min",
    60: "10 min"
}

Performance_Total_2["Time_frequency"] = Performance_Total_2["Time_frequency"].apply(lambda x: mapping[x])
Performance_Total_2_melt=Performance_Total_2.melt(
    id_vars=['Time_frequency','Data_type','Target_type'],
    value_vars=['R2', 'RMSE', 'MAE','MAPE'],
    var_name='Performance_type',
    value_name='value'
)
import seaborn as sns
import matplotlib.pyplot as plt
g=sns.catplot(
    data=Performance_Total_2_melt,
    x="Time_frequency",
    y='value',
    hue="Data_type",
    col='Performance_type',
    row="Target_type",
    kind="bar",
    dodge=True,
    sharey=False,
    height=3,
    aspect=1
)
g.set_titles("{col_name}")
for i, perf in enumerate(g.col_names):
    sub = Performance_Total_2_melt[Performance_Total_2_melt["Performance_type"] == perf]
    ymin, ymax = sub["value"].min(), sub["value"].max()
    margin = (ymax - ymin) * 0.12 if ymax > ymin else 0.1
    y0, y1 = max(0, ymin - margin), ymax + margin
    for j in range(len(g.row_names)):
        g.axes[j, i].set_ylim((y0, y1))

plt.show()
#--------------------------------------------------------------------------------------------------------------------------------------------------------
for ty in ['Methanol_selectivity', 'CO2_conversion_rate','CO_selectivity']:
    pivot_df=Performance_Total[Performance_Total['Target_type']==ty].pivot_table(
        index="Feature_number",
        columns="Source_type",
        values=["R2", "RMSE", "MAE", "MAPE"]
    )
    pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
    pivot_df = pivot_df.reset_index()
    pivot_df=pivot_df[pivot_df['Feature_number']>2]
    
    pivot_df["R2_diff_%"]   = round((pivot_df["R2_Simulated"]  - pivot_df["R2_Real-world"])  / pivot_df["R2_Real-world"]  * 100, 2)
    pivot_df["RMSE_diff_%"] = round((pivot_df["RMSE_Simulated"]- pivot_df["RMSE_Real-world"]) / pivot_df["RMSE_Real-world"] * 100, 2)
    pivot_df["MAE_diff_%"]  = round((pivot_df["MAE_Simulated"] - pivot_df["MAE_Real-world"]) / pivot_df["MAE_Real-world"]  * 100, 2)
    pivot_df["MAPE_diff_%"] = round((pivot_df["MAPE_Simulated"]- pivot_df["MAPE_Real-world"]) / pivot_df["MAPE_Real-world"] * 100, 2)
    print("Type:",ty, "R2:", round(pivot_df['R2_diff_%'].mean(), 2), "RMSE", round(pivot_df['RMSE_diff_%'].mean(), 2), "MAE:", round(pivot_df['MAE_diff_%'].mean(), 2), "MAPE:", round(pivot_df['MAPE_diff_%'].mean(), 2))
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
for ty in ['Methanol_selectivity', 'CO2_conversion_rate','CO_selectivity']:
    pivot_df=Performance_Total_2[Performance_Total_2['Target_type']==ty].pivot_table(
        index="Time_frequency",
        columns="Data_type",
        values=["R2", "RMSE", "MAE", "MAPE"]
    )
    pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
    pivot_df = pivot_df.reset_index()
    
    
    pivot_df["R2_diff_%"]   = round((pivot_df["R2_Simulated"]  - pivot_df["R2_Real-world"])  / pivot_df["R2_Real-world"]  * 100, 2)
    pivot_df["RMSE_diff_%"] = round((pivot_df["RMSE_Simulated"]- pivot_df["RMSE_Real-world"]) / pivot_df["RMSE_Real-world"] * 100, 2)
    pivot_df["MAE_diff_%"]  = round((pivot_df["MAE_Simulated"] - pivot_df["MAE_Real-world"]) / pivot_df["MAE_Real-world"]  * 100, 2)
    pivot_df["MAPE_diff_%"] = round((pivot_df["MAPE_Simulated"]- pivot_df["MAPE_Real-world"]) / pivot_df["MAPE_Real-world"] * 100, 2)
    print("Type:",ty, "R2:", round(pivot_df['R2_diff_%'].mean(), 2), "RMSE", round(pivot_df['RMSE_diff_%'].mean(), 2), "MAE:", round(pivot_df['MAE_diff_%'].mean(), 2), "MAPE:", round(pivot_df['MAPE_diff_%'].mean(), 2))




