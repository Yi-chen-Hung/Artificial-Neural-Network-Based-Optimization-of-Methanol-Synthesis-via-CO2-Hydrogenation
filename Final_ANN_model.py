import pandas as pd
import numpy as np
for typ in ['insert_up','osci','profil','rauf','runter']:
    globals()[f"Fdata_{typ}"]=pd.read_excel(f'Fdata_{typ}.xlsx').dropna()
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Connect two different pattern of Hydrogen input data(profil+osci) 
Fdata_total=pd.DataFrame()
for typ in ['profil','osci']:
    Fdata_total=pd.concat([Fdata_total,globals()[f"Fdata_{typ}"]],axis=0)
Fdata_total=Fdata_total.reset_index(drop=True)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Target parameters
targets = ["Methanol_lag", "CO2_lag", "CO_lag"]  #y
# Unions of Top-originals
Top_PC=list(set(top_per_pc.get('PC1')).union(top_per_pc.get('PC2')).union(top_per_pc.get('PC3')))+['Methanol_lag','CO2_lag','CO_lag']
Top_union=Fdata_total[Top_PC].reset_index(drop=True)
Top_union_df=Top_union.drop(columns=[col for col in Top_union.columns if col in targets])  #X
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
X_train, X_test, y_train, y_test = train_test_split(Top_union_df, Top_union[targets], test_size=0.2, random_state=42)
# Standardize inputs
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)

mlp = MLPRegressor(
    hidden_layer_sizes= (128, 64, 32),
    activation="relu",
    learning_rate_init= 0.001,
    solver="adam",
    max_iter=2000,
    alpha=0.001,
    random_state=42,
    early_stopping=True
)

# Train and evaluate
mlp.fit(X_train_scaled, y_train_scaled)

y_pred_val = mlp.predict(X_test_scaled)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Saved trained ANN model
import joblib
joblib.dump(mlp, "Methanol_synthesis_ANN.pkl")
