# All methanol synthesis data reading
import pandas as pd
import numpy as np
for Type in ['insert_up','osci','profil','rauf','runter']:
    excel_file=f"methanol_results_ptx_control_01_{Type}.xlsx"
    globals()[f"dfs_{Type}"]=pd.read_excel(excel_file, sheet_name=None)
    globals()[f"General_data_{Type}"]=globals()[f"dfs_{Type}"]['Tabelle1']
    globals()[f"Reactor_Temp_{Type}"]=globals()[f"dfs_{Type}"]['Tabelle2']
    globals()[f"CO_ratio_{Type}"]=globals()[f"dfs_{Type}"]['Tabelle3']
    globals()[f"CO2_ratio_{Type}"]=globals()[f"dfs_{Type}"]['Tabelle4']
    globals()[f"H2_ratio_{Type}"]=globals()[f"dfs_{Type}"]['Tabelle5']
    # Convert object to float type
    for j in ['General_data','Reactor_Temp','CO_ratio','CO2_ratio','H2_ratio']:
        name=globals()[f"{j}_{Type}"]
        for col in name.select_dtypes(include=['object']).columns:
            name[col] = pd.to_numeric(name[col], errors='coerce') # NaN for invalid values
General_data_insert_up.head()
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Methanol selectivity, CO2 conversion rate, CO selectivity calculation
for i in ['profil']:
    SYNGAS=globals()[f"General_data_{i}"].filter(like='Time').join(globals()[f"General_data_{i}"].filter(like='SYNGAS')).dropna()
    # syngas mole flow composition (H2,CO2,N2)
    Recycle=globals()[f"General_data_{i}"].filter(like='Time').join(globals()[f"General_data_{i}"].filter(like='S5')).dropna()
    # after seperation recycle mole flow composition (CO,CO2,H2,N2)
    RCT_IN=globals()[f"General_data_{i}"].filter(like='Time').join(globals()[f"General_data_{i}"].filter(like='RCT-IN')).dropna()
    # reactor inlet mole flow percentage (CO,CO2,H2)
    RCT_OUT=globals()[f"General_data_{i}"].filter(like='Time').join(globals()[f"General_data_{i}"].filter(like='RCT-OUT')).dropna()
    # reactor outlet mole flow (CO,CO2,H2)
    FL_IN=globals()[f"General_data_{i}"].filter(like='Time').join(globals()[f"General_data_{i}"].filter(like='FL-IN')).dropna()
    # Flash drum inlet flow (CO,CO2,H2,H2O,CH3OH,N2) 

# x1 CO2 + 3H2 <--> CH3OH +H2O (main reaction)
# x2 CO + 2H2 <--> CH3OH
# x3 CO2 + H2 <--> CO + H2O (side reaction)

A = np.array([
    [0, -1, 1],       # Equation 1: -x2 + x3 =  increase of CO mole 
    [1, 0, 1],        # Equation 2: x1 + x3 = usage of CO2 mole
    [3, 2, 1]        # Equation 3: 3x1 + 2x2 + x3 = usage of H2 mole
])

df_new=globals()[f"General_data_{i}"].dropna().join(pd.DataFrame(columns=['Methanol selectivity %','CO2 conversion rate %']))  # add target parameter to old dataframe

for n in range(1,len(RCT_IN)+1):
    Totalmoleflow=RCT_IN.loc[n,'STREAMS("RCT-IN").F']
    CO=Totalmoleflow*(RCT_OUT.loc[n,'STREAMS("RCT-OUTH").Zn("CO")']-RCT_IN.loc[n,'STREAMS("RCT-IN").Zn("CO")'])
    CO2=Totalmoleflow*(RCT_IN.loc[n,'STREAMS("RCT-IN").Zn("CO2")']-RCT_OUT.loc[n,'STREAMS("RCT-OUTH").Zn("CO2")'])
    H2=Totalmoleflow*(RCT_IN.loc[n,'STREAMS("RCT-IN").Zn("H2")']-RCT_OUT.loc[n,'STREAMS("RCT-OUTH").Zn("H2")'])
    b = np.array([
        CO,
        CO2,         
        H2,
        ])
    # Solve the linear system Ax = b using least squares
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # Unpack the solution
    x1, x2, x3 = x
    # Methanol selectivity from CO2
    Me_selectivity = x1 / (x1 + x3) * 100
    # CO2 conversion rate
    CO2_in = Totalmoleflow*(RCT_IN.loc[n,'STREAMS("RCT-IN").Zn("CO2")'])
    CO2_conv = (x1 + x3) / CO2_in * 100
    # CO selectivity
    CO_selectivity= x3 / (x1 +x3) * 100
    df_new.loc[n,'Methanol selectivity %']=Me_selectivity
    df_new.loc[n,'CO2 conversion rate %']=CO2_conv
    df_new.loc[n,'CO selectivity %']=CO_selectivity
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
