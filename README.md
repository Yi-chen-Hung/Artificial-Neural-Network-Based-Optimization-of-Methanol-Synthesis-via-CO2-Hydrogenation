# Artificial Neural Network based optimization of Methanol synthesis via CO₂ Hydrogenation
## 🧠 Project Overview

This project applies **Artificial Neural Networks (ANNs)** to model and optimize the performance of **methanol synthesis via CO₂ hydrogenation**, a key process in sustainable chemical production and carbon utilization.

Process data generated from **Aspen Plus simulations** are used to train ANN models capable of predicting critical performance indicators, including:

- Methanol selectivity  
- CO₂ conversion rate  
- CO selectivity  

The trained models are further analyzed to identify the **most essential process parameters**, with the objective of **minimizing the number of required monitoring detectors** while maintaining accurate process control and optimization.

---

## 🎯 Objectives

- Develop ANN models to accurately predict methanol synthesis performance  
- Compare ANN performance under **two data scenarios**:
  - **Simulated data**
  - **Real-world data**
- Identify key operating parameters influencing process performance  
- Reduce monitoring complexity by determining the minimum set of essential variables  

---

## 📊 Dataset Description

The dataset is generated using **Aspen Plus** process simulation and divided into two scenarios:

### 1️⃣ Simulated Scenario
- Fully simulated operating conditions  
- Included specific molcular mass flow rate and mole flow rate    

### 2️⃣ Real-World Scenario
- Simulation data remove the characteristic of specific molecule 

### Input Parameters
- Mole fraction
- Molecular mass flow rate
- Molecular mole flow rate
- Component temperature  
- Component pressure    
- Volumetric flow rate  
- Reactor residence time  

### Output Targets
- Methanol selectivity  
- CO₂ conversion  
- CO selectivity  

---

## 🧠 Methodology

1. **Data Construction**  
   - Derivative-based features and rolling-window statistics were introduced  
   - Time-lagged features were incorporated  

2. **Data Preprocessing**  
   - Data removes incomplete and inconsistent record 
   - Scenario-based dataset separation
   - Dimensionality reduction through Pearson correlation and PCA analysis
   - Training / validation / testing split  

3. **ANN Model Development**  
   - ANN model input combination optimization  
   - Hyperparameter tuning (hidden-layers, neurons, activation functions, initial learning rate)  
   - Performance evaluation using statistical metrics  

4. **Parameter Importance Analysis**  
   - Permutation importance
   - SHAP analysis  
   - Identification of essential monitoring variables  
