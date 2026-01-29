# Artificial Neural Network based optimization of Methanol synthesis via CO2 Hydrogenation
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
- Idealized, noise-free process data  
- Used to evaluate baseline ANN prediction performance  

### 2️⃣ Real-World Scenario
- Simulation data adjusted to reflect realistic operational uncertainties  
- Includes variability and noise representative of industrial conditions  
- Used to assess ANN robustness and practical applicability  

### Input Parameters (examples)
- Reactor temperature  
- Reactor pressure  
- H₂/CO₂ feed ratio  
- Space velocity  
- Feed composition  

### Output Targets
- Methanol selectivity  
- CO₂ conversion  
- CO selectivity  

---

## 🧠 Methodology

1. **Process Simulation**  
   - Methanol synthesis via CO₂ hydrogenation modeled in Aspen Plus  
   - Steady-state operating data exported for machine learning  

2. **Data Preprocessing**  
   - Data normalization  
   - Scenario-based dataset separation  
   - Training / validation / testing split  

3. **ANN Model Development**  
   - Feed-forward artificial neural networks  
   - Hyperparameter tuning (layers, neurons, activation functions)  
   - Performance evaluation using statistical metrics  

4. **Parameter Importance Analysis**  
   - Sensitivity analysis based on ANN predictions  
   - Identification of essential monitoring variables  
