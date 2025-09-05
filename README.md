# Self-Project
# Intelligent Fraud Detection Engine

An end-to-end machine learning project designed to detect fraudulent credit card transactions.  
This repository contains the complete workflow, from raw data processing and feature engineering to model optimization, training, and interpretation.

---

## Project Overview
In the financial world, detecting fraudulent transactions is crucial for minimizing losses and maintaining customer trust.  
This project builds a robust and intelligent system capable of identifying fraudulent patterns in a large-scale transaction dataset.  

We go beyond a simple baseline model by incorporating:
- **Advanced feature engineering**
- **Automated hyperparameter tuning**
- **Model interpretability**

The result is a solution that is not only **accurate** but also **transparent and trustworthy**.

---

##  Key Features
- **Advanced Feature Engineering**  
  Dynamic, behavioral features based on customer spending habits and rolling time-windows  
  (e.g., transaction counts in the last 1-hour and 24-hour periods).

- **High-Performance Modeling**  
  LightGBM with GPU acceleration for fast and highly accurate classification,  
  significantly outperforming traditional models.

- **Automated Hyperparameter Tuning**  
  Integrated **Optuna** to systematically search for the optimal model hyperparameters,  
  maximizing the F1-score on the imbalanced dataset.

- **Model Interpretability**  
  Implemented **LIME** and **SHAP** to explain the "why" behind individual predictions,  
  ensuring transparency and trustworthiness.

---

##  Tech Stack
- **Language:** Python 3.9+
- **Core Libraries:** Pandas, NumPy, Scikit-learn
- **Modeling:** LightGBM, RandomForest
- **Optimization:** Optuna
- **Interpretability:** LIME, SHAP
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook / Google Colab

---

## Project Workflow
1. **Data Ingestion & Preparation**  
   - Loaded and merged multiple daily transaction files (`.pkl`) into a single DataFrame.  

2. **Exploratory Data Analysis (EDA)**  
   - Analyzed distributions, trends, and identified **class imbalance**.  

3. **Feature Engineering**  
   - Created time-based, rule-based, and behavioral features capturing transaction patterns.  

4. **Model Training & Optimization**  
   - Stratified train-test split.  
   - Trained **LightGBM with GPU acceleration**.  
   - Tuned hyperparameters using **Optuna** for best performance.  

5. **Model Evaluation & Interpretation**  
   - Evaluated with **precision, recall, F1-score, and confusion matrix**.  
   - Applied **LIME** for local interpretability of predictions.  

---

## How to Run
Follow these steps to replicate the project on your local machine:

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   Set up a virtual environment and install dependencies

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Add the dataset
Create a folder named transaction_data in the root directory.
Place all your .pkl transaction files inside this folder.

Run the Jupyter Notebook
jupyter notebook Fraud_Detection.ipynb
Execute the cells sequentially to reproduce results.
