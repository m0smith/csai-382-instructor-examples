# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 4.3: Train Your First ML Models — Hyperparameter Tuning & Choosing the Best Model
# MAGIC 
# MAGIC Welcome! This instructor-facing notebook is designed for live coding demos in **CSAI-382 (AI Model Training) at Ensign College**.
# MAGIC 
# MAGIC **What we will do in this lab:**
# MAGIC - Load a **feature pipeline** and **transformed train/test data**.
# MAGIC - Train **two models**: Logistic Regression and Random Forest.
# MAGIC - Perform **hyperparameter tuning** with `GridSearchCV`.
# MAGIC - Compare models, pick the best one, and save it for later use.
# MAGIC 
# MAGIC **Story reminder:** We are building a STEDI model to predict fall risk from sensor data. This notebook corresponds to **Lab 4.3**.

# COMMAND ----------
# MAGIC %md
# MAGIC # Section 1: Load Libraries, Pipeline, and Data
# MAGIC 
# MAGIC In this section we:
# MAGIC - Import the key libraries: **numpy, pandas, scikit-learn, joblib**.
# MAGIC - Load the **feature pipeline**. This pipeline turns raw sensor inputs into numeric features that models understand.
# MAGIC - Load the **transformed training and test sets** so we can focus on modeling, not preprocessing.
# MAGIC 
# MAGIC **Instructor Tip:** Emphasize that we are *not* rebuilding the pipeline here. We simply reuse it, just like we will reuse models later.

# COMMAND ----------
# Import libraries
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1.1: Load the pipeline and datasets from Databricks FileStore
# MAGIC 
# MAGIC We saved these files in earlier labs. Now we load them directly from **/dbfs/FileStore**.

# COMMAND ----------
# Databricks-style paths (mounted to local /dbfs)
pipeline_path = "/dbfs/FileStore/stedi_feature_pipeline.pkl"
X_train_path = "/dbfs/FileStore/X_train_transformed.npy"
X_test_path = "/dbfs/FileStore/X_test_transformed.npy"
y_train_path = "/dbfs/FileStore/y_train.pkl"
y_test_path = "/dbfs/FileStore/y_test.pkl"

# Load the pipeline (not used directly for training here, but good to have)
feature_pipeline = joblib.load(pipeline_path)

# Load transformed feature arrays
X_train = np.load(X_train_path)
X_test = np.load(X_test_path)

# Load labels
y_train = joblib.load(y_train_path)
y_test = joblib.load(y_test_path)

# Quick sanity checks
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train value counts:\n", pd.Series(y_train).value_counts())

# MAGIC %md
# MAGIC **Instructor Tip:** Use `X_train.shape` as a quick check that the data loaded correctly and that the number of rows matches expectations.

# COMMAND ----------
# MAGIC %md
# MAGIC # Section 2: Model 1 — Logistic Regression (Baseline)
# MAGIC 
# MAGIC **What is Logistic Regression?** It is a simple model for binary classification. Here it predicts whether a sensor window indicates a **step** or **no_step**.
# MAGIC 
# MAGIC - It works well when the decision boundary is roughly a straight line in feature space.
# MAGIC - **Analogy:** Imagine drawing a straight line to separate two groups of dots on a paper.

# COMMAND ----------
# Instantiate the model with a higher max_iter to help convergence
log_reg_baseline = LogisticRegression(max_iter=300)

# Train on the training data
log_reg_baseline.fit(X_train, y_train)

# Evaluate on the test data
log_reg_baseline_accuracy = log_reg_baseline.score(X_test, y_test)
print(f"Baseline Logistic Regression accuracy: {log_reg_baseline_accuracy:.3f}")

# COMMAND ----------
# MAGIC %md
# MAGIC **Interpreting accuracy:** For this demo, an accuracy between **0.75 and 0.90** is typical, but it depends on the data. Higher is better, but we also care about other metrics.
# MAGIC 
# MAGIC **Instructor Tip:** Ask students why accuracy alone might not be enough for medical or safety-related models.

# COMMAND ----------
# MAGIC %md
# MAGIC # Section 3: Hyperparameter Tuning for Logistic Regression
# MAGIC 
# MAGIC **What is a hyperparameter?** A setting we choose **before** training (like oven temperature and baking time). We want to find the best settings.
# MAGIC 
# MAGIC **Analogy:** Trying different oven temperatures and baking times to get the perfect cake.
# MAGIC 
# MAGIC **What does `GridSearchCV` do?**
# MAGIC - Tries all combinations of the provided hyperparameters.
# MAGIC - Uses cross-validation (splits the training data into small folds) to estimate performance for each combination.

# COMMAND ----------
# Define the hyperparameter grid for Logistic Regression
log_reg_params = {
    "C": [0.01, 0.1, 1, 10],  # strength of regularization
    "penalty": ["l2"],
    "solver": ["lbfgs", "liblinear"],
}

log_reg_grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=300),
    param_grid=log_reg_params,
    cv=3,
    scoring="accuracy",
)

# Run grid search
log_reg_grid.fit(X_train, y_train)

print("Best parameters for Logistic Regression:", log_reg_grid.best_params_)
print(f"Best cross-validated accuracy: {log_reg_grid.best_score_:.3f}")

# COMMAND ----------
# MAGIC %md
# MAGIC **How to read the results:**
# MAGIC - `best_params_` tells us the winning settings (like the best oven temperature and time).
# MAGIC - `best_score_` is the cross-validated accuracy using those settings.
# MAGIC 
# MAGIC **Instructor Tip:** Ask which parameter students think had the biggest impact and why.

# COMMAND ----------
# MAGIC %md
# MAGIC # Section 4: Model 2 — Random Forest (Baseline)
# MAGIC 
# MAGIC **What is a Random Forest?** It is a group of many decision trees. Each tree votes on the answer, and the forest takes the majority vote.
# MAGIC 
# MAGIC - Great for **non-linear** patterns and noisy data (like sensor signals).
# MAGIC - **Analogy:** A committee of doctors gives a diagnosis instead of relying on just one doctor.

# COMMAND ----------
# Instantiate the model
rf_baseline = RandomForestClassifier(random_state=42)

# Train the model
rf_baseline.fit(X_train, y_train)

# Evaluate accuracy
rf_baseline_accuracy = rf_baseline.score(X_test, y_test)
print(f"Baseline Random Forest accuracy: {rf_baseline_accuracy:.3f}")

# COMMAND ----------
# MAGIC %md
# MAGIC Compare the baselines:
# MAGIC - Logistic Regression baseline accuracy: `log_reg_baseline_accuracy`
# MAGIC - Random Forest baseline accuracy: `rf_baseline_accuracy`
# MAGIC 
# MAGIC **Instructor Tip:** Ask students why Random Forest might work better on sensor data than Logistic Regression.

# COMMAND ----------
# MAGIC %md
# MAGIC # Section 5: Hyperparameter Tuning for Random Forest
# MAGIC 
# MAGIC Key hyperparameters in a Random Forest:
# MAGIC - **n_estimators:** number of trees in the forest.
# MAGIC - **max_depth:** how deep each tree can grow.
# MAGIC - **min_samples_split:** minimum samples needed to split a node.
# MAGIC - **min_samples_leaf:** minimum samples required to be at a leaf node.
# MAGIC 
# MAGIC **ESL-friendly descriptions:**
# MAGIC - More trees can capture more patterns but take longer to train.
# MAGIC - Deeper trees can fit complex patterns but can also overfit.
# MAGIC - `min_samples_split` and `min_samples_leaf` prevent trees from becoming too specific to the training data.

# COMMAND ----------
# Define the hyperparameter grid for Random Forest
rf_params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_params,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,  # use all cores for speed
)

# Run grid search
rf_grid.fit(X_train, y_train)

print("Best parameters for Random Forest:", rf_grid.best_params_)
print(f"Best cross-validated accuracy: {rf_grid.best_score_:.3f}")

# COMMAND ----------
# MAGIC %md
# MAGIC **Trade-offs:**
# MAGIC - More trees → usually better accuracy but slower training and prediction.
# MAGIC - Deeper trees → may fit the training data very well but risk overfitting.
# MAGIC 
# MAGIC **Instructor Tip:** Pause to note how long the grid search takes. Ask students why more parameter options mean longer training time.

# COMMAND ----------
# MAGIC %md
# MAGIC # Section 6: Compare Tuned Models
# MAGIC 
# MAGIC To compare models fairly:
# MAGIC - Use the **same metric** (accuracy here).
# MAGIC - Use scores from **cross-validation** so we are not biased by a single split.
# MAGIC - Check that we used the same training data and folds.

# COMMAND ----------
# Create a simple comparison
comparison = {
    "Logistic Regression (tuned)": log_reg_grid.best_score_,
    "Random Forest (tuned)": rf_grid.best_score_,
}

comparison_df = pd.DataFrame.from_dict(comparison, orient="index", columns=["CV Accuracy"])
print(comparison_df)

# Optional: quick bar chart for visualization
try:
    import matplotlib.pyplot as plt

    comparison_df.plot(kind="bar", legend=False, ylim=(0, 1), title="Model Comparison (CV Accuracy)")
    plt.ylabel("Accuracy")
    plt.show()
except Exception as e:
    print("Plotting skipped due to:", e)

# COMMAND ----------
# MAGIC %md
# MAGIC **Interpretation:**
# MAGIC - Even small differences (e.g., 0.89 vs. 0.91) can matter if the application is safety-critical.
# MAGIC - But sometimes a simpler model is preferred if it is easier to maintain and explain.
# MAGIC 
# MAGIC **Instructor Tip:** Ask students if a slightly higher accuracy always justifies using the more complex model.

# COMMAND ----------
# MAGIC %md
# MAGIC # Section 7: Choose and Save the Best Model
# MAGIC 
# MAGIC We save the best model so we can reuse it later without retraining. `joblib.dump` writes the model to disk, and `joblib.load` brings it back.
# MAGIC 
# MAGIC We will choose the model with the higher cross-validated accuracy.

# COMMAND ----------
# Choose the better model based on cross-validated accuracy
if rf_grid.best_score_ >= log_reg_grid.best_score_:
    best_model = rf_grid.best_estimator_
    best_model_name = "Random Forest"
else:
    best_model = log_reg_grid.best_estimator_
    best_model_name = "Logistic Regression"

print(f"Best model selected: {best_model_name}")

# Save the best model to FileStore
best_model_path = "/dbfs/FileStore/stedi_best_model.pkl"
joblib.dump(best_model, best_model_path)
print(f"Best model saved to {best_model_path}")

# List the file to confirm save
import os
print("Files in /dbfs/FileStore:")
print(os.listdir("/dbfs/FileStore"))

# COMMAND ----------
# MAGIC %md
# MAGIC **Instructor Tip:** Explain that saving the model is a step toward production. Later, we can load this file to make predictions in a dashboard or API.

# COMMAND ----------
# MAGIC %md
# MAGIC # Section 8: Mini Evaluation & Ethics Discussion (Instructor Notes)
# MAGIC 
# MAGIC Accuracy is useful, but it is not the whole story. For fall risk prediction, we also care about **false negatives** (missed risks) and **false positives** (unnecessary alarms).
# MAGIC 
# MAGIC **Ethics and fairness:**
# MAGIC - A model that looks “good” overall could still be unfair to certain groups if the training data is biased.
# MAGIC - We should consider balanced datasets, additional metrics (recall, precision), and ongoing monitoring.
# MAGIC 
# MAGIC **Spiritual connection:** As we judge model performance, remember principles of **fairness, truth, and righteous judgment**.
# MAGIC 
# MAGIC **Discussion questions:**
# MAGIC 1. How could a “good” model still hurt people if used carelessly?
# MAGIC 2. What additional checks would you want before deploying a balance-risk model for seniors?
# MAGIC 3. When might a simpler model be better than a more complex one?
# MAGIC 4. How can we make sure our model respects all users equally?
