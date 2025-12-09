# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Pipelines for STEDI – Instructor Demo
# MAGIC 
# MAGIC **Objectives**
# MAGIC * Train/test split
# MAGIC * Scaling numeric features
# MAGIC * One-hot encoding categorical features
# MAGIC * ColumnTransformer
# MAGIC * scikit-learn Pipeline
# MAGIC * Saving the pipeline for reuse
# MAGIC 
# MAGIC Databricks AI helpers can speed up coding, but they do not replace clear thinking and understanding. Use them as tools, not crutches.
# MAGIC 
# MAGIC > Instructor note: Pause after each section. Ask students to explain why we do the next step before running the cell.
# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Load Libraries and Data
# MAGIC * Start a Spark session.
# MAGIC * Read the `labeled_step_test` Spark table.
# MAGIC * Convert to pandas for scikit-learn.
# MAGIC 
# MAGIC We do this because scikit-learn works best with pandas / NumPy data.
# MAGIC 
# MAGIC > Instructor note: Invite students to compare Spark schema with pandas `info()`. This shows why we check data types before preprocessing.
# COMMAND ----------
# Import needed libraries
from pyspark.sql import SparkSession
import pandas as pd

# Create or get a Spark session
spark = SparkSession.builder.getOrCreate()

# Read the Spark table
spark_df = spark.table("labeled_step_test")

# Inspect schema and a few rows
spark_df.printSchema()
spark_df.show(5)

# Convert to pandas for scikit-learn
pandas_df = spark_df.toPandas()

# Look at the pandas DataFrame
print(pandas_df.head())
print(pandas_df.info())
# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Choose Features and Label
# MAGIC A **feature** is an input column we use for prediction.
# MAGIC A **label** is what we try to predict.
# MAGIC 
# MAGIC Features: `distance_cm`, `sensorType`, `deviceId`
# MAGIC 
# MAGIC Label: `step_label`
# COMMAND ----------
# Select feature and label columns
feature_cols_numeric = ["distance_cm"]
feature_cols_categorical = ["sensorType", "deviceId"]
label_col = "step_label"

# X holds the features, y holds the label
X = pandas_df[feature_cols_numeric + feature_cols_categorical]
y = pandas_df[label_col]

# Show the first rows of features
print(X.head())

# Count how many of each label we have
print(y.value_counts())
# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Train/Test Split
# MAGIC We split data to test our model on unseen rows.
# MAGIC * `test_size=0.2` keeps 20% for testing.
# MAGIC * `random_state=42` makes the split reproducible.
# MAGIC * `stratify=y` keeps label proportions the same in train and test.
# COMMAND ----------
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # repeatable split
    stratify=y          # keep label balance
)

# Show shapes of the split data
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Build Preprocessing with ColumnTransformer
# MAGIC * **StandardScaler** scales numeric features.
# MAGIC * **OneHotEncoder** turns categories into indicator columns.
# MAGIC 
# MAGIC ColumnTransformer lets us send different columns to different steps.
# MAGIC 
# MAGIC > Instructor note: Ask: "What happens if we scale categories or one-hot encode numbers?" Let students see why the right tool matters for each feature type.
# COMMAND ----------
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Scale numeric columns
numeric_transformer = StandardScaler()

# One-hot encode categorical columns
# handle_unknown="ignore" skips new categories seen only in test data
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# Combine transformers for each column group
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, feature_cols_numeric),
        ("cat", categorical_transformer, feature_cols_categorical)
    ]
)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Build a Full Pipeline
# MAGIC A scikit-learn **Pipeline** chains steps together.
# MAGIC One object keeps our full preprocessing recipe.
# MAGIC We can later add a model step after the preprocess step.
# COMMAND ----------
from sklearn.pipeline import Pipeline

# Create a pipeline with only preprocessing for now
pipeline = Pipeline(steps=[
    ("preprocess", preprocessor)
])

# Print to see the structure
print(pipeline)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Fit and Transform Data
# MAGIC * `.fit()` learns parameters from training data (means, variances, categories).
# MAGIC * `.transform()` applies the learned rules.
# MAGIC 
# MAGIC We fit on training data only, then transform both train and test.
# COMMAND ----------
# Fit the pipeline on training data
pipeline.fit(X_train)

# Transform both train and test sets
X_train_transformed = pipeline.transform(X_train)
X_test_transformed = pipeline.transform(X_test)

# Check types and shapes (may be sparse matrices)
print(type(X_train_transformed))
print("Train transformed shape:", X_train_transformed.shape)
print("Test transformed shape:", X_test_transformed.shape)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Simple Optional Model Demo
# MAGIC Here is a small classifier to show how we will use the pipeline later.
# MAGIC The same preprocessing is used for train and test, so we avoid hidden bugs.
# MAGIC 
# MAGIC > Instructor note: Keep the demo short. The focus is on the pipeline. Ask students what other models they might plug in next time.
# COMMAND ----------
from sklearn.linear_model import LogisticRegression

# Build a pipeline with preprocessing and a model
clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# Fit the model
clf.fit(X_train, y_train)

# Evaluate accuracy
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

print(f"Train accuracy: {train_accuracy:.3f}")
print(f"Test accuracy: {test_accuracy:.3f}")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Save the Pipeline with joblib
# MAGIC We save the preprocessing + model so we can reuse it later.
# MAGIC Databricks paths start with `/dbfs/FileStore/...` when using Python file paths.
# COMMAND ----------
import joblib

# Save the pipeline as a pickle file (serialized Python object)
joblib.dump(clf, "/dbfs/FileStore/stedi_feature_pipeline.pkl")

# Optional: list FileStore to confirm
try:
    files = dbutils.fs.ls("/FileStore")
    print(files)
except Exception as e:
    # Simple note if dbutils is not available
    print("dbutils not available in this environment:", e)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Ethics & Fairness Reflection
# MAGIC Using the **same pipeline** every time reduces hidden bias from manual steps. Consistent scaling and encoding help us treat similar cases in similar ways. Reproducible pipelines make it easier to audit models and explain choices to others.
# MAGIC 
# MAGIC **Discussion question:** How could inconsistent preprocessing create unfair results for some users?
# COMMAND ----------
# MAGIC %md
# MAGIC ## 11. Short Spiritual Thought
# MAGIC A feature pipeline follows the same path each time. Our spiritual routines can also be steady: prayer, scripture study, and keeping commandments. Doctrine & Covenants 130:20–21 teaches that blessings are linked to obeying laws. When we follow good patterns with humility, we can expect consistent guidance. Like a reliable pipeline, steady discipleship brings peace and direction.
# COMMAND ----------
# MAGIC %md
# MAGIC ## 12. Wrap-Up / Summary
# MAGIC Today we:
# MAGIC * Loaded the STEDI data
# MAGIC * Chose features and label
# MAGIC * Split into train/test
# MAGIC * Built ColumnTransformer and Pipeline
# MAGIC * Fit, transformed, and saved the pipeline
# MAGIC 
# MAGIC **Questions to ask students:**
# MAGIC * Why is it dangerous to fit your scaler on the test set?
# MAGIC * How does one-hot encoding change the shape of your feature matrix?
# MAGIC * Why is consistency in preprocessing important for fairness in ML?
