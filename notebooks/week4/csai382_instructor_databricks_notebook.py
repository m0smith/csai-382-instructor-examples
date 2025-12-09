# Databricks notebook source
# MAGIC %md
# MAGIC # CSAI-382 Week 4–6: STEDI Model Training Instructor Notebook
# MAGIC 
# MAGIC **Why this matters:** Students see the full path from raw sensor data to a trained model with fairness checks.
# MAGIC 
# MAGIC **Overview**
# MAGIC * Live, ESL-friendly teaching notebook for CSAI-382 (Week 4–6 pipeline)
# MAGIC * Covers ETL → feature building → model training → SHAP → bias reflection
# MAGIC * Designed for Databricks Free Tier with Pandas, PySpark, and scikit-learn
# MAGIC * Emphasizes clear steps, common mistakes, and responsible AI reminders
# MAGIC 
# MAGIC **Learning outcomes connection**
# MAGIC * Prepare and clean time-series sensor data
# MAGIC * Build reproducible ML pipelines
# MAGIC * Interpret models with SHAP
# MAGIC * Discuss fairness, bias, and stewardship in AI
# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load Data (Pandas and PySpark)
# MAGIC **Why this matters:** Clean inputs prevent bad models. Students must see both Pandas and Spark paths.
# MAGIC 
# MAGIC * We will load STEDI device messages and rapid step tests.
# MAGIC * Files live in FileStore for class demos.
# MAGIC * Remember: timestamps and units must be clear before modeling.
# MAGIC 
# MAGIC **Common mistakes**
# MAGIC * Forgetting `inferSchema=True` in Spark reads
# MAGIC * Mixing up milliseconds vs seconds timestamps
# MAGIC * Not checking column names before joins
# COMMAND ----------
# Pandas option: load CSVs
import pandas as pd

# These are placeholder paths for classroom demo
pandas_device_path = "/FileStore/tables/devicemessage.csv"
pandas_step_path = "/FileStore/tables/rapidsteptest.csv"

# Load with basic options and inspect
pandas_device_df = pd.read_csv(pandas_device_path)
pandas_step_df = pd.read_csv(pandas_step_path)

# Look at the schema and first rows
print("DeviceMessage (pandas) info:\n", pandas_device_df.info())
print(pandas_device_df.head())
print("RapidStepTest (pandas) info:\n", pandas_step_df.info())
print(pandas_step_df.head())
# COMMAND ----------
# PySpark option: load CSVs
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.getOrCreate()

spark_device_path = "/FileStore/tables/devicemessage.csv"
spark_step_path = "/FileStore/tables/rapidsteptest.csv"

spark_device_df = (
    spark.read.option("header", True).option("inferSchema", True).csv(spark_device_path)
)
spark_step_df = (
    spark.read.option("header", True).option("inferSchema", True).csv(spark_step_path)
)

# Inspect schemas and a few rows
spark_device_df.printSchema()
spark_device_df.show(5)
spark_step_df.printSchema()
spark_step_df.show(5)

# Quick visual of distances (Databricks display)
display(spark_device_df.select("distance"))

# Try This! Filter by deviceId (students can change the ID)
try_device_id = "student_device_01"
display(spark_device_df.filter(col("deviceId") == try_device_id))
# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Clean and Prepare Data
# MAGIC **Why this matters:** Good features reduce noise and bias. Cleaning steps make the model fairer.
# MAGIC 
# MAGIC Steps:
# MAGIC * Cast columns to correct types
# MAGIC * Convert timestamps to datetime
# MAGIC * Handle nulls
# MAGIC * Join DeviceMessage + RapidStepTest on a time window
# MAGIC * Aggregate distances for stability (avg, min, max, variance)
# MAGIC 
# MAGIC ⚠️ **Fairness note:** Noisy sensors may affect one device group more than another. Cleaning reduces hidden bias.
# MAGIC 
# MAGIC **Common mistakes**
# MAGIC * Forgetting to drop rows with missing labels
# MAGIC * Joining on exact timestamps instead of windows
# MAGIC * Aggregating before handling nulls
# COMMAND ----------
from pyspark.sql.functions import avg, min as spark_min, max as spark_max, variance, to_timestamp, expr

# Cast and timestamp handling
clean_device_df = (
    spark_device_df
    .withColumn("distance", col("distance").cast("double"))
    .withColumn("timestamp", to_timestamp(col("timestamp")))
)

clean_step_df = spark_step_df.withColumn("timestamp", to_timestamp(col("timestamp")))

# Handle nulls: drop rows missing critical fields
clean_device_df = clean_device_df.dropna(subset=["distance", "timestamp", "deviceId"])
clean_step_df = clean_step_df.dropna(subset=["timestamp", "deviceId", "stepPoints"])

# Join on deviceId and close timestamps (within 30 seconds)
time_joined_df = clean_device_df.alias("d").join(
    clean_step_df.alias("s"),
    (col("d.deviceId") == col("s.deviceId"))
    & (col("d.timestamp") >= col("s.timestamp") - expr("INTERVAL 30 seconds"))
    & (col("d.timestamp") <= col("s.timestamp") + expr("INTERVAL 30 seconds")),
    how="inner",
)

# Aggregate distance features per rapid step test
features_df = time_joined_df.groupBy("s.deviceId", "s.timestamp").agg(
    avg(col("d.distance")).alias("avgDistance"),
    spark_min(col("d.distance")).alias("minDistance"),
    spark_max(col("d.distance")).alias("maxDistance"),
    variance(col("d.distance")).alias("varianceDistance"),
    avg(col("s.stepPoints")).alias("avgStepPoints"),
)

# Inspect results
features_df.show(10)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Feature Engineering Pipeline (scikit-learn)
# MAGIC **Why this matters:** Pipelines cut human error. They keep preprocessing and models together.
# MAGIC 
# MAGIC * We use StandardScaler + SimpleImputer + LogisticRegression.
# MAGIC * You can swap in RandomForestClassifier if preferred.
# MAGIC * Avoid leaking test data by fitting on train only.
# MAGIC 
# MAGIC **Try This!** Add another feature such as `avgStepPoints` or a rolling median.
# COMMAND ----------
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Convert Spark DataFrame to pandas for scikit-learn
pandas_features = features_df.toPandas()

# Define features and label (placeholder label column for demo)
feature_cols = ["avgDistance", "minDistance", "maxDistance", "varianceDistance", "avgStepPoints"]
label_col = "step_label"  # Replace with actual label when available

# For demo, create a mock label if not present
if label_col not in pandas_features.columns:
    pandas_features[label_col] = (pandas_features["avgDistance"] > pandas_features["avgDistance"].median()).astype(int)

X = pandas_features[feature_cols]
y = pandas_features[label_col]

# Numeric preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

preprocess = ColumnTransformer(
    transformers=[("num", numeric_transformer, feature_cols)]
)

model = LogisticRegression(max_iter=100, solver="lbfgs")

clf = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

print("Pipeline ready. Components:\n", clf)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Train/Test Split and Model Training
# MAGIC **Why this matters:** Models must be tested on unseen data to avoid over-confidence.
# MAGIC 
# MAGIC * Select X (features) and y (label)
# MAGIC * Split into train and test
# MAGIC * Fit the pipeline
# MAGIC * View accuracy, precision, recall
# MAGIC 
# MAGIC **Common mistakes**
# MAGIC * Forgetting `stratify=y` for imbalanced labels
# MAGIC * Evaluating on train data only
# MAGIC * Ignoring confusion matrix insights
# COMMAND ----------
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf.fit(X_train, y_train)

# Predictions and metrics
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print("Confusion Matrix:\n", cm)

# Interpretation guidance
print("Accuracy alone can hide which class the model fails on. Confusion matrix shows false positives/negatives.")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. SHAP Explainability
# MAGIC **Why this matters:** SHAP shows what the model pays attention to. This helps spot bias.
# MAGIC 
# MAGIC * Build a SHAP explainer
# MAGIC * Show summary and bar plots
# MAGIC * Watch for one feature dominating the model
# MAGIC 
# MAGIC ⚠️ If SHAP is slow, sample rows first to keep the demo quick.
# COMMAND ----------
import shap
import matplotlib.pyplot as plt

# Fit a small explainer (use a sample to stay fast on Free Tier)
shap_sample = X_test.sample(min(200, len(X_test)), random_state=42)
explainer = shap.Explainer(clf.named_steps["model"], clf.named_steps["preprocess"].transform(X_train))

# Calculate SHAP values
shap_values = explainer(clf.named_steps["preprocess"].transform(shap_sample))

# Summary plot
shap.summary_plot(shap_values, shap_sample, show=False)
plt.show()

# Bar plot
shap.plots.bar(shap_values, max_display=10)
plt.show()
# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Bias Evaluation Mini-Framework
# MAGIC **Why this matters:** Fair models respect all users. We must question our data and steps.
# MAGIC 
# MAGIC * Where could **data bias** happen? (Missing devices, skewed timestamps?)
# MAGIC * Where could **feature bias** happen? (Distances miscalibrated for some devices?)
# MAGIC * Where could **model bias** happen? (Class imbalance?)
# MAGIC * Where could **process bias** happen? (Our choices on thresholds?)
# MAGIC 
# MAGIC Connect to CSAI-382 ethics outcomes: stewardship, integrity, and fairness when building AI.
# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. End-of-Class Review + Quick Challenges
# MAGIC **Why this matters:** Practice cements learning and keeps fairness in focus.
# MAGIC 
# MAGIC **Coding challenges**
# MAGIC 1. Add a new feature (e.g., rolling mean of distance) and retrain the pipeline.
# MAGIC 2. Swap LogisticRegression with RandomForestClassifier and compare metrics.
# MAGIC 3. Create a confusion matrix heatmap with matplotlib.
# MAGIC 
# MAGIC **Discussion questions (ethics & fairness)**
# MAGIC 1. How might missing data from certain devices create unfair predictions?
# MAGIC 2. What happens if one step test happens at a different speed? Does the model assume the same pace?
# MAGIC 3. How can we monitor this model in production to catch drift that affects fairness?
# MAGIC 
# MAGIC **Gospel integration thought**
# MAGIC * Fairness and stewardship: Caring for every learner and user reflects integrity and respect. Models should not favor one group unfairly.
