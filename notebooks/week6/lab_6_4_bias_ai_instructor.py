# Databricks notebook source
# MAGIC %md
# MAGIC # CSAI-382 – Assignment 6.4 Bias AI (Instructor Live Demo)
# MAGIC 
# MAGIC **Why this matters:** In Week 4–6, students move from raw STEDI data to features, models, and fairness checks. This notebook guides live coding with clear steps for ESL learners.
# MAGIC 
# MAGIC **Notebook purpose:**
# MAGIC - Practice ETL → feature engineering → model → SHAP → bias reflection.
# MAGIC - Prepare instructors to explain both *how* and *why* decisions matter.
# MAGIC - Use Databricks Free Tier tools: Pandas, PySpark, scikit-learn, SHAP.
# MAGIC 
# MAGIC **Connection to CSAI-382 outcomes:**
# MAGIC - Build pipelines that are reproducible and ethical.
# MAGIC - Detect and discuss bias in data and models.
# MAGIC - Communicate findings with simple visuals.
# MAGIC 
# MAGIC > Teacher tip: Keep sentences short. Pause often for comprehension checks.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Notebook Title + Overview
# MAGIC 
# MAGIC **Why this matters:** Setting the roadmap helps students see the big picture. It links the lesson to the Week 4–6 pipeline (ETL → features → model → SHAP → bias).
# MAGIC 
# MAGIC **Today we will:**
# MAGIC - Load STEDI sensor data (DeviceMessage + RapidStepTest).
# MAGIC - Clean and prepare timestamps, distances, and StepPoints (30 time intervals).
# MAGIC - Build features that affect fairness (averages, min/max, variance).
# MAGIC - Create a scikit-learn pipeline and train/test split.
# MAGIC - Explain predictions with SHAP.
# MAGIC - Reflect on bias in data, features, models, and our choices.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Load Data (Pandas and PySpark)
# MAGIC 
# MAGIC **Why this matters:** Students see two common ways to read data. They compare Pandas for small files and PySpark for larger distributed data.
# MAGIC 
# MAGIC **Files (placeholders):**
# MAGIC - `/FileStore/tables/devicemessage.csv`
# MAGIC - `/FileStore/tables/rapidsteptest.csv`
# MAGIC 
# MAGIC Columns to expect:
# MAGIC - `deviceId`: unique device.
# MAGIC - `timestamp`: event time (epoch or ISO).
# MAGIC - `distance`: distance from sensor.
# MAGIC - `StepPoints`: list/array of 30 timed step intervals.
# MAGIC - Other metadata: `user`, `riskFlag`, etc.
# MAGIC 
# MAGIC **Common mistake:** Forgetting to set the schema or parse timestamps, which causes nulls or wrong types.

# COMMAND ----------
# Pandas load example (good for quick inspection)
import pandas as pd

# Path placeholders for Databricks FileStore
pandas_device_path = "/FileStore/tables/devicemessage.csv"
pandas_step_path = "/FileStore/tables/rapidsteptest.csv"

# Read CSVs with parsing timestamp
# In live demo, these files should already be uploaded to FileStore
pdf_device = pd.read_csv(pandas_device_path, parse_dates=["timestamp"], infer_datetime_format=True)
pdf_step = pd.read_csv(pandas_step_path, parse_dates=["timestamp"], infer_datetime_format=True)

# Peek at data
pdf_device.head()

# COMMAND ----------
# PySpark load example (for bigger files)
spark_device_df = (spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(pandas_device_path)
)

spark_step_df = (spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(pandas_step_path)
)

# Show schema to verify types
spark_device_df.printSchema()

# COMMAND ----------
# MAGIC %md
# MAGIC **Check timestamps and distances visually.**
# MAGIC - The `display()` function is handy in Databricks for quick charts.
# MAGIC - Watch for strange spikes (could be sensor noise).

# COMMAND ----------
# Visual quick view (Databricks display)
display(spark_device_df.select("timestamp", "distance"))

# COMMAND ----------
# MAGIC %md
# MAGIC **Try This!** Filter by `deviceId`
# MAGIC - Ask students: "Show only deviceId = 'ABC-123'."

# COMMAND ----------
# Example filter (replace with a real deviceId)
display(spark_device_df.filter(spark_device_df.deviceId == "ABC-123"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Clean and Prepare Data
# MAGIC 
# MAGIC **Why this matters:** Clean data reduces hidden bias. Wrong types or missing times can skew averages and fairness checks.
# MAGIC 
# MAGIC Steps we will show:
# MAGIC - Cast numeric columns.
# MAGIC - Convert timestamps.
# MAGIC - Drop or fill nulls.
# MAGIC - Join DeviceMessage + RapidStepTest on a time window.
# MAGIC - Aggregate distances: avg, min, max, variance.
# MAGIC 
# MAGIC **Fairness note:** If one device has more missing values, its average distance may look worse, hurting that user's model outcome.

# COMMAND ----------
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Ensure timestamp is proper timestamp type
spark_device_typed = spark_device_df.withColumn("timestamp", F.to_timestamp("timestamp"))
spark_step_typed = spark_step_df.withColumn("timestamp", F.to_timestamp("timestamp"))

# Handle null distances (simple example: drop)
spark_device_clean = spark_device_typed.dropna(subset=["distance", "timestamp"])

# Define a time-window join: within +/- 15 seconds of a rapid step test
join_window = (F.abs(F.unix_timestamp(spark_device_clean.timestamp) - F.unix_timestamp(spark_step_typed.timestamp)) <= 15)

joined_df = (spark_device_clean.alias("d")
    .join(spark_step_typed.alias("r"), join_window, "inner")
    .select("d.deviceId", F.col("d.timestamp").alias("deviceTs"), F.col("r.timestamp").alias("testTs"), "distance", "StepPoints")
)

# Calculate aggregate distance features per device
agg_features = (joined_df
    .groupBy("deviceId")
    .agg(
        F.avg("distance").alias("avgDistance"),
        F.min("distance").alias("minDistance"),
        F.max("distance").alias("maxDistance"),
        F.variance("distance").alias("varianceDistance")
    )
)

# Show results
agg_features.show(5, truncate=False)

# COMMAND ----------
# MAGIC %md
# MAGIC **Why these features matter for fairness:**
# MAGIC - **avgDistance**: If sensors drift for one group, averages look different.
# MAGIC - **min/maxDistance**: Outliers may come from bad hardware; models might overreact.
# MAGIC - **varianceDistance**: High variance can mean unstable devices; the model might unfairly flag these users as risky.
# MAGIC - Always ask: *Is the difference due to people or devices?*

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Feature Engineering Pipeline (scikit-learn)
# MAGIC 
# MAGIC **Why this matters:** Pipelines reduce manual mistakes and keep preprocessing consistent between train and test. This prevents bias from sloppy steps.
# MAGIC 
# MAGIC We will build a pipeline with:
# MAGIC - `SimpleImputer` for missing values.
# MAGIC - `StandardScaler` for feature scaling.
# MAGIC - `LogisticRegression` (binary example). Replace with `RandomForestClassifier` if you want non-linear patterns.
# MAGIC 
# MAGIC **Warning:** Do not fit scalers on the full dataset before splitting; that leaks test information.

# COMMAND ----------
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Convert Spark features to Pandas for scikit-learn
df_features = agg_features.toPandas()

# Example target: riskFlag (placeholder). In real data, replace with actual column.
# If missing, create a simple dummy target for demo.
if "riskFlag" not in df_features.columns:
    df_features["riskFlag"] = (df_features["avgDistance"] > df_features["avgDistance"].mean()).astype(int)

feature_cols = ["avgDistance", "minDistance", "maxDistance", "varianceDistance"]
X = df_features[feature_cols]
y = df_features["riskFlag"]

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=200))
])

# COMMAND ----------
# MAGIC %md
# MAGIC **Try This!** Add another feature
# MAGIC - Ask students to create a new column like `rangeDistance = maxDistance - minDistance` and include it in `feature_cols`.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Train/Test Split + Model Training
# MAGIC 
# MAGIC **Why this matters:** We need a fair test to check model quality. Accuracy alone can hide imbalance problems.

# COMMAND ----------
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Fit pipeline
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("Confusion Matrix:\n", cm)

# COMMAND ----------
# MAGIC %md
# MAGIC **Interpretation guidance:**
# MAGIC - **Accuracy** can look high even if the model ignores a minority class.
# MAGIC - **Precision**: When the model predicts risk, how often is it correct?
# MAGIC - **Recall**: How many risky cases did we find? Missing them could be unfair.
# MAGIC - **Confusion matrix**: Shows false positives and false negatives. Ask: who is harmed by each error?

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. SHAP Explainability
# MAGIC 
# MAGIC **Why this matters:** SHAP shows what the model pays attention to. If one feature dominates, investigate bias.

# COMMAND ----------
import shap
import matplotlib.pyplot as plt

# Create explainer on training data (small sample for speed)
explainer = shap.Explainer(pipeline.named_steps["model"], pipeline.named_steps["scaler"].transform(X_train))
shap_values = explainer(pipeline.named_steps["scaler"].transform(X_train))

# Summary plot (requires matplotlib inline in Databricks)
shap.summary_plot(shap_values, features=X_train, feature_names=feature_cols, show=False)
plt.title("SHAP Summary: What the model cares about")
plt.show()

# Feature importance bar plot
shap.plots.bar(shap_values, max_display=10, show=False)
plt.title("SHAP Feature Importance")
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC **Plain language:**
# MAGIC - The chart is like a spotlight: bright areas show what the model notices most.
# MAGIC - If the model depends too much on `varianceDistance`, maybe one device type is noisy. Check hardware before blaming users.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Bias Evaluation Mini-Framework
# MAGIC 
# MAGIC **Why this matters:** Bias can enter at many points. Reflecting out loud helps students see their power to choose fair steps.
# MAGIC 
# MAGIC **Ask these prompts:**
# MAGIC - **Data bias:** Are some devices missing more data? Are times uneven across users?
# MAGIC - **Feature bias:** Do features reflect device noise instead of human behavior?
# MAGIC - **Model bias:** Does the classifier favor one device or time window?
# MAGIC - **Process bias:** Did we choose thresholds or labels without stakeholder input?
# MAGIC 
# MAGIC *Connect to CSAI-382 ethics outcomes:* We are stewards of data. Fairness and integrity are part of discipleship and professional duty.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. End-of-Class Review + Quick Challenges
# MAGIC 
# MAGIC **Why this matters:** Small tasks and questions help students practice and reflect.
# MAGIC 
# MAGIC **Coding challenges:**
# MAGIC 1. Create `rangeDistance = maxDistance - minDistance` and re-train the model.
# MAGIC 2. Change the time-window join from 15 seconds to 30 seconds. Does accuracy change?
# MAGIC 3. Swap `LogisticRegression` with `RandomForestClassifier`. Compare precision and recall.
# MAGIC 
# MAGIC **Discussion questions (ethics & fairness):**
# MAGIC 1. Who is impacted by false positives in this system? False negatives?
# MAGIC 2. How could sensor quality differ by location or cost? How to correct it?
# MAGIC 3. What governance steps (checklists, audits) would you add before deployment?
# MAGIC 
# MAGIC **Gospel integration thought:** Fairness is part of stewardship. We honor God and our neighbors by building models that respect truth and avoid harm.
