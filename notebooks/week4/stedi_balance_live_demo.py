# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # CSAI-382: STEDI Rapid Step Test – Live Coding Notebook
# MAGIC 
# MAGIC **Goal:** simple ML demo with Databricks Community edition.
# MAGIC 
# MAGIC **Instructor tips:**
# MAGIC - Short cells for live pace. Pause often for questions.
# MAGIC - Use `%md` for text and `%python` (default) for code.
# MAGIC - Encourage students to run cells with **Shift+Enter**.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Learning objectives ✅
# MAGIC - Explain the STEDI device and the Rapid Step Test in simple words.
# MAGIC - Load sensor and test data into Databricks (CSV or Delta).
# MAGIC - Explore data with `display`, `.printSchema()`, `.describe()`.
# MAGIC - Join sensor data with test summaries to make a feature table.
# MAGIC - Engineer simple features (average, min, max, variance, step interval).
# MAGIC - Train a small ML model (Logistic Regression) and check metrics.
# MAGIC - Read basic model explanations (feature importance, SHAP if available).
# MAGIC - Think about responsible AI and connect to gospel principles.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Quick intro: STEDI Rapid Step Test
# MAGIC - STEDI = **Smart Trampoline Energy Distribution Instrument** (fictional but useful!).
# MAGIC - Rapid Step Test: user steps on/off a platform quickly for 15–30 seconds.
# MAGIC - Sensors send **distance readings** over time (`DeviceMessage`).
# MAGIC - Test summary (`RapidStepTest`) has `startTime`, `testTime`, `totalSteps`, `stepPoints` (timestamps of steps).
# MAGIC - We want to see if we can predict **fast vs slow** test speed.

# COMMAND ----------
# MAGIC %md
# MAGIC ### About notebook commands
# MAGIC - `%md` cells show Markdown text.
# MAGIC - Default language is Python, so we can write code without `%python`.
# MAGIC - Good for live teaching: alternate text and small code.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Create small synthetic data
# MAGIC If file paths are unknown in Community edition, we make tiny DataFrames in memory.

# COMMAND ----------
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Seed for repeatable results
np.random.seed(42)

# Create sample DeviceMessage-like data
base_time = datetime(2024, 1, 1, 12, 0, 0)
times = [base_time + timedelta(seconds=i) for i in range(30)]
device_ids = ["alpha"] * 15 + ["beta"] * 15

# distance in centimeters, add small noise
alpha_dist = 50 + np.random.normal(0, 2, 15)  # steady
beta_dist = 60 + np.random.normal(0, 5, 15)   # more variable

sensor_pdf = pd.DataFrame({
    "deviceId": device_ids,
    "timestamp": times,
    "distanceCm": np.concatenate([alpha_dist, beta_dist])
})

# Quick peek
sensor_pdf.head()

# COMMAND ----------
# MAGIC %md
# MAGIC **Save synthetic sensor data as a Spark table (optional).**
# MAGIC - In Community edition, this writes to the local DBFS.
# MAGIC - Helpful to demo reading CSV or Delta later.

# COMMAND ----------
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
sensor_df = spark.createDataFrame(sensor_pdf)

# Write as Delta table
sensor_df.write.mode("overwrite").format("delta").saveAsTable("demo_sensor")

# Also write as CSV in DBFS
csv_path = "/tmp/demo_sensor.csv"
sensor_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(csv_path)

print("Saved demo_sensor table and CSV.")

# COMMAND ----------
# MAGIC %md
# MAGIC **Create RapidStepTest-like data.**
# MAGIC - `stepPoints` are list of timestamps when a step happened.

# COMMAND ----------
test_pdf = pd.DataFrame([
    {
        "deviceId": "alpha",
        "startTime": base_time,
        "testTime": 15.0,
        "totalSteps": 18,
        "stepPoints": [base_time + timedelta(seconds=s) for s in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
    },
    {
        "deviceId": "beta",
        "startTime": base_time,
        "testTime": 22.0,
        "totalSteps": 16,
        "stepPoints": [base_time + timedelta(seconds=s) for s in [2,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]]
    }
])

rapid_df = spark.createDataFrame(test_pdf)
rapid_df.createOrReplaceTempView("demo_rapid")
rapid_df.show(truncate=False)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Load data (CSV or Delta)
# MAGIC Choose one of the examples below. In class, show both.

# COMMAND ----------
# Example: read from Delta table
sensor_from_delta = spark.table("demo_sensor")

# Example: read from CSV
sensor_from_csv = spark.read.option("header", True).csv(csv_path)

print("Delta rows", sensor_from_delta.count())
print("CSV rows", sensor_from_csv.count())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Basic exploration (EDA)
# MAGIC - `display(df)`: shows a nice table or chart.
# MAGIC - `.printSchema()`: shows column names and types.
# MAGIC - `.describe()`: summary stats (count, mean, std, min, max).

# COMMAND ----------
display(sensor_from_delta.limit(5))  # show first rows
sensor_from_delta.printSchema()

# Summary stats
sensor_from_delta.describe(["distanceCm"]).show()

# COMMAND ----------
# MAGIC %md
# MAGIC For pandas users, we can also collect small data to pandas and inspect.

# COMMAND ----------
pandas_sample = sensor_from_delta.limit(10).toPandas()
pandas_sample.info()
pandas_sample.describe()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Join sensor data with RapidStepTest
# MAGIC We simplify: join on `deviceId` and a time window near `startTime`.

# COMMAND ----------
from pyspark.sql.functions import col, expr

# Simple join: same deviceId, and timestamp within 30 seconds after startTime
joined = sensor_from_delta.alias("s").join(
    rapid_df.alias("r"),
    (col("s.deviceId") == col("r.deviceId")) &
    (col("s.timestamp") >= col("r.startTime")) &
    (col("s.timestamp") <= col("r.startTime") + expr("INTERVAL 30 SECONDS"))
)

print("Joined rows:", joined.count())
display(joined.select("deviceId", "timestamp", "distanceCm", "testTime", "totalSteps"))

# COMMAND ----------
# MAGIC %md
# MAGIC **Feature table idea:**
# MAGIC - Take raw sensor rows.
# MAGIC - Aggregate per test to get features.
# MAGIC - Add label (fast vs slow).

# COMMAND ----------
from pyspark.sql import functions as F

# Aggregate sensor features per deviceId
feature_df = joined.groupBy("s.deviceId").agg(
    F.avg("distanceCm").alias("avg_distance"),
    F.min("distanceCm").alias("min_distance"),
    F.max("distanceCm").alias("max_distance"),
    F.variance("distanceCm").alias("var_distance")
).join(
    rapid_df.select("deviceId", "testTime", "totalSteps", "stepPoints"),
    on="deviceId",
    how="inner"
)

# Label: fast test if testTime < 20 seconds
feature_df = feature_df.withColumn("is_fast", (col("testTime") < 20).cast("int"))

feature_df.show(truncate=False)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Feature engineering: step intervals
# MAGIC - `stepPoints` is a list of timestamps.
# MAGIC - We compute mean and variance of step intervals (seconds).

# COMMAND ----------
# UDFs to compute intervals from list of timestamps

def interval_stats(timestamps):
    if timestamps is None or len(timestamps) < 2:
        return (None, None)
    seconds = [t.timestamp() for t in timestamps]
    diffs = np.diff(sorted(seconds))
    return (float(np.mean(diffs)), float(np.var(diffs)))

interval_udf = F.udf(interval_stats, "struct<mean:double,var:double>")

feature_df = feature_df.withColumn("intervals", interval_udf("stepPoints"))
feature_df = feature_df.withColumn("mean_step_interval", col("intervals.mean"))
feature_df = feature_df.withColumn("step_interval_var", col("intervals.var"))
feature_df = feature_df.drop("intervals")

display(feature_df)

# COMMAND ----------
# MAGIC %md
# MAGIC 🧪 **Try it yourself:** change the feature set
# MAGIC - Add median distance
# MAGIC - Add step interval standard deviation
# MAGIC - Add rolling average

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Train a simple ML model (scikit-learn)
# MAGIC - Goal: predict `is_fast` (1 = fast test).
# MAGIC - Use a small dataset, so keep expectations realistic.

# COMMAND ----------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Collect to pandas for scikit-learn
train_pdf = feature_df.select(
    "avg_distance", "min_distance", "max_distance", "var_distance",
    "mean_step_interval", "step_interval_var", "is_fast"
).toPandas().fillna(0)

X = train_pdf.drop(columns=["is_fast"])
y = train_pdf["is_fast"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

print("Accuracy", accuracy_score(y_test, preds))
print("Precision", precision_score(y_test, preds, zero_division=0))
print("Recall", recall_score(y_test, preds, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, preds))

# COMMAND ----------
# MAGIC %md
# MAGIC 🧪 **Try it yourself:** change the train/test split
# MAGIC - Try `test_size=0.5`
# MAGIC - Try `random_state=123`

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Model interpretation
# MAGIC - **Feature importance:** coefficients from Logistic Regression.
# MAGIC - **Global view:** which features matter overall.
# MAGIC - **Local view:** why one prediction was made.

# COMMAND ----------
import matplotlib.pyplot as plt

importance = model.coef_[0]
features = list(X.columns)

plt.figure(figsize=(6,4))
plt.barh(features, importance)
plt.title("Feature importance (coefficients)")
plt.xlabel("weight")
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Optional: SHAP values (if installed)
# MAGIC - Shows impact of each feature on each prediction.
# MAGIC - Works best with more data; here it is just a demo.

# COMMAND ----------
try:
    import shap
    explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test)

    # Global summary plot
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    # Local explanation for first row
    shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
except ImportError:
    print("SHAP not installed in this workspace. Skipping plot.")

# COMMAND ----------
# MAGIC %md
# MAGIC 🧪 **Try it yourself:** add one more limitation or risk to the list below.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Ethics, humility, and responsible AI
# MAGIC - This simple model **is not** a medical device. ⚠️
# MAGIC - Be honest about accuracy and limits.
# MAGIC - Be fair: does it work the same for all ages or body types?
# MAGIC - Protect privacy of sensor data.
# MAGIC - Check bias: more data may be needed for older adults or people with disabilities.
# MAGIC 
# MAGIC **Gospel thought:**
# MAGIC - Paul taught that the body has many members working together (1 Corinthians 12).
# MAGIC - Like the body, data scientists, engineers, and clinicians must work together.
# MAGIC - Matthew 7:16 says, "Ye shall know them by their fruits." Our model's "fruits" are its real-world impacts.
# MAGIC 
# MAGIC **Discuss (bullet prompts):**
# MAGIC - What good could this model do? (training aid? early warning?)
# MAGIC - How could it be misused? (over-trust? ignore doctors?)
# MAGIC - How do we measure fairness here?
# MAGIC - How can we stay humble when results look strong?

# COMMAND ----------
# MAGIC %md
# MAGIC ### Wrap-up
# MAGIC - We loaded sensor and test data.
# MAGIC - We explored, joined, and engineered features.
# MAGIC - We trained and explained a simple model.
# MAGIC - Next: collect real data, validate more, and compare models.

# COMMAND ----------
