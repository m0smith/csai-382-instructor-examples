# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # CSAI-382 Assignment 6.3: Annotated Data Stories (Databricks)
# MAGIC 
# MAGIC Welcome! We will live-code together. Keep sentences short, ask questions often.
# MAGIC 
# MAGIC **Scenario:** STEDI Balance Device measures how well a person can step in place (Rapid Step Test).
# MAGIC 
# MAGIC **Learning goals (today):**
# MAGIC - ✅ Load sample STEDI data (device readings + test summary)
# MAGIC - ✅ Explore data with pandas/Spark
# MAGIC - ✅ Join sensor data to test results
# MAGIC - ✅ Build simple features for balance risk
# MAGIC - ✅ Train and explain a tiny ML model
# MAGIC - ✅ Think about responsible AI and gospel insights
# MAGIC
# MAGIC *Tip:* Cells that start with `%md` are Markdown (text). Cells without `%` are Python. Use the ▶️ play button to run each cell.
# COMMAND ----------
# MAGIC %md
# MAGIC ### Quick "bad data story" vs. annotated version
# MAGIC - **Bad example (what not to do):** show a chart with no labels, give a vague claim.
# MAGIC - **Better example:** same chart, but add a clear title, labels, and a short takeaway.
# MAGIC - Use this slide to discuss why context and annotations matter.
# COMMAND ----------
import matplotlib.pyplot as plt

months = ["Jan", "Feb", "Mar", "Apr"]
values = [30, 45, 25, 50]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# Bad story: no labels, vague note
axes[0].bar(months, values, color="gray")
axes[0].set_title("Vague chart")
axes[0].text(0.1, 48, "Looks fine?", fontsize=10)

# Annotated story: labels + specific takeaway
axes[1].bar(months, values, color="steelblue")
axes[1].set_title("Clinic visits by month")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("# Visits")
axes[1].axhline(40, color="red", linestyle="--", linewidth=1)
axes[1].text(2.2, 42, "🔴 Feb spike—check staffing", color="red", fontsize=9)
axes[1].text(1.0, 12, "Add note: Promo in Feb?", fontsize=9)

plt.suptitle("Bad vs. annotated data story", fontsize=14)
plt.tight_layout()
plt.show()
# COMMAND ----------
# MAGIC %md
# MAGIC ### Stakeholder persona (communication target)
# MAGIC - **Audience:** Clinic Director (non-technical)
# MAGIC - **Cares about:** Safety, false alarms, trust
# MAGIC - **Does NOT care about:** Hyperparameters
# MAGIC - **Tip:** Keep language plain, connect to patient impact.
# COMMAND ----------
# MAGIC %md
# MAGIC ## 1) Quick intro: STEDI Rapid Step Test
# MAGIC - The device records **distance** of each step over time.
# MAGIC - The Rapid Step Test has a **startTime**, **testTime** (seconds), **totalSteps**, and **stepPoints** (timestamps of each step).
# MAGIC - We want to see if sensor signals can show balance or fall risk.
# MAGIC 
# MAGIC *Instructor note:* Pause to ask: "What could go wrong if we trust the model too much?"
# COMMAND ----------
# MAGIC %md
# MAGIC ## 2) Load data into Databricks
# MAGIC We may not have real files, so we create small **synthetic** tables. Same code works for CSV/Delta.
# MAGIC 
# MAGIC Example file read (if you have a path):
# MAGIC ```python
# MAGIC df = spark.read.csv('/FileStore/device_message.csv', header=True, inferSchema=True)
# MAGIC df_delta = spark.read.format('delta').load('/delta/rapid_step_test')
# MAGIC ```
# MAGIC Now, let's make sample data in-memory.
# COMMAND ----------
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Small helper to create timestamps
start = datetime(2024, 1, 1, 12, 0, 0)

device_data = pd.DataFrame({
    "deviceId": ["A"] * 8 + ["B"] * 8,
    "eventTime": [start + timedelta(seconds=i*5) for i in range(16)],
    "distance": [0.9, 1.0, 1.1, 1.05, 1.2, 1.15, 1.1, 1.05, 0.6, 0.65, 0.7, 0.68, 0.72, 0.7, 0.65, 0.6]
})

rapid_step = pd.DataFrame({
    "deviceId": ["A", "B"],
    "startTime": [start, start + timedelta(minutes=1)],
    "testTime": [40, 50],  # seconds
    "totalSteps": [20, 16],
    "stepPoints": ["0,5,10,15,20,25,30,35", "0,6,12,18,24,30,36,42"]
})

print("Pandas device_data rows:", len(device_data))
print("Pandas rapid_step rows:", len(rapid_step))
# COMMAND ----------
# MAGIC %md
# MAGIC ### Convert to Spark (optional, but good for big data)
# MAGIC - Spark DataFrames scale to large files.
# MAGIC - Pandas is easier for small demos.
# MAGIC - We can switch between them.
# COMMAND ----------
# Convert pandas to Spark
device_sdf = spark.createDataFrame(device_data)
rapid_step_sdf = spark.createDataFrame(rapid_step)

device_sdf.show(5, truncate=False)
rapid_step_sdf.show(truncate=False)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 3) Explore the data (EDA)
# MAGIC 
# MAGIC Tools we will use:
# MAGIC - `display(df)` → pretty table or chart
# MAGIC - `.printSchema()` (Spark) or `.info()` (pandas) → columns + types
# MAGIC - `.describe()` → quick stats (mean, std, min, max)
# MAGIC 
# MAGIC *Instructor:* Explain why stats matter for ML (scale, outliers, missing values).
# COMMAND ----------
# Pandas info and describe
print("=== Pandas device_data info ===")
device_data.info()
print("\n=== Pandas device_data stats ===")
print(device_data.describe())

print("\n=== Pandas rapid_step info ===")
rapid_step.info()
print("\n=== Pandas rapid_step stats ===")
print(rapid_step.describe())
# COMMAND ----------
# Spark schema and stats
device_sdf.printSchema()
rapid_step_sdf.printSchema()

print("\nSpark describe distance:")
device_sdf.describe("distance").show()
# COMMAND ----------
# MAGIC %md
# MAGIC ### Visual check with `display`
# MAGIC In Databricks, `display(df)` gives quick charts. Great for spotting issues.
# COMMAND ----------
# display(device_sdf)  # Uncomment in Databricks for a table or chart
# COMMAND ----------
# MAGIC %md
# MAGIC ## 4) Join sensor data with test summary
# MAGIC We join on `deviceId`. For time, we use a simple rule: keep sensor rows 0-60 seconds after `startTime`.
# MAGIC 
# MAGIC This gives us one table to build features (a **feature table** = columns we feed to the model).
# COMMAND ----------
from pyspark.sql import functions as F

# Add a simple end time for each test
rapid_step_sdf = rapid_step_sdf.withColumn("stopTime", F.col("startTime") + F.expr("INTERVAL testTime SECONDS"))

# Join: deviceId match AND eventTime between start and stop
joined = device_sdf.join(
    rapid_step_sdf,
    on=["deviceId"],
    how="inner"
).where((F.col("eventTime") >= F.col("startTime")) & (F.col("eventTime") <= F.col("stopTime")))

print("Joined rows:", joined.count())
joined.orderBy("deviceId", "eventTime").show(truncate=False)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 5) Simple feature engineering
# MAGIC We compute per-test features:
# MAGIC - average distance
# MAGIC - min/max distance
# MAGIC - variance (spread) of distance
# MAGIC - mean step interval (from stepPoints list)
# MAGIC 
# MAGIC Why? These may tell us about balance stability.
# COMMAND ----------
# Helper to split stepPoints string into intervals
def mean_step_interval(step_points_str):
    times = [float(x) for x in step_points_str.split(',') if x]
    if len(times) < 2:
        return None
    diffs = np.diff(times)
    return float(np.mean(diffs))

# Register as pandas UDF if needed; here we do it in pandas first
rapid_step["meanStepInterval"] = rapid_step["stepPoints"].apply(mean_step_interval)
rapid_step_sdf = spark.createDataFrame(rapid_step)

# Aggregate sensor features per device/test
agg_features = joined.groupBy("deviceId", "startTime", "testTime", "totalSteps", "stepPoints").agg(
    F.avg("distance").alias("avgDistance"),
    F.min("distance").alias("minDistance"),
    F.max("distance").alias("maxDistance"),
    F.variance("distance").alias("varDistance")
)

# Add mean step interval by joining back
features = agg_features.join(rapid_step_sdf.select("deviceId", "startTime", "meanStepInterval"), on=["deviceId", "startTime"], how="left")

features.show(truncate=False)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 6) Train a simple ML model (scikit-learn)
# MAGIC - Label: **fast vs slow** test. Threshold at 45 seconds.
# MAGIC - Model: Logistic Regression (good for small, binary labels).
# MAGIC - Steps: split → fit → predict → evaluate.
# COMMAND ----------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Convert Spark to pandas for scikit-learn
features_pd = features.toPandas()

# Label: 1 = fast (under 45s), 0 = slow
features_pd["label_fast"] = (features_pd["testTime"] < 45).astype(int)

feature_cols = ["avgDistance", "minDistance", "maxDistance", "varDistance", "meanStepInterval"]
X = features_pd[feature_cols]
y = features_pd["label_fast"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("log_reg", LogisticRegression())
])

pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)

acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, zero_division=0)
rec = recall_score(y_test, preds, zero_division=0)
cm = confusion_matrix(y_test, preds)

print("Accuracy", acc)
print("Precision", prec)
print("Recall", rec)
print("Confusion matrix:\n", cm)
# COMMAND ----------
# MAGIC %md
# MAGIC ### Reflection before explainability
# MAGIC If most training data comes from younger users, what might go wrong? Write one risk in plain language (no code) before we look at SHAP.
# COMMAND ----------
# MAGIC %md
# MAGIC ## 7) Model interpretation
# MAGIC - **Feature importance:** how much each feature moves the prediction (absolute logistic weights here).
# MAGIC - **Global vs local:** global = overall patterns; local = one row's explanation.
# MAGIC - If `shap` is installed, we show a quick plot (may be small for demo).
# COMMAND ----------
import matplotlib.pyplot as plt
import numpy as np

# Get absolute coefficients from logistic regression
coef = pipe.named_steps["log_reg"].coef_[0]
importance = np.abs(coef)

plt.figure(figsize=(6,4))
plt.bar(feature_cols, importance, color="skyblue")
plt.xticks(rotation=30)
plt.title("Feature importance (abs weight)")
plt.ylabel("Weight")
plt.tight_layout()
plt.show()
# COMMAND ----------
# MAGIC %md
# MAGIC ### Optional: SHAP (if available)
# MAGIC - SHAP shows how each feature pushes a prediction.
# MAGIC - Global summary = overall influence; Local = one prediction.
# MAGIC - If import fails, explain conceptually and move on.
# MAGIC - **Legend:** Red = pushes prediction higher. Blue = pushes prediction lower. Size shows strength.
# COMMAND ----------
try:
    import shap
    shap.enable_js(display=False)

    explainer = shap.Explainer(pipe.named_steps["log_reg"], X_train)
    shap_values = explainer(X_test)

    # Global summary
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary (global)")
    plt.tight_layout()
    plt.show()

    # Local explanation for first test sample
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("SHAP local explanation (first test row)")
    plt.show()
except Exception as e:
    print("SHAP not available or failed:", e)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 8) Ethics & responsible AI (short reflection)
# MAGIC - ⚠️ This is **not** a medical device. Do not over-claim.
# MAGIC - Ask: Could this be unfair to some groups (age, mobility, injury)?
# MAGIC - Ask: What data is missing? How could that bias results?
# MAGIC - Ask: How will we keep humans in the loop? (coach, nurse, student)
# MAGIC - Ask: How do we communicate limits? (precision/recall, small sample)
# MAGIC
# MAGIC **Gospel thought:** "By their fruits ye shall know them" (Matthew 7:16). We judge models by outcomes: Are we serving people with love and humility? Are we honest about risks?
# COMMAND ----------
# MAGIC %md
# MAGIC ### What would you say out loud?
# MAGIC Imagine you have **60 seconds** with a manager. In one or two sentences, what would you say about this model's strengths, risks, and next step?
# COMMAND ----------
# MAGIC %md
# MAGIC ## 9) 🧪 Try it yourself prompts
# MAGIC - 🧪 Try it yourself: change the feature set (remove one, add another).
# MAGIC - 🧪 Try it yourself: change the train/test split (e.g., 0.2 or 0.5) and see metrics.
# MAGIC - 🧪 Try it yourself: add one more limitation or risk to the list above.
# MAGIC 
# MAGIC *Instructor tip:* Pause after each prompt. Let students edit the code and run.
# COMMAND ----------
# MAGIC %md
# MAGIC ### Bonus: Save feature table as Delta (optional)
# MAGIC If you have DBFS access, you can save features for later use.
# MAGIC ```python
# MAGIC features.write.format("delta").mode("overwrite").save("/tmp/stedi_features")
# MAGIC ```
# MAGIC Use `spark.read.format('delta').load('/tmp/stedi_features')` to load again.
