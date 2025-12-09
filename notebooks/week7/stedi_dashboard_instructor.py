# Databricks notebook source
# MAGIC %md
# MAGIC # CSAI-382 – Week 7 – STEDI Databricks Dashboard – Instructor Demo
# MAGIC 
# MAGIC This notebook is for **live demos**. It shows how to create queries and visuals that later go into a dashboard. We use very small **toy data** so students focus on ideas, not big datasets.
# MAGIC 
# MAGIC **Gospel thought:** "Ye shall know the truth, and the truth shall make you free." (John 8:32) Dashboards help us **see clearly** so we can make responsible choices and bless others.
# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Setup and Sample Data
# MAGIC 
# MAGIC * Spark lets us work with tables in the cloud.
# MAGIC * In homework, students already have real STEDI tables.
# MAGIC * Here we build **small example tables** in memory for teaching.
# COMMAND ----------
# Import needed libraries
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.getOrCreate()

# Instructor Notes: These are tiny example tables for demo. The real project uses larger, real data from sensors.

# Create labeled_step_test sample data
labeled_data = [
    ("deviceA", 1, 85.0),
    ("deviceA", 0, 20.0),
    ("deviceB", 1, 90.0),
    ("deviceB", 1, 88.5),
    ("deviceC", 0, 18.0),
    ("deviceC", 0, 25.0)
]

labeled_step_test_df = spark.createDataFrame(labeled_data, ["deviceId", "step_label", "distance_cm"])

# Create stedi_model_metrics sample data
metrics_data = [
    ("accuracy", 0.88),
    ("precision", 0.82),
    ("recall", 0.76),
    ("f1", 0.79)
]

stedi_model_metrics_df = spark.createDataFrame(metrics_data, ["metric_name", "value"])

# Register temp views for SQL
labeled_step_test_df.createOrReplaceTempView("labeled_step_test")
stedi_model_metrics_df.createOrReplaceTempView("stedi_model_metrics")

# Show small samples so students can see the data
print("Sample labeled_step_test rows:")
labeled_step_test_df.show()
print("Sample stedi_model_metrics rows:")
stedi_model_metrics_df.show()
# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Data Overview Panel (SQL + Visualization-friendly query)
# MAGIC 
# MAGIC A dashboard needs a **data overview**. We want to see how many rows and how many examples of each label.
# MAGIC 
# MAGIC Checklist:
# MAGIC * Step 1: Write a SQL query.
# MAGIC * Step 2: Check the result.
# MAGIC * Step 3: Turn it into a chart in the UI. 📊
# COMMAND ----------
# MAGIC %sql
# MAGIC -- Count rows and show label distribution
# MAGIC -- Instructor Notes: Grouping by step_label lets us make a bar chart that shows how many "step" vs "no step" examples we have.
# MAGIC SELECT
# MAGIC   COUNT(*) AS total_rows,
# MAGIC   SUM(CASE WHEN step_label = 1 THEN 1 ELSE 0 END) AS steps,
# MAGIC   SUM(CASE WHEN step_label = 0 THEN 1 ELSE 0 END) AS non_steps
# MAGIC FROM labeled_step_test;
# COMMAND ----------
# MAGIC %sql
# MAGIC -- Distribution by label
# MAGIC -- Instructor Notes: This will become a bar chart. Ask: "Why is it helpful to see balance between labels?"
# MAGIC SELECT step_label, COUNT(*) AS count_per_label
# MAGIC FROM labeled_step_test
# MAGIC GROUP BY step_label
# MAGIC ORDER BY step_label;
# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Time / Device Panel Example
# MAGIC 
# MAGIC We can slice by device or time. In STEDI there can be many devices. Dashboards help us compare them.
# COMMAND ----------
# MAGIC %sql
# MAGIC -- Count records per device
# MAGIC -- Instructor Notes: Ask students, "What question does this chart answer?" If one device has very few records, maybe the model learns less about it.
# MAGIC SELECT deviceId, COUNT(*) AS records_per_device
# MAGIC FROM labeled_step_test
# MAGIC GROUP BY deviceId
# MAGIC ORDER BY records_per_device DESC;
# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Model Performance Metrics Panel
# MAGIC 
# MAGIC * Accuracy: how many predictions are correct overall.
# MAGIC * Precision: when the model says "step," how often is it right.
# MAGIC * Recall: how many real steps we catch.
# MAGIC 
# MAGIC Dashboards often show these metrics as a bar chart from a table like `stedi_model_metrics`.
# COMMAND ----------
# Show the metrics table
print("Model metrics for quick display:")
stedi_model_metrics_df.show()

# Instructor Notes: Ask which metric matters most for fall risk. Discuss why high accuracy alone is not enough.

# Quick display for bar plot in Databricks UI
# Use display() for Databricks to pick a chart. Students can choose a bar chart.
display(stedi_model_metrics_df)
# COMMAND ----------
# MAGIC %md
# MAGIC **Ethics in AI:** If we miss many real steps (low recall), we may give people false confidence about their balance. This can be unsafe. We must think about people, not just numbers. ⚠️
# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Confusion Matrix Example
# MAGIC 
# MAGIC * True Positive (TP): actual 1, predicted 1.
# MAGIC * True Negative (TN): actual 0, predicted 0.
# MAGIC * False Positive (FP): actual 0, predicted 1.
# MAGIC * False Negative (FN): actual 1, predicted 0.
# MAGIC 
# MAGIC A confusion matrix can be stored as a table like `stedi_predictions`.
# COMMAND ----------
# Create sample predictions
predictions_data = [
    (1, 1),  # TP
    (1, 1),  # TP
    (1, 0),  # FN
    (0, 0),  # TN
    (0, 0),  # TN
    (0, 1)   # FP
]

stedi_predictions_df = spark.createDataFrame(predictions_data, ["actual_label", "predicted_label"])
stedi_predictions_df.createOrReplaceTempView("stedi_predictions")

print("Sample predictions:")
stedi_predictions_df.show()
# COMMAND ----------
# MAGIC %sql
# MAGIC -- Build confusion matrix counts
# MAGIC -- Instructor Notes: This can become a heatmap. Ask: "Which type of error is most dangerous here?"
# MAGIC SELECT actual_label, predicted_label, COUNT(*) AS count
# MAGIC FROM stedi_predictions
# MAGIC GROUP BY actual_label, predicted_label
# MAGIC ORDER BY actual_label DESC, predicted_label DESC;
# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Feature Importance / SHAP-style Panel
# MAGIC 
# MAGIC Feature importance shows which input features matter most to the model. In real projects, students may have SHAP plots. In this demo we use a simple table.
# COMMAND ----------
# Create feature importance sample data
feature_importance_data = [
    ("distance_cm", 0.45),
    ("avg_step_time_ms", 0.30),
    ("step_variance", 0.25)
]

feature_importance_df = spark.createDataFrame(feature_importance_data, ["feature_name", "importance"])
feature_importance_df.createOrReplaceTempView("feature_importance")

print("Feature importance table:")
feature_importance_df.show()
# COMMAND ----------
# MAGIC %sql
# MAGIC -- Order features by importance
# MAGIC -- Instructor Notes: Turn this into a bar chart. Ask: "Does this match our intuition? If a feature seems wrong, what should we do?"
# MAGIC SELECT *
# MAGIC FROM feature_importance
# MAGIC ORDER BY importance DESC;
# COMMAND ----------
# MAGIC %md
# MAGIC **Ethics note:** If our model uses data that is noisy or not related to true fall risk, we may make unfair or unsafe decisions. Responsible AI means checking if important features make sense. 💡
# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Bias & Fairness Panel (Reflection)
# MAGIC 
# MAGIC We do **not** have personal demographic data here, but we still think about fairness:
# MAGIC * Are certain types of movements misclassified more?
# MAGIC * Are borderline distances more often wrong?
# MAGIC * Do some devices act differently?
# MAGIC 
# MAGIC Reflection questions:
# MAGIC * Where might bias come from in STEDI data?
# MAGIC * What happens if all our data comes from young athletes, but we apply it to seniors?
# MAGIC * How can we test for these risks?
# MAGIC 
# MAGIC **Gospel thought:** Each person is a child of God. We avoid harmful assumptions and act as honest, careful stewards of data and technology.
# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Putting It All Together for a Dashboard
# MAGIC 
# MAGIC Building blocks:
# MAGIC * Data overview queries.
# MAGIC * Device/time views.
# MAGIC * Metrics table.
# MAGIC * Confusion matrix.
# MAGIC * Feature importance.
# MAGIC * Bias and fairness reflection.
# MAGIC 
# MAGIC Simple steps to demo:
# MAGIC 1. Run a `%sql` cell.
# MAGIC 2. Click **Visualization**.
# MAGIC 3. Choose bar/heatmap/table.
# MAGIC 4. Click **Pin to dashboard**. ✅
# COMMAND ----------
# Quick instructor reminder
# Instructor Notes: Do a 5–10 minute walkthrough.
# 1) Create one query.
# 2) Turn it into a chart.
# 3) Pin it.
# Encourage students to build their own dashboard with the same pattern.
