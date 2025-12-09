# Databricks notebook source
# MAGIC %md
# MAGIC # Week 4 – Model Evaluation Metrics (CSAI-382)
# MAGIC 
# MAGIC Welcome to a live coding notebook on **model evaluation for binary classifiers**. We will look at how to read a confusion matrix, compute precision/recall, and connect these ideas to the STEDI fall-risk story.
# MAGIC 
# MAGIC **Learning outcomes**
# MAGIC - Explain what a **confusion matrix** is and why it matters.
# MAGIC - Compute **accuracy, precision, recall, and F1** on a small example.
# MAGIC - Interpret evaluation metrics for a **fall-risk** scenario.
# MAGIC - Spot and avoid common mistakes when calculating metrics.
# MAGIC - Practice live coding in **Databricks** with Python and SQL cells.
# MAGIC 
# MAGIC _This notebook is designed for live demos on Databricks. Keep it interactive and fun!_

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup & Imports
# MAGIC 
# MAGIC We will use a few friendly libraries:
# MAGIC - **pyspark.sql** for small DataFrames that work well in Databricks.
# MAGIC - **pandas** for quick metric calculations and printing.
# MAGIC - **scikit-learn** for handy metric functions.
# MAGIC 
# MAGIC Databricks clusters already include these libraries. Just make sure you attach the notebook to a running cluster before you start.

# COMMAND ----------
# INSTRUCTOR NOTE: Explain why we need this import before you run it.
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

spark = SparkSession.builder.getOrCreate()
print("Environment ready ✅")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Data Access (STEDI context)
# MAGIC 
# MAGIC In STEDI, we predict **fall risk** from sensor data. The real tables include **DeviceMessage** and **RapidStepTest**, but for this live demo we will use a tiny synthetic sample that acts like the output of a model.
# MAGIC 
# MAGIC If you have access to the real table, you can load it with:
# MAGIC 
# MAGIC ```python
# MAGIC spark.read.table("<CATALOG>.<SCHEMA>.<STEDI_TABLE>")
# MAGIC ```
# MAGIC 
# MAGIC We will keep it simple and create a small DataFrame with true labels and predicted labels.

# COMMAND ----------
# INSTRUCTOR NOTE: If the real STEDI table is available, replace this synthetic data with a spark.read.table(...) command.
sample_data = [
    (1, "user_001", 1, 1),
    (2, "user_002", 0, 0),
    (3, "user_003", 1, 0),
    (4, "user_004", 0, 1),
    (5, "user_005", 1, 1),
    (6, "user_006", 0, 0),
    (7, "user_007", 1, 1),
    (8, "user_008", 0, 0)
]
columns = ["event_id", "user_id", "true_label", "predicted_label"]
stedi_df = spark.createDataFrame(sample_data, columns)

# Show the tiny dataset
stedi_df.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Core Concept Demo – PART 1 (Basic Idea)
# MAGIC 
# MAGIC **What is a confusion matrix?**
# MAGIC - It is a 2x2 table for binary classification.
# MAGIC - It counts **True Positives (TP)**, **False Positives (FP)**, **True Negatives (TN)**, and **False Negatives (FN)**.
# MAGIC - From these counts we get accuracy, precision, recall, and F1.
# MAGIC - In STEDI, **Positive** can mean "high fall risk"; **Negative** means "not high risk".

# COMMAND ----------
# Step 1: Collect the small dataset into pandas for easy metric math
pdf = stedi_df.toPandas()
print(pdf)

# Step 2: Build a confusion matrix
cm = confusion_matrix(pdf["true_label"], pdf["predicted_label"])
print("Confusion matrix (rows=true, cols=predicted):")
print(cm)

# Step 3: Compute simple metrics
acc = accuracy_score(pdf["true_label"], pdf["predicted_label"])
prec = precision_score(pdf["true_label"], pdf["predicted_label"])
rec = recall_score(pdf["true_label"], pdf["predicted_label"])
f1 = f1_score(pdf["true_label"], pdf["predicted_label"])

print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
print(f"F1: {f1:.2f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Core Concept Demo – PART 2 (Slightly More Realistic)
# MAGIC 
# MAGIC Now let's make the data look a bit closer to STEDI sensor output. Imagine we have a score between 0 and 1 from a model, and we choose a **threshold** to decide if someone is high risk. Lowering the threshold usually raises recall but might lower precision.

# COMMAND ----------
from pyspark.sql import functions as F

# Create a pseudo score column
stedi_scores_df = stedi_df.withColumn(
    "risk_score",
    F.when(col("user_id") == "user_004", 0.65)
     .when(col("user_id") == "user_003", 0.40)
     .when(col("user_id") == "user_008", 0.15)
     .otherwise(0.85)
)

# INSTRUCTOR NOTE: Ask students to predict what this line will do before you run it.
stedi_thresholded_df = stedi_scores_df.withColumn("predicted_label_new", (col("risk_score") >= 0.6).cast("int"))

display(stedi_thresholded_df)

# COMMAND ----------
# Step 1: Convert to pandas for metric comparison
pdf_scores = stedi_thresholded_df.select("true_label", "predicted_label_new", "risk_score").toPandas()
print(pdf_scores)

# Step 2: Compute metrics with the new threshold
acc_new = accuracy_score(pdf_scores["true_label"], pdf_scores["predicted_label_new"])
prec_new = precision_score(pdf_scores["true_label"], pdf_scores["predicted_label_new"])
rec_new = recall_score(pdf_scores["true_label"], pdf_scores["predicted_label_new"])

print(f"New threshold Accuracy: {acc_new:.2f}")
print(f"New threshold Precision: {prec_new:.2f}")
print(f"New threshold Recall: {rec_new:.2f}")

# Step 3: Highlight the teaching moment
print("\nNotice: Recall increased when we predicted more positives, but precision dropped.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Common Mistakes & Debugging
# MAGIC 
# MAGIC Common mistakes students make:
# MAGIC 1. **Mixing up labels**: Treating 0 as positive and 1 as negative by accident.
# MAGIC 2. **Forgetting to cast**: Comparing strings to integers, which makes metrics wrong.
# MAGIC 3. **Using accuracy alone**: Accuracy hides false negatives in imbalanced data.

# COMMAND ----------
# INSTRUCTOR NOTE: Run the broken code first, ask students what the error means.
try:
    # Wrong: labels are strings here, so metrics misbehave
    broken_cm = confusion_matrix(pdf_scores["true_label"].astype(str), pdf_scores["predicted_label_new"].astype(str))
    print(broken_cm)
    print("This looks okay, but the labels are strings. Precision/recall will be off if we expect 0/1.")
except Exception as e:
    print("Error:", e)

# INSTRUCTOR NOTE: Then show the fixed code and explain the difference.
fixed_cm = confusion_matrix(pdf_scores["true_label"].astype(int), pdf_scores["predicted_label_new"].astype(int))
print("Fixed confusion matrix with integer labels:")
print(fixed_cm)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Ethics & Responsible AI Tie-In
# MAGIC 
# MAGIC Metrics are not just numbers; they affect people. If we only chase high accuracy, we might ignore that false negatives leave some high-risk users without help. Poor thresholds can hide unfairness against certain groups. As **trusted stewards** (Doctrine & Covenants 88:118 encourages learning by study and faith), we should check metrics for bias, test with diverse data, and be honest about model limits. Choosing thresholds carefully shows care for the vulnerable and accountability for safety.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Try It Yourself – Mini Exercises
# MAGIC 
# MAGIC Challenge 1: Change the threshold from 0.60 to 0.50 and see what happens to precision and recall.
# MAGIC 
# MAGIC Challenge 2: Add a new row with `true_label=1` and a low `risk_score` to test false negatives.
# MAGIC 
# MAGIC Challenge 3: Use a SQL cell to count how many predictions are 1 vs 0 in `stedi_thresholded_df`.
# MAGIC 
# MAGIC Challenge 4: Compute F1 for the new threshold using `f1_score`.

# COMMAND ----------
# INSTRUCTOR SOLUTION: Challenge 1
new_threshold_df = stedi_scores_df.withColumn("predicted_label_050", (col("risk_score") >= 0.5).cast("int"))
pdf_thr = new_threshold_df.select("true_label", "predicted_label_050").toPandas()
print("Precision @0.50:", precision_score(pdf_thr["true_label"], pdf_thr["predicted_label_050"]))
print("Recall @0.50:", recall_score(pdf_thr["true_label"], pdf_thr["predicted_label_050"]))

# COMMAND ----------
# INSTRUCTOR SOLUTION: Challenge 2
added_df = stedi_scores_df.unionByName(spark.createDataFrame([(9, "user_009", 1, 0, 0.10)], stedi_scores_df.columns))
pdf_added = added_df.withColumn("predicted_label_new", (col("risk_score") >= 0.6).cast("int")).select("true_label", "predicted_label_new").toPandas()
print("Recall with extra low-score positive:", recall_score(pdf_added["true_label"], pdf_added["predicted_label_new"]))

# COMMAND ----------
# INSTRUCTOR SOLUTION: Challenge 3 (SQL)
stedi_thresholded_df.createOrReplaceTempView("stedi_predictions")

# MAGIC %sql
# MAGIC SELECT predicted_label_new, COUNT(*) AS cnt
# MAGIC FROM stedi_predictions
# MAGIC GROUP BY predicted_label_new

# COMMAND ----------
# INSTRUCTOR SOLUTION: Challenge 4
print("F1 @0.60:", f1_score(pdf_scores["true_label"], pdf_scores["predicted_label_new"]))
print("F1 @0.50:", f1_score(pdf_thr["true_label"], pdf_thr["predicted_label_050"]))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary & Key Takeaways
# MAGIC 
# MAGIC - A **confusion matrix** shows TP, FP, TN, FN counts.
# MAGIC - **Precision** answers: of predicted high-risk cases, how many were correct?
# MAGIC - **Recall** answers: of real high-risk cases, how many did we catch?
# MAGIC - Changing thresholds trades off precision and recall; pick one that matches the business goal.
# MAGIC - Databricks makes it easy to explore metrics with both Python and SQL in one place.
# MAGIC 
# MAGIC Keep practicing! Small experiments like these build confidence and help you teach with clarity.
