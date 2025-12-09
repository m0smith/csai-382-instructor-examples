# Databricks notebook source
# MAGIC %md
# MAGIC # Week 7 – 7.6 Pipeline Documentation (CSAI-382)
# MAGIC 
# MAGIC Welcome! This live demo shows how to document machine learning pipelines in Databricks. We keep examples small so you can run them quickly in class.
# MAGIC 
# MAGIC **Learning outcomes**
# MAGIC - Describe why pipeline documentation helps teams and future you.
# MAGIC - Capture pipeline steps, parameters, and data lineage with simple tools.
# MAGIC - Practice combining Python and SQL notes inside Databricks.
# MAGIC - Spot common documentation mistakes and how to avoid them.
# MAGIC 
# MAGIC _This notebook is built for live coding in Databricks._
# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Setup & Imports
# MAGIC 
# MAGIC Libraries we use:
# MAGIC - `pyspark.sql` for small Spark DataFrame examples.
# MAGIC - `pandas` for quick, readable tables.
# MAGIC - `textwrap` to format doc strings for clarity.
# MAGIC 
# MAGIC 💡 Databricks clusters already include Spark; we just import what we need.
# COMMAND ----------
# INSTRUCTOR NOTE: Explain why we need this import before you run it.
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import textwrap

spark = SparkSession.builder.getOrCreate()
print("Environment ready for pipeline documentation demo ✅")
# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Data Access (STEDI or Synthetic)
# MAGIC 
# MAGIC The STEDI case is about fall-risk detection from balance tests. We often work with device events like RapidStepTest counts.
# MAGIC 
# MAGIC For a quick demo, we'll use a tiny synthetic DataFrame that mimics a device message table.
# MAGIC 
# MAGIC _If you have the real table, swap in `spark.read.table("<catalog>.<schema>.DeviceMessage")`._
# COMMAND ----------
# INSTRUCTOR NOTE: If the real STEDI table is available, replace this synthetic data with a spark.read.table(...) command.
sample_data = [
    {"participant_id": 101, "step_count": 32, "test_date": "2024-03-01", "device": "watch"},
    {"participant_id": 102, "step_count": 21, "test_date": "2024-03-02", "device": "band"},
    {"participant_id": 103, "step_count": 15, "test_date": "2024-03-02", "device": "watch"},
    {"participant_id": 104, "step_count": 40, "test_date": "2024-03-03", "device": "band"},
    {"participant_id": 105, "step_count": 27, "test_date": "2024-03-03", "device": "watch"},
]

pdf = pd.DataFrame(sample_data)
df = spark.createDataFrame(pdf)

print("Synthetic STEDI-like data")
display(df)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Core Concept Demo – PART 1 (Basic Idea)
# MAGIC 
# MAGIC Pipeline documentation records **what happens** to data from start to finish. Think of it as a travel log: input → transformations → output. Good notes answer:
# MAGIC - What data did we use?
# MAGIC - What steps did we run (and in what order)?
# MAGIC - What parameters or thresholds were chosen?
# MAGIC - Where do outputs live and who owns them?
# MAGIC 
# MAGIC Let's show a tiny pipeline with clear, in-line documentation.
# COMMAND ----------
# Step 1: Define a small transformation with doc comments
# INSTRUCTOR NOTE: Pause here and ask if students have questions.
def normalize_steps(df):
    """
    Normalize step counts by dividing by the max step_count.
    Keeps the schema simple for the demo.
    """
    max_steps = df.agg(F.max("step_count").alias("max_sc")).collect()[0]["max_sc"]
    return df.withColumn("norm_step", F.col("step_count") / F.lit(max_steps))

# Step 2: Apply and show the result
normalized_df = normalize_steps(df)
print("Pipeline stage: normalize step counts")
display(normalized_df)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Core Concept Demo – PART 2 (Slightly More Realistic)
# MAGIC 
# MAGIC Now we simulate a mini pipeline with two stages and capture the documentation in a simple table. This mirrors STEDI work where we track transformations and ownership.
# MAGIC 
# MAGIC We'll add a classification flag and record each step in a documentation log.
# COMMAND ----------
# INSTRUCTOR NOTE: Ask students to predict what this line will do before you run it.
from datetime import datetime

doc_log = []

# Step 1: Filter out low-quality tests (e.g., too few steps)
threshold = 20
filtered_df = df.filter(F.col("step_count") >= threshold)
doc_log.append({
    "step": 1,
    "name": "filter_low_steps",
    "description": f"Keep records with step_count >= {threshold}",
    "owner": "data_eng",
    "timestamp": datetime.utcnow().isoformat()
})

# Step 2: Add a simple risk flag for illustration
flagged_df = filtered_df.withColumn(
    "risk_flag",
    F.when(F.col("step_count") < 25, F.lit("high")).otherwise(F.lit("medium"))
)
doc_log.append({
    "step": 2,
    "name": "add_risk_flag",
    "description": "Mark high risk when step_count < 25",
    "owner": "ml_team",
    "timestamp": datetime.utcnow().isoformat()
})

print("Documented pipeline output")
display(flagged_df)

print("Pipeline documentation log (pandas for readability)")
display(pd.DataFrame(doc_log).sort_values("step"))
# COMMAND ----------
# MAGIC %md
# MAGIC ### SQL View of Documentation
# MAGIC We can store pipeline docs in a Delta table. Here we use a temp view so it stays in session.
# COMMAND ----------
doc_df = spark.createDataFrame(pd.DataFrame(doc_log))
doc_df.createOrReplaceTempView("pipeline_docs")

# INSTRUCTOR NOTE: This is a good place to show how to inspect the schema.
print(doc_df.printSchema())
# COMMAND ----------
%sql
-- A SQL-friendly view of the pipeline documentation
SELECT step, name, description, owner, timestamp
FROM pipeline_docs
ORDER BY step;
# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Common Mistakes & Debugging
# MAGIC 
# MAGIC Common issues when documenting pipelines:
# MAGIC 1. **Forgetting the order of steps** – logs without step numbers make reproducing work hard.
# MAGIC 2. **Missing parameter values** – notes like "filtered data" are too vague.
# MAGIC 3. **Saving docs separately** – storing docs away from code means they drift apart.
# MAGIC 
# MAGIC Let's see a broken example and a fixed one.
# COMMAND ----------
# INSTRUCTOR NOTE: Run the broken code first, ask students what the error means.
# Broken: missing the column will trigger an analysis error
try:
    broken_df = df.withColumn("norm_again", F.col("step_counts") / 40)
    display(broken_df)
except Exception as e:
    print("⚠️ Error happened:", e)

# INSTRUCTOR NOTE: Then show the fixed code and explain the difference.
fixed_df = df.withColumn("norm_again", F.col("step_count") / 40)
print("Fixed version with correct column name")
display(fixed_df)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Ethics & Responsible AI Tie-In
# MAGIC 
# MAGIC Documenting pipelines is part of being a trusted steward of data. Clear records help teams spot biased steps, like filters that remove certain groups more often. When we log parameters and owners, we support accountability and fairness checks. Doctrine & Covenants 88:118 reminds us to learn by study and by faith; careful documentation reflects honest learning and respect for those our models affect. By keeping transparent notes, we care for vulnerable users and ensure safety in fall-risk predictions.
# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. "Try It Yourself" Mini Exercises
# MAGIC 
# MAGIC **Challenge 1:** Change the filter threshold to 30 and note how many rows remain. Update the documentation log accordingly.
# MAGIC 
# MAGIC **Challenge 2:** Add a new column `device_flag` that labels `watch` as "wrist" and others as "other". Record this as a pipeline step.
# MAGIC 
# MAGIC **Challenge 3:** Save the documentation log to a temporary view and query it with SQL ordered by timestamp.
# MAGIC 
# MAGIC **Challenge 4:** Write a short doc string for the risk flag step explaining the business meaning.
# MAGIC 
# MAGIC _Try them live! Keep each change small and visible._
# COMMAND ----------
# INSTRUCTOR SOLUTION: Challenge 1
new_threshold = 30
filtered_df2 = df.filter(F.col("step_count") >= new_threshold)
print("Rows after new threshold:", filtered_df2.count())

# Update log
solution_log = doc_log.copy()
solution_log.append({
    "step": 3,
    "name": "filter_update",
    "description": f"Updated filter to step_count >= {new_threshold}",
    "owner": "instructor",
    "timestamp": datetime.utcnow().isoformat()
})
display(filtered_df2)
# COMMAND ----------
# INSTRUCTOR SOLUTION: Challenge 2
solution_log.append({
    "step": 4,
    "name": "add_device_flag",
    "description": "Label watch devices as wrist and others as other",
    "owner": "instructor",
    "timestamp": datetime.utcnow().isoformat()
})
with_device_flag = filtered_df2.withColumn(
    "device_flag",
    F.when(F.col("device") == "watch", F.lit("wrist")).otherwise(F.lit("other"))
)
display(with_device_flag)
# COMMAND ----------
# INSTRUCTOR SOLUTION: Challenge 3
solution_doc_df = spark.createDataFrame(pd.DataFrame(solution_log))
solution_doc_df.createOrReplaceTempView("pipeline_docs_solution")
%sql
SELECT * FROM pipeline_docs_solution ORDER BY timestamp;
# COMMAND ----------
# INSTRUCTOR SOLUTION: Challenge 4
risk_flag_doc = textwrap.dedent(
    """
    Risk flag meaning:
    - "high": participant step_count below 25, may need coaching.
    - "medium": step_count at or above 25, continue normal monitoring.
    Keep this with the code so future teams know the business logic.
    """
)
print(risk_flag_doc)
# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Summary & Key Takeaways
# MAGIC 
# MAGIC - You saw how to document pipeline steps with clear names, owners, and timestamps.
# MAGIC - You practiced logging parameters like thresholds to improve reproducibility.
# MAGIC - You learned to keep docs close to code using pandas, Spark, and SQL views.
# MAGIC - You spotted common mistakes such as missing columns or unclear filters.
# MAGIC - Transparent pipeline notes support fairness, safety, and stewardship in AI projects.
# MAGIC 
# MAGIC Keep experimenting! Small, clear notes today prevent big headaches tomorrow. ✅
