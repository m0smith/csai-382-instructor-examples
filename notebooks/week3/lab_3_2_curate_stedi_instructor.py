# Databricks notebook source
# COMMAND ----------
%md
# 3.2 Instructor Demo — Curate STEDI Dataset (ETL & Timestamp Alignment)

Welcome! This notebook is designed for **instructors** to live-code the ETL steps for Assignment **3.2 ETL Pipelines: Curate Dataset**. It walks through how to combine raw STEDI sensor data with rapid step tests, add labels, and save a curated Silver table for later ML work.

**Learning objectives (instructor lens)**
- Understand the two STEDI datasets and their roles in labeling.
- Demonstrate joining time-series data using timestamps and device IDs.
- Show how to label sensor rows as `step` vs. `no_step`, and differentiate `source_label`.
- Save a curated Silver table and verify alignment with simple queries.

**Instructor Tip:** Pause after each concept block and ask students to predict the next step (e.g., "What will the join do?", "Will the boundary be inclusive?").

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Setup and Environment
# MAGIC 
# MAGIC We import PySpark helpers and set configurable variables. Change the catalog and schema **before class** so the demos run.
# MAGIC 
# MAGIC **Instructor Tip:** Type the catalog/schema live so students see where table names come from.

# COMMAND ----------
from pyspark.sql import functions as F
from pyspark.sql import types as T

# Instructor: set these to match your workspace
# Example: target_catalog = "workspace"
# Example: target_schema = "bronze" or "silver"
target_catalog = "YOUR_CATALOG_NAME_HERE"
target_schema = "YOUR_SCHEMA_NAME_HERE"

device_table = f"{target_catalog}.{target_schema}.device_message_raw"
rapid_table = f"{target_catalog}.{target_schema}.rapid_step_test_raw"

print("Configured device table:", device_table)
print("Configured rapid step table:", rapid_table)

# COMMAND ----------
%md
## 2. Understanding the Two Datasets (Schemas & Samples)

- **device_message_raw**: Continuous sensor readings (like a security camera recording every moment).
- **rapid_step_test_raw**: Start/stop windows and step counts for each test (like a log of when something meaningful happened).

**Instructor Tip:** Ask: *Which dataset is probably larger and why?* (Hint: continuous readings are usually bigger.)

# COMMAND ----------
# Show schema and small samples for both tables
print("device_message_raw schema")
spark.read.table(device_table).printSchema()

display(spark.read.table(device_table).limit(10))

print("rapid_step_test_raw schema")
spark.read.table(rapid_table).printSchema()

display(spark.read.table(rapid_table).limit(10))

# COMMAND ----------
%md
### What matters most?
- Timestamps are critical for alignment.
- `deviceId` links the two tables.
- `distance` often arrives as text (e.g., "12cm"), so we will clean it.

# COMMAND ----------
%md
## 3. Toy Example: Aligning Time Ranges and Timestamps

Before the real data, warm up with a tiny example. Think of it like matching moments in a security camera to a visitor log.

Goal: label each sensor row as **inside** or **outside** a session window.

**Instructor Tip:** Have students guess which rows fall inside the window before running the code.

# COMMAND ----------
# Create tiny DataFrames inline
sensor_data = [
    ("2024-01-01 10:00:00", "devA"),
    ("2024-01-01 10:00:02", "devA"),
    ("2024-01-01 10:05:00", "devA"),
]
session_data = [
    ("2024-01-01 10:00:01", "2024-01-01 10:00:03", "devA")
]

sensor_df = spark.createDataFrame(sensor_data, ["timestamp", "deviceId"])
session_df = spark.createDataFrame(session_data, ["startTime", "stopTime", "deviceId"])

print("Raw toy sensor data:")
display(sensor_df)
print("Raw toy session data:")
display(session_df)

# COMMAND ----------
# Cast strings to timestamps to avoid type mismatches
sensor_df_ts = sensor_df.withColumn("timestamp", F.to_timestamp("timestamp"))
session_df_ts = (
    session_df
    .withColumn("startTime", F.to_timestamp("startTime"))
    .withColumn("stopTime", F.to_timestamp("stopTime"))
)

# Cross join + filter by time range, keeping the same deviceId
aligned_toy_df = (
    sensor_df_ts.alias("s")
    .join(session_df_ts.alias("w"), on="deviceId")
    .withColumn(
        "is_in_session",
        F.col("s.timestamp").between(F.col("w.startTime"), F.col("w.stopTime"))
    )
)

print("Toy alignment result:")
display(aligned_toy_df)

# COMMAND ----------
%md
### Toy example reflections
- We used `between(startTime, stopTime)` which is **inclusive** on both ends.
- If timestamps stay as strings, the comparisons can fail or sort incorrectly.
- Timezones matter in real data; keep everything in the same zone.

**Instructor Tip:** Ask: *What if there are overlapping sessions?* Students may suggest window functions or deduping.

# COMMAND ----------
%md
## 4. Cleaning and Preparing the Real STEDI Tables

Now apply the same ideas to the real tables.
- Clean `distance` (e.g., "1cm" → `1.0`).
- Ensure all time fields are proper timestamps.

**Instructor Tip:** Show the “before” and “after” distance column so students see the change.

# COMMAND ----------
# Load the raw tables
device_df = spark.read.table(device_table)
rapid_df = spark.read.table(rapid_table)

# Clean the distance column without overwriting the original
# Keep the raw column for debugging or lineage
# "1cm" -> 1.0 (double)
device_df_clean = device_df.withColumn(
    "distance_cm",
    F.regexp_replace(F.col("distance"), "cm", "").cast("double")
)

# Cast time fields to timestamp to support comparisons
rapid_df_clean = (
    rapid_df
    .withColumn("startTime", F.to_timestamp("startTime"))
    .withColumn("stopTime", F.to_timestamp("stopTime"))
)

device_df_clean = device_df_clean.withColumn("timestamp", F.to_timestamp("timestamp"))

print("After cleaning distance and casting timestamps (device):")
display(device_df_clean.select("timestamp", "deviceId", "distance", "distance_cm").limit(5))

print("After casting timestamps (rapid step test):")
display(rapid_df_clean.select("startTime", "stopTime", "deviceId", "totalSteps").limit(5))

# COMMAND ----------
%md
We create a **new** numeric column instead of overwriting `distance` so we can debug if something looks wrong later.

**Instructor Tip:** If you skip `to_timestamp`, comparisons may silently fail or give wrong ordering—run a quick example to show this.

# COMMAND ----------
%md
## 5. Aligning Sensor Readings with Rapid Step Tests (Real Data)

Goal: determine which sensor readings fall inside a Rapid Step Test window.

Strategy:
1. Join on `deviceId`.
2. Filter where `device_timestamp BETWEEN startTime AND stopTime`.
3. Label matching rows as `step`, others as `no_step`.

**Instructor Tip:** Mention that a naive cross join can be expensive; filters keep it smaller.

# COMMAND ----------
# Alias for readability
s = device_df_clean.alias("s")
w = rapid_df_clean.alias("w")

# Join sensor rows to windows on deviceId, then filter by time range
sensor_with_window = (
    s.join(w, on="deviceId", how="left")
     .withColumn(
         "step_label",
         F.when(
             F.col("s.timestamp").between(F.col("w.startTime"), F.col("w.stopTime")),
             F.lit("step")
         ).otherwise(F.lit("no_step"))
     )
)

# Optional: keep only necessary columns for clarity
labeled_steps_df = sensor_with_window.select(
    F.col("s.timestamp").alias("sensor_timestamp"),
    F.col("s.deviceId"),
    F.col("s.sensorType"),
    F.col("s.distance_cm"),
    F.col("w.startTime"),
    F.col("w.stopTime"),
    F.col("w.totalSteps"),
    "step_label"
)

print("Labeled sensor readings:")
display(labeled_steps_df.limit(10))

# COMMAND ----------
%md
- `between(startTime, stopTime)` includes both start and stop.
- Rows with no matching window become `no_step` because of the `otherwise` clause.
- If you suspect unlabeled rows, count them!

**Instructor Tip:** Try `labeled_steps_df.filter(F.col("step_label") == "no_step").count()` live to show counts.

# COMMAND ----------
%md
## 6. Creating the `source_label` Column ("device" vs. "step")

We want to know **where** each row came from.
- Sensor readings → `source_label = "device"`
- (Optional) aggregated step records → `source_label = "step"`

This helps during ML feature engineering to trace lineage.

**Instructor Tip:** Clarify that `step_label` is about **time inside a step window**, while `source_label` is about **data origin**.

# COMMAND ----------
# Add source_label for device readings
labeled_with_source_df = labeled_steps_df.withColumn("source_label", F.lit("device"))

# Optional: create a summarized step-level DataFrame to union (for demonstration)
step_summary_df = rapid_df_clean.select(
    F.col("deviceId"),
    F.col("startTime").alias("sensor_timestamp"),
    F.col("totalSteps"),
    F.lit("step").alias("step_label"),
    F.lit("step").alias("source_label")
)

# Union the two for a combined view (device rows + step summaries)
combined_df = labeled_with_source_df.select("sensor_timestamp", "deviceId", "step_label", "source_label", "distance_cm", "totalSteps") \
    .unionByName(step_summary_df.select("sensor_timestamp", "deviceId", "step_label", "source_label", F.lit(None).cast("double"), "totalSteps"))

print("Combined dataset with source labels:")
display(combined_df.limit(10))

# COMMAND ----------
%md
## 7. Handling Common Student Mistakes (Explicit Examples)

### Common Mistakes and How to Demonstrate Them
1. **Misaligned timestamps** (strings vs. timestamps).
   - Wrong:
     ```python
     # Fails silently because strings compare lexicographically
     device_df.join(rapid_df, "deviceId").filter("timestamp between startTime and stopTime")
     ```
   - Correct:
     ```python
     device_df_clean.join(rapid_df_clean, "deviceId").filter(F.col("timestamp").between("startTime", "stopTime"))
     ```
   **Instructor Tip:** Run the wrong example and show it returns zero matches.

2. **Forgetting to cast start/stop to timestamps.**
   - Wrong: `rapid_df` without `to_timestamp`.
   - Correct: `rapid_df_clean = rapid_df.withColumn("startTime", F.to_timestamp("startTime"))...`

3. **Labeling everything as "step" due to join logic.**
   - Wrong: using an `inner join` without the time filter.
   - Correct: `left join` + `when()` to set `no_step` when no match.

4. **Not cleaning the distance column.**
   - Wrong: using `distance` as a string -> math fails.
   - Correct: `regexp_replace("distance", "cm", "").cast("double")`.

5. **Accidentally performing a cross join.**
   - Wrong:
     ```python
     sensor_df_ts.crossJoin(session_df_ts).filter(...)
     ```
   - Fix: join on `deviceId` first, then filter by time range.

6. **Saving the table in the wrong catalog/schema.**
   - Fix: always reference `target_catalog` and `target_schema` variables.

7. **Weak verification queries.**
   - Fix: count `step` vs `no_step`, and spot-check device windows.

**Instructor Tip:** Encourage students to debug by printing counts after each step.

# COMMAND ----------
%md
## 8. Saving the Curated Dataset

A **Silver table** is clean, reliable data ready for analytics/ML, with lineage preserved. We will overwrite for teaching simplicity.

**Instructor Tip:** Emphasize checking the catalog/schema before writing.

# COMMAND ----------
curated_table = f"{target_catalog}.{target_schema}.stedi_curated_steps"

# Save as a managed table (overwrite during demos)
(labeled_with_source_df
    .write
    .mode("overwrite")
    .saveAsTable(curated_table)
)

print(f"Curated table saved to: {curated_table}")

# COMMAND ----------
%md
## 9. Verification Query Demo

Purpose: ensure labels and alignment make sense.

Example checks:
- Count how many `step` vs `no_step` rows.
- Spot-check a single device to see if timestamps line up with window boundaries.

**Instructor Tip:** Ask students what other checks they would add before trusting the data.

# COMMAND ----------
# Count rows by step_label
display(spark.table(curated_table).groupBy("step_label").count())

# Spot-check one deviceId (replace with a known ID from your data)
sample_device = spark.table(curated_table).select("deviceId").limit(1).collect()[0][0]

print("Sample device for spot-check:", sample_device)

device_check_df = spark.table(curated_table).filter(F.col("deviceId") == sample_device)

# Show earliest and latest sensor timestamps for this device
print("Sensor timestamp range for sample device:")
display(
    device_check_df.agg(
        F.min("sensor_timestamp").alias("min_sensor_ts"),
        F.max("sensor_timestamp").alias("max_sensor_ts")
    )
)

# Compare with the original rapid step windows
print("Original rapid step windows for sample device:")
display(
    rapid_df_clean.filter(F.col("deviceId") == sample_device)
                 .select("startTime", "stopTime", "totalSteps")
)

# COMMAND ----------
%md
## 10. Short Ethics Reflection Prompt (Instructor Version)

**Ethics Check: Instructor Notes**
- Labeling fairness: If we mislabel steps, who is affected? (e.g., false alarms or missed activity)
- Identity protection: Mask or remove identifiers before sharing outside the class.
- Avoid medical claims: This is a classroom example, not a clinical tool.

Ask the class:
- “If our labels are wrong 10% of the time, who might be harmed?”
- “How could we anonymize this dataset better?”
- “What’s the difference between showing step frequency and making health diagnoses?”

**Instructor Tip:** Keep this reflection short but meaningful; connect it to responsible ML practices.

