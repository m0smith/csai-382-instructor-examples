# Databricks notebook source
# COMMAND ----------
%md
# 2.6 Metadata Catalog — INSTRUCTOR NOTEBOOK

*Assignment duration: ~30 minutes*

**Goal:** Help students explore Bronze tables, design a clean Silver schema, verify it via information_schema, and export the schema as CSV.

**Why this matters for ML:** Reliable features start with trustworthy metadata. Knowing lineage (Bronze → Silver) and column intent prevents downstream label leakage, schema drift, and mis-specified models.

**What we will walk through in this notebook**
- Quick overview of the assignment and ML relevance.
- Guided exploration of two Bronze tables.
- Designing a clear Silver schema for `labeled_step_test`.
- Exemplary CREATE TABLE DDL.
- Optional ETL logic patterns (distance parsing, time-window joins, labeling).
- Querying `workspace.information_schema.columns` to verify schema.
- Steps to export the schema as CSV.
- Troubleshooting and discussion prompts.

**7 common student struggles**
1. Confusing Bronze vs. Silver roles and forgetting the catalog/schema prefix.
2. Not understanding what each Bronze column means before selecting Silver fields.
3. Over/under-selecting Silver fields (omitting lineage fields or adding unknowns).
4. Struggling to convert distance strings to numeric values.
5. Getting time-window logic wrong when labeling steps.
6. SQL `CREATE TABLE` syntax errors or using the wrong schema name.
7. Filtering the information schema with the wrong table_schema/table_name (empty results).

# COMMAND ----------
%md
## Section 2 – Exploring Bronze Tables (Instructor Demo)

**What “Bronze” means here (med-tech / sensor data):**
- Raw, minimally processed device emissions and test logs.
- Contains inconsistencies (string numbers, mixed casing, potential nulls).
- Primary purpose: preserve original fidelity for lineage and re-processing.

**Instructor prompts while exploring**
- “Which columns look like timestamps? Are they consistent?”
- “Where do you see device identifiers? Are they strings or numbers?”
- “Any obvious dirty values (units, typos, missing data)?”
- “How would these fields help us label steps?”

# COMMAND ----------
%sql
SELECT *
FROM workspace.bronze.raw_device_message
LIMIT 10;

# COMMAND ----------
%sql
SELECT *
FROM workspace.bronze.raw_rapid_step_test
LIMIT 10;

# COMMAND ----------
%md
### Model answers: 2–3 sentence summaries

- **workspace.bronze.raw_device_message**: Stores individual sensor readings with timestamps, device IDs, sensor types, and distance values (often as strings like "12cm"). Useful for fine-grained signal analysis and labeling each reading. Serves as the primary source for per-reading features and lineage back to raw signals.
- **workspace.bronze.raw_rapid_step_test**: Stores step-test sessions/windows with start/stop times and device IDs that mark when stepping occurs. Provides temporal boundaries to label each sensor reading as "step" vs. "no_step". Enables grouping readings into sessions.

**Acceptable student answers**
- Mention key fields (timestamp, deviceId, sensor info for device messages; start/stop window for rapid step tests).
- Recognize Bronze as raw/unclean data.
- Connect each table to its ML role (features vs. labels/windows).

**Red flags / misconceptions**
- Claiming Bronze is already cleaned/ready for modeling.
- Ignoring time fields or assuming distance is already numeric.
- Not linking deviceId between the two tables.

# COMMAND ----------
%md
## Section 3 – Designing the Silver Schema (Conceptual)

**Silver purpose:** Clean, structured, ML-ready data that keeps lineage. Keep columns only when their meaning is understood.

**Column-by-column guidance**
- `timestamp` (TIMESTAMP, from `raw_device_message`): Sensor reading time; critical for ordering and window joins.
- `deviceId` (STRING/BIGINT, present in both): Key to link readings to test sessions; choose type matching Bronze.
- `sensorType` (STRING, from `raw_device_message`): Identifies which sensor produced the reading; helps feature engineering per sensor.
- `distance_cm` (INT/DOUBLE, from `raw_device_message`, cleaned): Numeric distance for modeling; remove units/characters first.
- `step_label` (STRING, derived via time window): "step" if timestamp within test window, else "no_step"; target label for classification.
- `source` (STRING, added): Notes ingestion source (e.g., `device_message_raw`); aids lineage and auditing.
- `startTime` (TIMESTAMP, from `raw_rapid_step_test`): Window start; supports labeling and quality checks.
- `stopTime` (TIMESTAMP, from `raw_rapid_step_test`): Window end; paired with startTime for interval logic.
- `bronze_record_id` (STRING/BIGINT, from `raw_device_message`): Direct lineage back to Bronze row.
- `testId` (STRING/BIGINT, optional from `raw_rapid_step_test` or generated): Groups readings into sessions; helpful for evaluation splits.

**Required vs. optional**
- Required: `timestamp`, `deviceId`, `sensorType`, `distance_cm`, `step_label`, `source`, `startTime`, `stopTime`, `bronze_record_id`.
- Optional but recommended: `testId` (or similar session identifier).

**Mapping: Bronze → Silver**
- `raw_device_message.timestamp` → `timestamp`
- `raw_device_message.deviceId` → `deviceId`
- `raw_device_message.sensorType` → `sensorType`
- `raw_device_message.distance` (string) → `distance_cm` (clean numeric)
- `raw_device_message.id` (or equivalent key) → `bronze_record_id`
- `raw_rapid_step_test.startTime` → `startTime`
- `raw_rapid_step_test.stopTime` → `stopTime`
- Derived via timestamp between start/stop → `step_label`
- Added literal or pipeline value → `source`
- `raw_rapid_step_test.testId` or generated → `testId`

# COMMAND ----------
%md
## Section 4 – Building the Silver Table (SQL)

High-level approach:
- Join device messages to rapid step tests on `deviceId` and timestamp window (`timestamp BETWEEN startTime AND stopTime`).
- Clean `distance` strings into numeric `distance_cm` using `regexp_replace` or similar; cast to INT/DOUBLE.
- Derive `step_label` based on window membership.
- Decide join strategy: **INNER** yields only matched readings; **LEFT** preserves all readings with possible null labels.

**Instructor note:** Highlight tradeoffs (e.g., missing test windows → null labels). Encourage students to pick a reasonable default and document it.

# COMMAND ----------
%sql
-- Exemplar DDL for the Silver table
CREATE OR REPLACE TABLE workspace.silver.labeled_step_test (
  timestamp        TIMESTAMP COMMENT 'Sensor reading time from raw_device_message',
  deviceId         STRING    COMMENT 'Device identifier linking readings to tests',
  sensorType       STRING    COMMENT 'Sensor type from raw_device_message',
  distance_cm      DOUBLE    COMMENT 'Numeric distance in centimeters, cleaned from raw string',
  step_label       STRING    COMMENT 'Derived label: step vs no_step based on test window',
  source           STRING    COMMENT 'Original data source for lineage',
  startTime        TIMESTAMP COMMENT 'Start time of the rapid step test window',
  stopTime         TIMESTAMP COMMENT 'Stop time of the rapid step test window',
  bronze_record_id STRING    COMMENT 'Lineage key back to raw_device_message',
  testId           STRING    COMMENT 'Optional session identifier from rapid_step_test or generated'
);

# COMMAND ----------
%md
**Common student mistakes**
- Using the wrong schema (e.g., `default.labeled_step_test` instead of `silver`).
- Omitting column comments or using `VARCHAR` instead of `STRING`.
- Forgetting optional fields like `testId` or misnaming columns.
- Mismatched data types (casting distance to INT when decimals exist).
- Not specifying catalog/schema in the table name, causing accidental creation elsewhere.

# COMMAND ----------
%md
## Section 5 – Example ETL Logic (Optional but Helpful)

Use this as a live demo pattern. Students do not need to match code exactly; focus on the concepts (cleaning, joining, labeling).

# COMMAND ----------
%sql
-- 1) Clean distance strings like "12cm" to numeric
WITH cleaned_distance AS (
  SELECT
    timestamp,
    deviceId,
    sensorType,
    CAST(regexp_replace(distance, '[^0-9\\.]', '') AS DOUBLE) AS distance_cm,
    id AS bronze_record_id
  FROM workspace.bronze.raw_device_message
),

-- 2) Join device messages to rapid step tests by deviceId and time window
joined AS (
  SELECT
    c.timestamp,
    c.deviceId,
    c.sensorType,
    c.distance_cm,
    r.startTime,
    r.stopTime,
    r.testId,
    c.bronze_record_id,
    CASE
      WHEN c.timestamp BETWEEN r.startTime AND r.stopTime THEN 'step'
      ELSE 'no_step'
    END AS step_label
  FROM cleaned_distance c
  LEFT JOIN workspace.bronze.raw_rapid_step_test r
    ON c.deviceId = r.deviceId
   AND c.timestamp BETWEEN r.startTime AND r.stopTime
)

-- 3) Write out to Silver with lineage info
CREATE OR REPLACE TABLE workspace.silver.labeled_step_test AS
SELECT
  timestamp,
  deviceId,
  sensorType,
  distance_cm,
  step_label,
  'device_message_raw + rapid_step_test_raw' AS source,
  startTime,
  stopTime,
  bronze_record_id,
  testId
FROM joined;

# COMMAND ----------
%md
**Instructor tips for this ETL demo**
- Emphasize `BETWEEN startTime AND stopTime` and why time zones matter.
- Show how changing to `INNER JOIN` removes rows without windows.
- Mention that `distance` may include commas or spaces; adjust regex as needed.
- Encourage logging counts before/after cleaning to spot data loss.

# COMMAND ----------
%md
## Section 6 – Querying Metadata from Information Schema

`workspace.information_schema.columns` lists columns, data types, and comments for any table. Use it to verify the Silver table schema and catch typos early.

# COMMAND ----------
%sql
SELECT 
  column_name   AS `Column Name`,
  data_type     AS `Data Type`,
  comment       AS `Comment`
FROM workspace.information_schema.columns
WHERE table_schema = 'silver'
  AND table_name   = 'labeled_step_test'
ORDER BY ordinal_position;

# COMMAND ----------
%md
**Instructor notes**
- Table names are case-insensitive in Hive metastore but case-sensitive in some contexts; stick to lowercase here.
- To adapt, change `table_schema`/`table_name` literals to match the created table.
- If zero rows return: confirm the table exists, check catalog/schema, and ensure the query ran in the correct workspace.

# COMMAND ----------
%md
## Section 7 – Exporting Schema as CSV (Explained)

1. Run the information_schema query above.
2. In the Databricks query result grid, click **… → Download → CSV**.
3. Rename the file to `silver_rapid_step_test_schema.csv` (or similar descriptive name).
4. Verify the CSV includes column names, types, and comments.

**Common issues**
- Downloading from an old result set (stale query). Re-run before download.
- Wrong table/schema selected, leading to empty or incorrect CSV.
- Forgetting to rename the file for submission.

# COMMAND ----------
%md
## Section 8 – Troubleshooting Guide (Instructor Cheatsheet)

**Symptom → Likely cause → Fix**
- *Table not found*: Probably missing catalog/schema (`workspace.silver.labeled_step_test`). Re-run CREATE TABLE with full name.
- *Empty information_schema result*: Wrong `table_schema`/`table_name` or table not created. Double-check spelling and rerun DDL.
- *`distance_cm` all NULL*: Regex did not strip units or cast failed. Inspect raw values and adjust cleaning logic.
- *All labels are `no_step`*: Join/window condition incorrect or start/stop times mismatched. Verify time zones and interval logic.
- *Duplicated rows after join*: Missing distinct test windows or overlapping intervals. Consider using window bounds or selecting the nearest window.

**Quick checks**
- Confirm row counts before/after cleaning.
- Print sample rows where `step_label = 'step'` to ensure labeling works.
- Validate data types with `DESCRIBE EXTENDED workspace.silver.labeled_step_test;` if needed.

# COMMAND ----------
%md
## Section 9 – Discussion & Reflection Prompts (For Class)

- Why is metadata so important before training ML models?
- What could go wrong if we skip the Silver layer and train on raw Bronze data?
- How would you explain the purpose of `step_label` to a non-technical stakeholder?
- If you had to add one more lineage field, what would it be and why?
- How does clean metadata support responsible and transparent model deployment?

Optionally connect to broader reflection: clear, well-labeled data enables trustworthy decisions and fairer models.
