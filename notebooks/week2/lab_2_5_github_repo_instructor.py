# Databricks notebook source
# COMMAND ----------
%md
# 2.5 GitHub Repository Lab — Instructor Training Notebook

*Instructor-only version. Use this as your live demo script.*
* Focus: upload parquet files, run SQL and Python, create a join + plot, and push to GitHub.
* Tone: calm, encouraging, and simple — ideal for students who are new to Databricks, SQL, Python, and Git.

Remember: this notebook is a **show-and-tell**. Pause often, ask questions, and let students try steps after you demo them.

# COMMAND ----------
%md
## Learning Goals for This Lab
- Students can upload `.parquet` files as tables in the correct catalog and schema (`workspace.bronze`).
- Students can write basic SQL to explore and filter data.
- Students can mirror SQL logic using PySpark DataFrames.
- Students can join two tables, compute basic aggregates, and create a simple plot.
- Students can push their notebook to GitHub with clear commit messages.
- Students can reflect briefly on ethics, accuracy, and discipleship in data preparation.

**Instructor notes**
- Hardest parts: connecting the notebook to GitHub, matching SQL logic in Python, and join syntax.
- Best live demos: first table upload, first SQL query, first Python filter, and the GitHub commit/push flow.

# COMMAND ----------
%md
## 3. Setup: Importing the .parquet Files into Databricks

### 3.1 Step-by-step instructions for the live demo
1. Download `device_messages.parquet` and `rapid_step_tests.parquet` to your computer.
2. In Databricks UI, go to **Catalog ➜ Create Table ➜ Create or Modify Table ➜ Upload**.
3. Choose **Catalog = `workspace`** and **Schema = `bronze`**.
4. Name the tables exactly:
   - `device_message_raw`
   - `rapid_step_tests_raw`
5. Click **Preview Table** to confirm rows appear. Say aloud: “We see data, so the upload worked.”

**Talk track suggestion**
- “Watch the catalog and schema names carefully. Small typos cause big errors.”
- “After upload, peek at the preview. Quick checks save time.”
- “I will run a tiny SQL query to double-check. You will do the same.”

### 3.2 SQL demo: verifying imports
Below, run each cell and show students the results.

# COMMAND ----------
%sql
SHOW TABLES IN workspace.bronze;

# COMMAND ----------
%sql
SELECT * FROM workspace.bronze.device_message_raw LIMIT 10;

# COMMAND ----------
%sql
SELECT * FROM workspace.bronze.rapid_step_tests_raw LIMIT 10;

# COMMAND ----------
%md
Ask the class: “What do you notice about these tables?”
- How many columns? What kinds of fields (device_id, timestamp, steps, etc.)?
- Rough row counts? Are there any nulls or odd values?
Encourage students to describe what they see before you explain.

# COMMAND ----------
%md
## 4. SQL Part 1 – Exploring the Raw Tables (1.1–1.7)
This mirrors what students do in Part 1 of the lab. Keep questions simple and invite guesses before running queries.

### Part 1: SQL Demo (1.1–1.7)
- Show basic SELECTs, filters, counts, and a grouped aggregation.
- Remind students: always include the catalog and schema (`workspace.bronze`).
- Common mistake: wrong table or schema name.

# COMMAND ----------
%md
**Question to students:** “Which fields matter for the tests?”
- We will pick specific columns from `rapid_step_tests_raw`.
- Hint: students may forget the schema in the table name.

# COMMAND ----------
%sql
SELECT device_id, test_id, test_type, start_time, end_time, total_steps
FROM workspace.bronze.rapid_step_tests_raw
LIMIT 20;

# COMMAND ----------
%md
**Question:** “How can we focus on one device?”
- We will filter rows by a device ID.
- Ask students to predict the number of rows before running.

# COMMAND ----------
%sql
SELECT device_id, test_id, total_steps
FROM workspace.bronze.rapid_step_tests_raw
WHERE device_id = 'device_001';

# COMMAND ----------
%md
**Question:** “What is the range of steps recorded?”
- Use simple aggregates: COUNT, MIN, MAX, AVG.
- Common mistake: typing `count` as `counts`.

# COMMAND ----------
%sql
SELECT COUNT(*) AS row_count,
       MIN(total_steps) AS min_steps,
       MAX(total_steps) AS max_steps,
       AVG(total_steps) AS avg_steps
FROM workspace.bronze.rapid_step_tests_raw;

# COMMAND ----------
%md
**Question:** “Which device averages the most steps per test?”
- Group by device_id and compute an average.
- Remind: `GROUP BY` device_id must match selected columns.

# COMMAND ----------
%sql
SELECT device_id, AVG(total_steps) AS avg_steps_per_test
FROM workspace.bronze.rapid_step_tests_raw
GROUP BY device_id
ORDER BY avg_steps_per_test DESC
LIMIT 10;

# COMMAND ----------
%md
## 5. Python Part 2 – Mirroring SQL Logic with DataFrames (2.1–2.6)
We now show the same ideas in PySpark DataFrames. Emphasize the SQL-to-Python mapping.

**Mapping reminder**
- SQL `SELECT` + `WHERE` ➜ Python `.select()` + `.filter()`
- SQL `GROUP BY` ➜ Python `.groupBy()`
- SQL `AVG()` ➜ Python `avg()` from `pyspark.sql.functions`
- Use `spark.table("workspace.bronze.table_name")` to load tables.

Common struggles:
- Forgetting to assign the result to a variable.
- Using SQL-only syntax in Python cells.
- Case sensitivity of column names.

# COMMAND ----------
# Python: load the tables as DataFrames
rapid_df = spark.table("workspace.bronze.rapid_step_tests_raw")
device_df = spark.table("workspace.bronze.device_message_raw")
print("Rapid tests rows:", rapid_df.count())
print("Device messages rows:", device_df.count())

# COMMAND ----------
%md
**SQL vs Python side-by-side**
- SQL: `SELECT device_id, test_id FROM workspace.bronze.rapid_step_tests_raw LIMIT 5;`
- Python: `rapid_df.select("device_id", "test_id").limit(5)`

We will now run the Python version and display results.

# COMMAND ----------
# Python: select specific columns
rapid_df.select("device_id", "test_id", "total_steps").limit(5).display()

# COMMAND ----------
%md
**Filtering example**
- SQL: `... WHERE device_id = 'device_001'`
- Python: `.filter(rapid_df.device_id == "device_001")`

# COMMAND ----------
# Python: filter by one device
rapid_df.filter(rapid_df.device_id == "device_001").select("device_id", "test_id", "total_steps").display()

# COMMAND ----------
%md
**Aggregation example**
- SQL: `GROUP BY device_id` + `AVG(total_steps)`
- Python: `.groupBy("device_id").agg(avg("total_steps"))`

# COMMAND ----------
from pyspark.sql.functions import avg

rapid_df.groupBy("device_id").agg(avg("total_steps").alias("avg_steps_per_test")).orderBy("avg_steps_per_test", ascending=False).display()

# COMMAND ----------
%md
## 6. Joining, Feature Creation, and Visual Check (2.5–2.6)
Purpose: combine the rapid step tests with device messages to build simple features.
- Join on `device_id` (and optionally time alignment if available).
- Create a small feature table.
- Show a quick visual filtered to one device.

# COMMAND ----------
from pyspark.sql.functions import col, avg as avg_, count as count_

# Example feature: average reported heart rate per device from device messages
heartrate_features = device_df.groupBy("device_id").agg(avg_("heartrate").alias("avg_heartrate"))

# Join features back to rapid step tests
feature_table = rapid_df.join(heartrate_features, on="device_id", how="left") \
    .select("device_id", "test_id", "total_steps", "avg_heartrate")

# Display the joined feature table
feature_table.display()

# COMMAND ----------
%md
Now make a simple visual for one device. Remind students to click the **Plot/Visualization** tab above the output and pick a line or bar chart.

# COMMAND ----------
# Filter to one device and show data ready for plotting
sample_device = "device_001"
plot_df = feature_table.filter(col("device_id") == sample_device)
plot_df.display()

# COMMAND ----------
%md
Explain the visual in plain language:
- “We are checking if step counts look reasonable for this device.”
- “If a bar looks very high or very low, we might have data entry errors.”
Encourage students to try another device_id on their own.

# COMMAND ----------
%md
## 7. Ethics & Reflection Demo
Accuracy is an act of honesty and discipleship. If we mislabel tests, someone could make a wrong health choice. Being careful with data honors our commitment to truth and to following Christ.

**Reflection prompts for class**
- “What could go wrong if the labels in this dataset are wrong?”
- “How does being honest and careful with data relate to following Christ?”
- “Where in your code could a small error have a big effect?”

Invite students to add 2–3 sentences of reflection in their notebook.

# COMMAND ----------
# Optional demo: printing a reflection
print("Reflection: Accurate data is a form of honesty. I will double-check joins and labels so decisions are trustworthy.")

# COMMAND ----------
%md
## 8. GitHub: Commit & Push from Databricks
This is the part many students find new. Go slowly and narrate each click.

**Steps to demo**
1. Ensure the notebook is attached to a Git repo in Databricks (Version Control panel ➜ Connect).
2. Open the **Git** panel on the right.
3. Write a clear commit message, e.g.,
   - “Imported raw step test data and verified row counts.”
   - “Added SQL queries for device-level aggregates.”
   - “Implemented PySpark join and visual check.”
4. Click **Commit**.
5. Click **Push** to send changes to GitHub.
6. Open GitHub in the browser to verify the notebook and history are visible.

**What to point out on screen**
- The **Commit** button is not the same as **Save**. Saving edits the notebook; committing records history.
- The **Push** step sends commits to GitHub. Without push, instructors cannot grade.
- Show the branch name and remind students to stay on their lab branch if instructed.

**Quick GitHub grading checklist**
- The GitHub link opens.
- At least 2–3 meaningful commits.
- Commit messages are specific, not “update.”

# COMMAND ----------
%md
## 9. Common Student Errors & How to Debug Them
Use these to coach rather than fix for students. Ask questions like “What schema are you using?”

1. **Symptom:** “Table not found.”
   - Likely cause: Wrong catalog/schema (used `hive_metastore` instead of `workspace`).
   - Coach: Ask them to print `spark.catalog.listTables("workspace", "bronze")` or run `SHOW TABLES IN workspace.bronze;`.
2. **Symptom:** “No such table rapid_step_test_raw.”
   - Likely cause: Misspelled table name (`rapid_step_tests_raw` is correct).
   - Coach: Have them copy-paste the exact table name from the catalog.
3. **Symptom:** “mismatched input '%'” in a Python cell.
   - Likely cause: They wrote `%sql` inside a Python cell.
   - Coach: Remind them to put `%sql` only in SQL cells.
4. **Symptom:** `.show()` or `.display()` fails because object is `None`.
   - Likely cause: They did not assign the DataFrame from an earlier step.
   - Coach: Ask, “What does this variable contain? Can you print its schema?”
5. **Symptom:** Join returns too many rows.
   - Likely cause: Missing or wrong join condition.
   - Coach: Ask them to state the join keys aloud, then check the code.
6. **Symptom:** Plot not appearing.
   - Likely cause: They used `print()` instead of `display()`.
   - Coach: Ask them to rerun with `.display()` and click the Visualization tab.
7. **Symptom:** GitHub link missing or private.
   - Likely cause: Repo is private or push not completed.
   - Coach: Have them open GitHub and copy the public link; ensure push succeeded.

# COMMAND ----------
%md
## 10. Instructor Checklist Before Grading
Use this quick scan (30–60 seconds):
- **Data Import:** Both tables exist in `workspace.bronze` with data visible.
- **SQL:** Queries run with clear Markdown notes.
- **Python:** DataFrame logic mirrors SQL; code runs and is readable.
- **Join & Features:** One joined table with at least one meaningful feature.
- **Plot:** Visual uses correct fields and filters to a device.
- **Reflection:** 2–3 sentences on accuracy, ethics, and discipleship.
- **GitHub:** Notebook is in the repo with 2–3 clear commits; link works.

Celebrate small wins. Encourage honesty and careful work as part of discipleship.
