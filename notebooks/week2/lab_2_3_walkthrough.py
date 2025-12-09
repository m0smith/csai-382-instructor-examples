# Databricks notebook source
# MAGIC %md
# MAGIC # CSAI 382 – Lab 2.3 Walkthrough: Reproducibility & Logging Basics
# MAGIC 
# MAGIC This teacher demo notebook shows how to build a reproducible ETL pipeline with logging. It is meant for instructors to demonstrate Assignment 2.3 in class. Use these cells in Databricks or Jupyter and explain each step to students.
# MAGIC 
# MAGIC > "Let all things be done decently and in order." — 1 Corinthians 14:40
# MAGIC >
# MAGIC > We keep code organized, reproducible, and trustworthy for our teammates.
# MAGIC 
# COMMAND ----------
# MAGIC %md
# MAGIC ## Section 1 – Setup and Project Structure (Part A)
# MAGIC 
# MAGIC ```text
# MAGIC csai382_lab_2_4_example/
# MAGIC ├── data/
# MAGIC │   ├── menu_items.csv
# MAGIC │   └── order_details.csv
# MAGIC ├── logs/
# MAGIC ├── README.md
# MAGIC ├── RUN.md
# MAGIC ├── config.yaml        # optional
# MAGIC └── lab_2_4_repro_logging.ipynb
# MAGIC ```
# MAGIC 
# MAGIC * `data/` holds the input CSV files for the lab.
# MAGIC * `logs/` stores run logs so we can review history.
# MAGIC * `README.md` explains the project at a high level.
# MAGIC * `RUN.md` tells classmates how to run the notebook.
# MAGIC * `config.yaml` is optional for settings.
# MAGIC * `lab_2_4_repro_logging.ipynb` is the demo notebook.
# MAGIC 
# MAGIC > **Instructor Tip:** A clear structure makes grading and teamwork easier. Everyone knows where files live.
# MAGIC 
# COMMAND ----------
# MAGIC %md
# MAGIC ## Section 2 – Logging Basics (Part B)
# MAGIC 
# MAGIC Logging is how our code writes a running journal. It records what happened and when.
# MAGIC 
# MAGIC `print()` shows quick messages, but `logging` is better for real tracking.
# MAGIC 
# MAGIC | Tool    | Good For                   |
# MAGIC | ------- | -------------------------- |
# MAGIC | print   | Quick checks while coding  |
# MAGIC | logging | Serious tracking over time |
# MAGIC 
# MAGIC > **Instructor Tip:** Run the logging cells twice to show how the log file grows with each run.
# COMMAND ----------
# Logging setup with console + file handlers
import logging
import os
from datetime import datetime
import platform

# Ensure logs/ exists
os.makedirs("logs", exist_ok=True)

# Build timestamped log filename
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
log_path = os.path.join("logs", f"run_{run_timestamp}.log")

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Remove existing handlers to avoid duplicate logs in notebook reruns
logger.handlers.clear()

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.INFO)

# Shared formatter
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Log run metadata
logging.info("Starting ETL demo run")
logging.info("Cluster/runtime info: Python %s on %s", platform.python_version(), platform.system())
logging.info("Config: %s", {"env": "demo", "author": "Instructor"})
logging.info("Logs saved to %s", log_path)

# COMMAND ----------
# Example log messages
logging.info("Loading data files...")
logging.warning("Example warning message")
logging.error("Example error message (demo only)")

# COMMAND ----------
# MAGIC %md
# MAGIC Look in the `logs/run_YYYYMMDD_HHMM.log` file. Point out the timestamp, level (INFO/WARNING/ERROR), and message.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Section 3 – Reproducibility Setup (Part C)
# MAGIC 
# MAGIC Reproducibility means: same code + same data + same environment → same results. We set random seeds so shuffling or sampling gives the same output each time. We can also capture our Python packages with `%pip freeze > requirements.txt`.
# COMMAND ----------
# Fix random seeds for reproducibility
import random
import numpy as np

os.environ['PYTHONHASHSEED'] = '0'
random.seed(0)
np.random.seed(0)

logging.info("Random seeds fixed for reproducibility")

# COMMAND ----------
# In Databricks / Jupyter, instructor can run this to capture packages:
# %pip freeze > requirements.txt

# MAGIC %md
# MAGIC `requirements.txt` lists the exact packages and versions. Example:
# MAGIC 
# MAGIC ```
# MAGIC pandas==2.1.0
# MAGIC numpy==1.24.4
# MAGIC pyyaml==6.0.1
# MAGIC ```
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Section 4 – File Hashes for Input Data (Part C)
# MAGIC 
# MAGIC A **SHA-256 hash** is like a fingerprint for a file. If the file changes, the hash changes. We hash `menu_items.csv` and `order_details.csv` to prove we all used the same data.
# COMMAND ----------
import hashlib
import json

# Helper to compute SHA-256 hash of a file
def compute_file_hash(path: str) -> str:
    with open(path, "rb") as f:
        file_bytes = f.read()
    return hashlib.sha256(file_bytes).hexdigest()

files_to_hash = ["data/menu_items.csv", "data/order_details.csv"]

# Compute hashes
data_hashes = {path: compute_file_hash(path) for path in files_to_hash}

# Save hashes to JSON
with open("data_hashes.json", "w") as f:
    json.dump(data_hashes, f, indent=2)

logging.info("Computed data hashes and saved to data_hashes.json")

# COMMAND ----------
# MAGIC %md
# MAGIC Open `data_hashes.json` to show the fingerprints. Explain that using the same data keeps comparisons fair when discussing model results.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Section 5 – ETL with Pandas (Part D)
# MAGIC 
# MAGIC ### 5.1 Load Data
# MAGIC 
# MAGIC We have two tables:
# MAGIC 
# MAGIC * `menu_items`: menu_item_id, item_name, category, price.
# MAGIC * `order_details`: order_id, order_date, order_time, item_id.
# MAGIC 
# MAGIC Join key:
# MAGIC 
# MAGIC ```text
# MAGIC menu_items.menu_item_id  <---->  order_details.item_id
# MAGIC ```
# COMMAND ----------
import pandas as pd

# Load CSVs
menu_df = pd.read_csv("data/menu_items.csv")
orders_df = pd.read_csv("data/order_details.csv")

logging.info("Loaded menu_items.csv with shape %s", menu_df.shape)
logging.info("Loaded order_details.csv with shape %s", orders_df.shape)

menu_df.head(), orders_df.head()

# COMMAND ----------
# MAGIC %md
# MAGIC ### 5.2 Basic Cleaning
# MAGIC 
# MAGIC * Convert `order_date` and `order_time` into a single datetime.
# MAGIC * Strip extra spaces from text fields.
# MAGIC * Make sure `price` is numeric.
# COMMAND ----------
# Create a proper datetime column
orders_df['order_datetime'] = pd.to_datetime(
    orders_df['order_date'] + ' ' + orders_df['order_time']
)
logging.info("Created order_datetime column")

# Clean text fields
menu_df['item_name'] = menu_df['item_name'].str.strip()
menu_df['category'] = menu_df['category'].str.strip()
logging.info("Stripped whitespace from item_name and category")

# Ensure price is float
menu_df['price'] = menu_df['price'].astype(float)
logging.info("Converted price to float")

# COMMAND ----------
# MAGIC %md
# MAGIC ### 5.3 Join the Tables
# MAGIC 
# MAGIC We join to attach item names, categories, and prices to each order line. Use an inner join to keep matching rows only.
# COMMAND ----------
orders_before = orders_df.shape
orders_enriched = pd.merge(
    orders_df,
    menu_df,
    left_on='item_id',
    right_on='menu_item_id',
    how='inner'
)
orders_after = orders_enriched.shape

logging.info("Orders before merge: %s", orders_before)
logging.info("Orders after merge: %s", orders_after)

# COMMAND ----------
# MAGIC %md
# MAGIC ### 5.4 Tidy Table and Metrics
# MAGIC 
# MAGIC Tidy table columns:
# MAGIC * order_id
# MAGIC * order_datetime
# MAGIC * item_name
# MAGIC * category
# MAGIC * price
# MAGIC 
# MAGIC Metrics to compute:
# MAGIC * Top 5 items by quantity
# MAGIC * Revenue by category
# MAGIC * Busiest hour of day
# COMMAND ----------
# Keep only the columns we need
tidy_df = orders_enriched[['order_id', 'order_datetime', 'item_name', 'category', 'price']].copy()
logging.info("Tidy table shape: %s", tidy_df.shape)

tidy_df.head()

# COMMAND ----------
# Metrics
# Top 5 items by order count
top_items = tidy_df.groupby("item_name")['order_id'].count().sort_values(ascending=False).head(5)

# Revenue by category
revenue_by_category = tidy_df.groupby("category")['price'].sum().sort_values(ascending=False)

# Busiest hour of day
busiest_hour = tidy_df.groupby(tidy_df['order_datetime'].dt.hour)['order_id'].count().sort_values(ascending=False)

logging.info("Computed metrics: top_items, revenue_by_category, busiest_hour")

print("Top 5 Items:\n", top_items)
print("\nRevenue by Category:\n", revenue_by_category)
print("\nBusiest Hour of Day (24h):\n", busiest_hour)

# COMMAND ----------
# MAGIC %md
# MAGIC Interpret the metrics in plain language. Example: “These top 5 items are ordered most often. We might highlight them on the menu.”
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Section 6 – Saving Outputs with Timestamps (Part D)
# MAGIC 
# MAGIC Save outputs with timestamps to avoid overwriting. Example path: `etl_output/metrics_YYYYMMDD_HHMM.csv`.
# COMMAND ----------
# Save metrics with timestamp
os.makedirs("etl_output", exist_ok=True)

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")

# Turn top_items into a DataFrame
top_items_df = top_items.reset_index()
top_items_df.columns = ['item_name', 'order_count']

metrics_path = f"etl_output/metrics_{timestamp_str}.csv"
top_items_df.to_csv(metrics_path, index=False)

logging.info("Saved metrics to %s", metrics_path)

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC 
# MAGIC ## Section 7 – Simple Tests with Asserts (Part D)
# MAGIC 
# MAGIC `assert` checks conditions. If the condition is false, Python stops and shows an error. Failing tests are good because they catch problems early.
# MAGIC 
# MAGIC > **Instructor Tip:** Break a test on purpose to show the error.
# COMMAND ----------
# Simple data quality checks
expected_columns = {"order_id", "order_datetime", "item_name", "category", "price"}
assert expected_columns.issubset(set(tidy_df.columns)), "Missing expected columns in tidy_df"

assert len(tidy_df) > 0, "tidy_df is empty – did the merge fail?"

assert (tidy_df["price"] > 0).all(), "Found non-positive prices in tidy_df"

logging.info("All simple assert tests passed")
