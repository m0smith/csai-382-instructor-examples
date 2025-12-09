# Databricks notebook source
# COMMAND ----------
# Title: CSAI-382 Week 1 Instructor Notebook — Data Wrangling with Pandas
# Audience: Instructors preparing to teach students new to Pandas on Databricks Volumes
# Style: Friendly, ESL-aware, richly annotated with teaching notes, tips, and common pitfalls

# COMMAND ----------
"""
# 📚 Section 1 – Introduction & Purpose

Welcome, instructors! This notebook is a **ready-to-teach** walkthrough for **CSAI-382 Week 1: Data Wrangling with Pandas**. It mixes runnable code with instructor-facing notes so you can guide students confidently.

**Learning Goals for Students**
- Understand how to access data stored in Databricks Volumes
- Load CSV files with `pandas.read_csv`
- Inspect datasets with `.head()`, `.info()`, `.shape`, and `.describe()`
- Perform essential cleaning: missing values, inconsistent categories, duplicates, and date parsing
- Merge DataFrames safely and correctly
- Summarize data with `groupby` aggregations
- Explore time-based features from datetime columns
- Create quick visualizations for insights

📖 **Instructor Tip:** Remind students that every step has a purpose: clarity, correctness, and reproducibility.
"""

# COMMAND ----------
"""
# 🎨 Section 2 – Teaching Notes for Instructors

- Keep the pace gentle; ESL learners benefit from shorter sentences and repeated key terms.
- Show the **WHY** before the **HOW**. Students remember reasons more than commands.
- Use frequent checkpoints: "Turn to your neighbor and explain what `.info()` tells us."
- Model mistakes intentionally (bad merge, wrong datetime parsing) so students learn to debug.
- 📢 Common mistakes to watch for:
  - Forgetting correct file paths when using Volumes
  - Ignoring `dtype` warnings on read
  - Performing merges on the wrong key or mismatched column names
  - Dropping rows too early (losing data) instead of imputing
  - Forgetting to convert strings to datetime before using `.dt` accessors
"""

# COMMAND ----------
# 🛠 Section 3 – Load Libraries

import pandas as pd
import matplotlib.pyplot as plt

# Make plots show inside notebook
%matplotlib inline

# COMMAND ----------
"""
# 📂 Section 4 – Loading Data + Explanations

Databricks Volumes store files at paths like `/Volumes/<workspace>/<catalog>/<path>`. We will read two CSV files:

- `menu_items.csv` (menu data)
- `order_details.csv` (individual ordered items)

💡 Teaching Note: Reinforce the path structure. Students often forget the leading `/Volumes/` prefix.
"""

# COMMAND ----------
# Define file paths (adjust if your workspace or catalog differs)
menu_path = "/Volumes/workspace/default/pandas/menu_items.csv"
order_path = "/Volumes/workspace/default/pandas/order_details.csv"

# Read the CSVs
menu_df = pd.read_csv(menu_path)
order_df = pd.read_csv(order_path)

print("Loaded menu_df with shape:", menu_df.shape)
print("Loaded order_df with shape:", order_df.shape)

# COMMAND ----------
"""
# 🧾 Section 5 – Inspecting Data (with teaching notes)

Why inspect? To verify columns, spot weird values, and plan cleaning steps.
"""

# COMMAND ----------
# Look at column names and first rows
menu_df.head()

# COMMAND ----------
order_df.head()

# COMMAND ----------
# Quick info and summary stats
print("Menu info:")
menu_info = menu_df.info()
print("\nOrder info:")
order_info = order_df.info()

menu_shape = menu_df.shape
order_shape = order_df.shape
print("\nShapes -> menu:", menu_shape, "order:", order_shape)

print("\nMenu describe (numeric):")
menu_desc = menu_df.describe()
menu_desc

# COMMAND ----------
"""
📃 Teaching Note: `.info()` reveals missing values and dtypes. `.describe()` shows numeric distribution. Encourage students to read outputs aloud.

🚨 Common Mistake: Students may assume `.head()` guarantees data cleanliness. Remind them it only shows the first few rows.
"""

# COMMAND ----------
"""
# 🛂 Section 6 – Cleaning Data (with examples + mistakes)

Goals:
- Detect missing values
- Decide to fill or drop
- Fix inconsistent categories
- Remove duplicates
- Convert strings to datetime

Use the smallest destructive action possible; preserve data when unsure.
"""

# COMMAND ----------
# Detect missing values
menu_missing = menu_df.isna().sum()
order_missing = order_df.isna().sum()

print("Missing values in menu_df:\n", menu_missing)
print("\nMissing values in order_df:\n", order_missing)

# COMMAND ----------
"""
💡 Teaching Note: Discuss trade-offs of dropping vs. filling. Ask, "What happens to our analysis if we drop 5% of rows?" Also, missingness can be informative.
"""

# COMMAND ----------
# Example: fill missing category with 'Unknown'
menu_df["category"] = menu_df["category"].fillna("Unknown")

# Example: drop rows where price is missing (if critical)
menu_df = menu_df.dropna(subset=["price"])

# Example: inconsistent categories (trim/standardize case)
menu_df["category"] = menu_df["category"].str.strip().str.title()

# COMMAND ----------
"""
🚨 Common Mistake: Students may overwrite original data. Encourage creating `clean_menu_df = menu_df.copy()` before heavy changes. Here we keep changes in place for brevity.
"""

# COMMAND ----------
# Remove duplicate menu items by id
before_dupes = len(menu_df)
menu_df = menu_df.drop_duplicates(subset=["menu_item_id"])
after_dupes = len(menu_df)
print(f"Removed {before_dupes - after_dupes} duplicate menu rows")

# COMMAND ----------
# Convert date and time columns
order_df["order_date"] = pd.to_datetime(order_df["order_date"], errors="coerce")
order_df["order_time"] = pd.to_datetime(order_df["order_time"], errors="coerce").dt.time

# Show rows where conversion failed
bad_dates = order_df[order_df["order_date"].isna()]
bad_times = order_df[pd.isna(order_df["order_time"])]
print("Rows with unparseable dates:", len(bad_dates))
print("Rows with unparseable times:", len(bad_times))

# COMMAND ----------
"""
🚨 Common Mistake: Using `.dt` before converting to datetime leads to `AttributeError`. Demonstrate failure once to make the lesson memorable.
"""

# COMMAND ----------
# BAD example (will raise AttributeError if uncommented)
# order_df["order_date"].dt.day_name()

# GOOD example
order_df["order_day"] = order_df["order_date"].dt.day_name()
order_df["order_hour"] = pd.to_datetime(order_df["order_time"], format="%H:%M:%S", errors="coerce").dt.hour
order_df[["order_date", "order_day", "order_hour"]].head()

# COMMAND ----------
"""
# 🔍 Section 7 – Merging DataFrames

Objective: join menu information onto order details.

Example keys:
- `order_df.item_id` should match `menu_df.menu_item_id`

🚨 Common Mistake: Students accidentally merge on `order_id` or mismatched column names, causing duplicated rows or all-null columns.
"""

# COMMAND ----------
# Correct merge
merged_df = order_df.merge(menu_df, how="left", left_on="item_id", right_on="menu_item_id", validate="many_to_one")
merged_df.head()

# COMMAND ----------
# Bad merge example: wrong key (shows problem)
try:
    bad_merge = order_df.merge(menu_df, left_on="order_id", right_on="menu_item_id", how="left", validate="many_to_one")
except Exception as e:
    print("🚨 Bad merge error:", e)

# COMMAND ----------
"""
💡 Teaching Note: Introduce `validate` to catch logic errors. If students see `ValidationError`, they know their keys are wrong.
"""

# COMMAND ----------
"""
# 🤝 Section 8 – GroupBy and Aggregation Examples

Goal: Summarize orders by category, by hour, and by item.
"""

# COMMAND ----------
# Orders per category
orders_by_category = merged_df.groupby("category").size().reset_index(name="order_count")
orders_by_category

# COMMAND ----------
# Revenue per item
merged_df["revenue"] = merged_df["price"]
revenue_by_item = merged_df.groupby("item_name")["revenue"].sum().reset_index().sort_values(by="revenue", ascending=False)
revenue_by_item.head()

# COMMAND ----------
# Orders per hour
orders_by_hour = merged_df.groupby("order_hour").size().reset_index(name="order_count").sort_values("order_hour")
orders_by_hour.head()

# COMMAND ----------
"""
💡 Checkpoint: Ask students to interpret which category sells the most. How might missing categories affect this table?
"""

# COMMAND ----------
"""
# 📍 Section 9 – Time-Based Analysis (hour, day of week)
"""

# COMMAND ----------
# Orders by day of week
orders_by_day = merged_df.groupby("order_day").size().reset_index(name="order_count").sort_values("order_count", ascending=False)
orders_by_day

# COMMAND ----------
# Combine hour and day for peak-time insight
peak_times = merged_df.groupby(["order_day", "order_hour"]).size().reset_index(name="order_count").sort_values(by="order_count", ascending=False)
peak_times.head(10)

# COMMAND ----------
"""
# 🎭 Section 10 – Visualizations

Simple plots using Pandas/Matplotlib. Keep visuals lightweight for the free tier.
"""

# COMMAND ----------
# Bar chart: Orders by category
orders_by_category.set_index("category")["order_count"].plot(kind="bar", color="skyblue", figsize=(6,4), title="Orders by Category")
plt.ylabel("Orders")
plt.show()

# COMMAND ----------
# Line plot: Orders by hour
orders_by_hour.set_index("order_hour")["order_count"].plot(kind="line", marker="o", figsize=(6,4), title="Orders by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Orders")
plt.xticks(range(0,24,2))
plt.show()

# COMMAND ----------
"""
# 🔐 Section 11 – Ethics in Data (Markdown)

🗡️ **Ethics Callout:** Data cleaning is powerful. Removing or imputing values can change business outcomes. Ask:
- Are we introducing bias by dropping rows?
- Do category fixes reflect the business reality?
- Are we transparent about changes? (Document decisions!)

💡 Teaching Note: Encourage students to write a short "data decisions" log for every cleaning step.
"""

# COMMAND ----------
"""
# 😇 Section 12 – Spiritual Thought (Markdown)

💡 **Gospel Principle Callout:** Clarity in data is like spiritual clarity. Just as we clean noisy data to see truth in patterns, we seek to remove distractions to hear promptings more clearly. "God is not the author of confusion, but of peace" (1 Cor. 14:33). Encourage students to value order and light in both study and life.
"""

# COMMAND ----------
"""
# 📣 Section 13 – Instructor Discussion Questions

- What would you log to justify each cleaning choice? (transparency)
- How do we avoid over-cleaning and losing legitimate outliers?
- How would analysis change if time zones were incorrect?
- Which merge key is safest here, and why?
- Reflect: How does bringing order to data help us serve customers better?

📝 **Checkpoint:** Have students pair up and explain the difference between `groupby().size()` and `groupby().count()`.
"""

# COMMAND ----------
"""
# 📑 Section 14 – Summary

Today we:
- Loaded CSVs from Databricks Volumes
- Inspected data for structure and issues
- Cleaned missing values, categories, duplicates, and dates
- Merged tables safely using keys and `validate`
- Aggregated with `groupby` and explored time-based features
- Created quick visualizations
- Considered ethics and a spiritual analogy about clarity

💡 Encourage students to save their cleaned datasets and document decisions for future reproducibility.
"""
