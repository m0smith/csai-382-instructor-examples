# Databricks notebook source
# MAGIC %md
# MAGIC # CSAI-382 Week 1 – STEDI Data in Databricks (Instructor Notebook)
# MAGIC 
# MAGIC **What students will learn today:**
# MAGIC - Use Databricks notebooks (markdown + code) and run cells.
# MAGIC - Explore the STEDI device data using pandas (also similar in PySpark).
# MAGIC - Clean, join, and engineer simple features for model training.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ### Instructor Note (Do not read word-for-word)
# MAGIC - Flow for the live session:
# MAGIC   1. Demo Databricks basics (cells, Shift+Enter, markdown vs. code).
# MAGIC   2. Show sample STEDI data (DeviceMessage + RapidStepTest).
# MAGIC   3. Explore with `head`, `info`, `describe`.
# MAGIC   4. Clean distance and timestamps.
# MAGIC   5. Join sensor rows to a test window.
# MAGIC   6. Engineer simple features.
# MAGIC   7. Ethics discussion + short spiritual thought.
# MAGIC - Speak slowly, pause often, and check for questions (ESL friendly).
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Getting Started in Databricks
# MAGIC - Cells can be **Markdown** (text) or **Code**.
# MAGIC - Click a cell and press **Shift + Enter** to run it.
# MAGIC - Markdown is for explanations. Code is for Python commands.
# MAGIC - Demo: run the code cell below to check the cluster is ready.
# COMMAND
print("Welcome to CSAI-382 Databricks live demo! If this prints, the cluster is ready.")

# COMMAND
# MAGIC %md
# MAGIC ## Sample STEDI Data: DeviceMessage and RapidStepTest
# MAGIC The STEDI device measures balance using an ultrasonic sensor.
# MAGIC 
# MAGIC **Two main tables (small mock versions here):**
# MAGIC - **DeviceMessage** (high-frequency readings)
# MAGIC   - `deviceId`: which device sent the reading
# MAGIC   - `messageOrigin`: where the data came from (e.g., "mobileApp")
# MAGIC   - `sensorType`: type of sensor (e.g., "distance")
# MAGIC   - `distance`: string like "1.2cm" (distance from the ultrasonic sensor)
# MAGIC   - `timestamp_ms`: time in milliseconds since 1970 (Unix epoch)
# MAGIC   - `message`: optional text from the device
# MAGIC - **RapidStepTest** (summary for one test session)
# MAGIC   - `customer`: who did the test
# MAGIC   - `deviceId`: which device was used
# MAGIC   - `startTime_ms`: test start (ms since epoch)
# MAGIC   - `stopTime_ms`: test end (ms since epoch)
# MAGIC   - `testTime_sec`: duration in seconds
# MAGIC   - `totalSteps`: how many steps were detected
# MAGIC   - `stepPoints`: list of times when steps happened (ms)
# MAGIC 
# MAGIC We will use tiny DataFrames so everyone can see the logic. The same ideas work for large tables in Databricks.
# COMMAND
import pandas as pd

# Small DeviceMessage mock data
# Each row is one sensor reading

device_data = [
    {"deviceId": "dev-1", "messageOrigin": "mobileApp", "sensorType": "distance", "distance": "1.0cm", "timestamp_ms": 1_700_000_000_000, "message": "ok"},
    {"deviceId": "dev-1", "messageOrigin": "mobileApp", "sensorType": "distance", "distance": "1.3cm", "timestamp_ms": 1_700_000_000_500, "message": "ok"},
    {"deviceId": "dev-1", "messageOrigin": "mobileApp", "sensorType": "distance", "distance": "0.9cm", "timestamp_ms": 1_700_000_001_000, "message": "ok"},
    {"deviceId": "dev-1", "messageOrigin": "mobileApp", "sensorType": "distance", "distance": "1.5cm", "timestamp_ms": 1_700_000_001_500, "message": "slight sway"},
    {"deviceId": "dev-2", "messageOrigin": "mobileApp", "sensorType": "distance", "distance": "2.0cm", "timestamp_ms": 1_700_000_002_000, "message": "ok"},
]

device_df = pd.DataFrame(device_data)

# Small RapidStepTest mock data
# One row per test session

rapid_step_data = [
    {
        "customer": "Alice",
        "deviceId": "dev-1",
        "startTime_ms": 1_700_000_000_000,
        "stopTime_ms": 1_700_000_002_000,
        "testTime_sec": 20,
        "totalSteps": 4,
        "stepPoints": [1_700_000_000_200, 1_700_000_000_900, 1_700_000_001_400, 1_700_000_001_900],
    },
    {
        "customer": "Ben",
        "deviceId": "dev-2",
        "startTime_ms": 1_700_000_002_000,
        "stopTime_ms": 1_700_000_003_000,
        "testTime_sec": 10,
        "totalSteps": 2,
        "stepPoints": [1_700_000_002_200, 1_700_000_002_800],
    },
]

rapid_step_df = pd.DataFrame(rapid_step_data)

print("DeviceMessage sample:")
print(device_df)
print("\nRapidStepTest sample:")
print(rapid_step_df)

# COMMAND
# MAGIC %md
# MAGIC ## Exploring the Data
# MAGIC - `head()` shows the first few rows.
# MAGIC - `info()` shows column types and null counts.
# MAGIC - `describe()` gives simple statistics (for numeric columns).
# MAGIC - Instructor tip: ask students what they notice about data types and ranges.
# COMMAND
# Look at the first few rows
print("DeviceMessage head:")
print(device_df.head())

print("\nRapidStepTest head:")
print(rapid_step_df.head())

# COMMAND
# Check data types and null counts
print("DeviceMessage info:")
print(device_df.info())

print("\nRapidStepTest info:")
print(rapid_step_df.info())

# COMMAND
# Basic statistics (only numeric columns)
print("DeviceMessage describe:")
print(device_df.describe())

print("\nRapidStepTest describe:")
print(rapid_step_df.describe())

# COMMAND
# MAGIC %md
# MAGIC ## Cleaning the Distance Column
# MAGIC Problem: `distance` is a string like "1.0cm". We need a number for math.
# MAGIC 
# MAGIC Cleaning steps:
# MAGIC 1. Remove the letters `cm`.
# MAGIC 2. Convert the remaining part to a float.
# MAGIC 3. Store it in a new column `distance_cm`.
# MAGIC 
# MAGIC This keeps the original column (for reference) and creates a clean numeric column.
# COMMAND
# Remove the "cm" text and convert to float
# 1) strip the letters
# 2) convert to float
# 3) save as a new column

device_df["distance_cm"] = device_df["distance"].str.replace("cm", "", regex=False).astype(float)
print(device_df[["distance", "distance_cm"]])

# COMMAND
# MAGIC %md
# MAGIC ## Working with Time (Timestamps to Datetime)
# MAGIC - The times are in **milliseconds since 1970** (Unix epoch).
# MAGIC - Humans read datetimes (YYYY-MM-DD HH:MM:SS) more easily.
# MAGIC - We will convert millisecond columns to readable datetime.
# MAGIC 
# MAGIC Note: For big data, PySpark has similar functions, but pandas is fine for this demo.
# COMMAND
# Convert timestamp columns to datetime
# pandas uses nanoseconds by default, so set unit="ms" for milliseconds

device_df["timestamp_dt"] = pd.to_datetime(device_df["timestamp_ms"], unit="ms")
rapid_step_df["startTime_dt"] = pd.to_datetime(rapid_step_df["startTime_ms"], unit="ms")
rapid_step_df["stopTime_dt"] = pd.to_datetime(rapid_step_df["stopTime_ms"], unit="ms")

print(device_df[["timestamp_ms", "timestamp_dt"]])
print("\nRapidStepTest with datetime:")
print(rapid_step_df[["startTime_dt", "stopTime_dt"]])

# COMMAND
# MAGIC %md
# MAGIC ## Linking Sensor Readings to a Test
# MAGIC - One **RapidStepTest** row describes a test window (start to stop).
# MAGIC - Many **DeviceMessage** rows happen during that window.
# MAGIC - We will filter rows where:
# MAGIC   1. `deviceId` matches.
# MAGIC   2. `timestamp_ms` is between `startTime_ms` and `stopTime_ms` (inclusive).
# MAGIC 
# MAGIC For the demo, we will use the first test row (customer = Alice).
# COMMAND
# Pick the first RapidStepTest row (Alice)
alice_test = rapid_step_df.iloc[0]

# Filter DeviceMessage rows for Alice's device and time window
mask_same_device = device_df["deviceId"] == alice_test["deviceId"]
mask_in_window = device_df["timestamp_ms"].between(alice_test["startTime_ms"], alice_test["stopTime_ms"])

alice_device_rows = device_df[mask_same_device & mask_in_window]

print("Alice's test window:")
print(alice_test[["startTime_ms", "stopTime_ms"]])
print("\nDeviceMessage rows for Alice's test:")
print(alice_device_rows[["timestamp_dt", "distance_cm", "message"]])

# COMMAND
# MAGIC %md
# MAGIC ## Simple Feature Engineering
# MAGIC - A **feature** is a numeric description of data that helps a model.
# MAGIC - For balance, distance changes can show sway or stepping patterns.
# MAGIC - Helpful features from distance readings:
# MAGIC   - `averageDistance`: overall sway during the test
# MAGIC   - `minDistance`: closest distance (maybe a strong step)
# MAGIC   - `maxDistance`: farthest distance (maybe leaning away)
# MAGIC   - `varianceDistance`: how much the distance changes (stability)
# MAGIC 
# MAGIC We will calculate these for Alice's test and attach them to the RapidStepTest row.
# COMMAND
# Calculate simple stats for Alice's DeviceMessage rows
average_distance = alice_device_rows["distance_cm"].mean()
min_distance = alice_device_rows["distance_cm"].min()
max_distance = alice_device_rows["distance_cm"].max()
variance_distance = alice_device_rows["distance_cm"].var()  # pandas sample variance

# Create a new DataFrame with one row of features
alice_features = pd.DataFrame([
    {
        "customer": alice_test["customer"],
        "deviceId": alice_test["deviceId"],
        "averageDistance": average_distance,
        "minDistance": min_distance,
        "maxDistance": max_distance,
        "varianceDistance": variance_distance,
    }
])

print("Engineered features for Alice:")
print(alice_features)

# Attach to the RapidStepTest row (left join on customer + deviceId)
rapid_step_with_features = rapid_step_df.merge(alice_features, on=["customer", "deviceId"], how="left")
print("\nRapidStepTest with new feature columns:")
print(rapid_step_with_features)

# COMMAND
# MAGIC %md
# MAGIC ## Visualizing a Single Test
# MAGIC - A line chart can show how distance changes over time.
# MAGIC - **x-axis:** time (datetime)
# MAGIC - **y-axis:** distance in centimeters
# MAGIC - Instructor tip: Ask students where steps might have occurred based on peaks or dips.
# COMMAND
import matplotlib.pyplot as plt

# Plot Alice's distance over time
plt.figure(figsize=(8, 4))
plt.plot(alice_device_rows["timestamp_dt"], alice_device_rows["distance_cm"], marker="o")
plt.title("Alice - Distance over Time (RapidStepTest)")
plt.xlabel("Time")
plt.ylabel("Distance (cm)")
plt.grid(True)
plt.show()

# COMMAND
# MAGIC %md
# MAGIC ## Ethics in AI – STEDI Data
# MAGIC 
# MAGIC This balance data represents real people. We must use it with care:
# MAGIC - The model should not make strong medical claims without clinical evidence.
# MAGIC - Bias risk: if most data comes from one age group or region, the model may be unfair to others.
# MAGIC - Privacy: protect identities; avoid sharing raw timestamps or device IDs outside the team.
# MAGIC - Transparency: explain what the model can and cannot do.
# MAGIC 
# MAGIC **Discussion prompts (ask the class):**
# MAGIC 1. What could go wrong if we assume this model is perfectly accurate?
# MAGIC 2. How could bias in who takes the test affect results?
# MAGIC 3. How can we protect user privacy when storing or sharing this data?
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Spiritual Thought – Balance and Small Measurements
# MAGIC - The STEDI device uses many tiny distance readings to see balance over time.
# MAGIC - Our spiritual balance also comes from small, daily choices.
# MAGIC - Scripture: **Alma 37:6** – “by small and simple things are great things brought to pass.”
# MAGIC - Just as tiny sensor readings show a pattern, small spiritual habits (prayer, kindness, scripture study) show our pattern.
# MAGIC - Invite students to consider: *What is one small spiritual “measurement” you could notice or record this week?*
# MAGIC 
# MAGIC This thought can take 2–3 minutes. Keep the tone gentle and respectful.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Wrap-Up and Next Steps
# MAGIC **Today we:**
# MAGIC - Loaded small STEDI-like datasets into pandas DataFrames.
# MAGIC - Explored the data with `head`, `info`, `describe`.
# MAGIC - Cleaned `distance` strings and converted timestamps.
# MAGIC - Linked sensor rows to test windows using deviceId and time.
# MAGIC - Engineered simple features (average, min, max, variance).
# MAGIC - Plotted distance over time, discussed ethics, and shared a spiritual thought.
# MAGIC 
# MAGIC **Student practice ideas:**
# MAGIC - Change the sample data values and re-run the notebook.
# MAGIC - Add a new feature (e.g., step rate = totalSteps / testTime_sec).
# MAGIC - Try the same steps in PySpark DataFrames.
# MAGIC - Sketch how you might handle missing data or noisy readings.
