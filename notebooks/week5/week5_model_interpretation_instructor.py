# Databricks notebook source
# MAGIC %md
# MAGIC # CSAI-382 – Model Interpretation Live Demo (Instructor)
# MAGIC 
# MAGIC This notebook is for **Week 5–6: Model Interpretation, Explainability, and Fairness**.
# MAGIC 
# MAGIC Students will see:
# MAGIC - How to train and evaluate a **binary classifier**.
# MAGIC - Core metrics: **accuracy, precision, recall, F1, confusion matrix**.
# MAGIC - **Model comparison**: Logistic Regression vs Random Forest.
# MAGIC - **Explainability** with **feature importance** and **SHAP**.
# MAGIC - **Fairness and ethics** reminders with a short gospel connection.
# MAGIC 
# MAGIC > **Teacher Note:** This live demo can take **30–40 minutes** with discussion. Move faster or slower based on class questions.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Setup & Imports
# MAGIC 
# MAGIC We load common libraries for data, modeling, and plots.

# COMMAND ----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import shap

# COMMAND ----------
# MAGIC %md
# MAGIC **Library overview (simple phrases):**
# MAGIC - **numpy / pandas:** handle numbers and tables.
# MAGIC - **matplotlib:** draw plots.
# MAGIC - **scikit-learn:** make datasets, split data, train models, and measure metrics.
# MAGIC - **shap:** explain why a model makes a prediction.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Create and Explore a Synthetic Dataset
# MAGIC 
# MAGIC We will use a **synthetic dataset**.
# MAGIC 
# MAGIC - Each row is a “test result” for a person.
# MAGIC - The target label is **`high_risk`** (1) or **`low_risk`** (0).
# MAGIC - Features represent simple numeric measurements, for example:
# MAGIC   - reaction time
# MAGIC   - step speed
# MAGIC   - balance variation
# MAGIC   - muscle stability
# MAGIC 
# MAGIC This is **not** real medical data. It is only for learning.

# COMMAND ----------
# Create synthetic classification data
n_samples = 1000
random_state = 42

X, y = make_classification(
    n_samples=n_samples,
    n_features=9,
    n_informative=6,
    n_redundant=1,
    n_repeated=0,
    n_classes=2,
    class_sep=1.2,
    flip_y=0.02,
    random_state=random_state
)

feature_names = [
    "reaction_time",
    "step_speed",
    "balance_variation",
    "muscle_stability",
    "ankle_range",
    "vision_score",
    "hearing_score",
    "fatigue_level",
    "coordination"
]

df = pd.DataFrame(X, columns=feature_names)
df["high_risk"] = y

print(f"Dataset shape: {df.shape}")
df.head()

# COMMAND ----------
# Quick numeric summary
summary = df.describe().T
summary

# COMMAND ----------
# MAGIC %md
# MAGIC **Talking points:**
# MAGIC - Each **row** = one person’s test results.
# MAGIC - Each **column** (except `high_risk`) = a numeric feature.
# MAGIC - **Label** (`high_risk`): 1 = higher fall risk, 0 = lower risk.
# MAGIC - Synthetic data now, but the steps will be similar for future STEDI-style data.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Train a Baseline Model (Logistic Regression)
# MAGIC 
# MAGIC - Logistic Regression is a **simple, interpretable** model.
# MAGIC - It predicts the **probability** of the positive class.

# COMMAND ----------
# Split data
X = df[feature_names]
y = df["high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=random_state, stratify=y
)

# Train Logistic Regression
log_reg_model = LogisticRegression(max_iter=1000, solver="lbfgs")
log_reg_model.fit(X_train, y_train)

# Evaluate
log_reg_predictions = log_reg_model.predict(X_test)
log_reg_report = classification_report(y_test, log_reg_predictions)

print("Logistic Regression Classification Report:\n")
print(log_reg_report)

# COMMAND ----------
# Confusion matrix for Logistic Regression
log_reg_cm = confusion_matrix(y_test, log_reg_predictions)
print("Logistic Regression Confusion Matrix:\n", log_reg_cm)

# COMMAND ----------
# MAGIC %md
# MAGIC **Metric meanings (plain language):**
# MAGIC - **Accuracy:** overall percent correct.
# MAGIC - **Precision (class 1):** of the people we predicted **high risk**, how many were really high risk?
# MAGIC - **Recall (class 1):** of all true **high-risk** people, how many did we find?
# MAGIC - **F1:** balance between precision and recall.
# MAGIC - **Confusion matrix:** shows counts for true/false positives and negatives.
# MAGIC 
# MAGIC > **Teacher Note:** Ask: “Which is more serious here, a **false positive** or a **false negative**? If recall is low, what does that mean for high-risk people?”

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Train a Second Model (Random Forest) and Compare
# MAGIC 
# MAGIC - Random Forest = many decision trees **working together**.
# MAGIC - Often more powerful, sometimes less interpretable.

# COMMAND ----------
# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=random_state,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Evaluate
rf_predictions = rf_model.predict(X_test)
rf_report = classification_report(y_test, rf_predictions)

print("Random Forest Classification Report:\n")
print(rf_report)

# Confusion matrix
rf_cm = confusion_matrix(y_test, rf_predictions)
print("Random Forest Confusion Matrix:\n", rf_cm)

# COMMAND ----------
# Compare accuracy side by side
log_reg_accuracy = accuracy_score(y_test, log_reg_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)

models = ["Logistic Regression", "Random Forest"]
accuracies = [log_reg_accuracy, rf_accuracy]

plt.figure(figsize=(6, 4))
plt.bar(models, accuracies, color=["skyblue", "seagreen"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
for idx, acc in enumerate(accuracies):
    plt.text(idx, acc + 0.02, f"{acc:.3f}", ha="center")
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC **Quick comparison:**
# MAGIC - Logistic Regression: more **interpretable** (we can see coefficients), may be a bit less flexible.
# MAGIC - Random Forest: often **higher performance**, but harder to explain.
# MAGIC 
# MAGIC Sentence starters:
# MAGIC - “Although Logistic Regression is simpler, Random Forest performed better because it can capture **nonlinear patterns**.”
# MAGIC - “If we need to explain every decision, we may still choose Logistic Regression.”

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Confusion Matrix Visualization + Interpretation
# MAGIC 
# MAGIC Visualizing helps students see true/false positives and negatives.

# COMMAND ----------
import seaborn as sns

plt.figure(figsize=(5, 4))
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Random Forest Confusion Matrix")
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC **How to read the matrix:**
# MAGIC - Top-left: predicted **low risk** and true label is **low risk** (true negative).
# MAGIC - Top-right: predicted **high risk** but true label is **low risk** (false positive).
# MAGIC - Bottom-left: predicted **low risk** but true label is **high risk** (false negative).
# MAGIC - Bottom-right: predicted **high risk** and true label is **high risk** (true positive).
# MAGIC 
# MAGIC > **Teacher Note:** Ask students to pick one cell and describe what it means for a person in real life.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Feature Importance and SHAP (Explainability)
# MAGIC 
# MAGIC Why explain? We want to know **why** the model predicts something.
# MAGIC - **Feature importance (global):** which features the model uses the most across many cases.
# MAGIC - **SHAP values (local and global):** show how each feature **pushes** a prediction up or down.

# COMMAND ----------
# Feature importance from Random Forest
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(7, 5))
plt.bar(range(len(importances)), importances[indices], color="teal")
plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=45, ha="right")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC **Discussion prompt:**
# MAGIC - Which features are most important?
# MAGIC - Does that match human intuition about fall risk?

# COMMAND ----------
# MAGIC %md
# MAGIC ### SHAP for deeper explanations
# MAGIC 
# MAGIC - We sample a small set to keep it fast in class.
# MAGIC - Each SHAP value shows how much a feature **pushes** the prediction toward high risk or low risk.

# COMMAND ----------
# Use a smaller background dataset for speed
background_sample = X_train.sample(n=200, random_state=random_state)
explainer = shap.Explainer(rf_model, background_sample)

sample_for_shap = X_test.sample(n=200, random_state=random_state)
shap_values = explainer(sample_for_shap)

# COMMAND ----------
# SHAP summary plot (global view)
shap.summary_plot(shap_values, sample_for_shap, show=False)
plt.title("SHAP Summary Plot (Random Forest)")
plt.show()

# COMMAND ----------
# SHAP force plot for one example (local view)
# Databricks can render matplotlib output; for force plot we convert to matplotlib with shap.plots.force + matplotlib=True
single_example = sample_for_shap.iloc[0]
shap.plots.force(explainer(single_example), matplotlib=True)
plt.title("SHAP Force Plot for One Person")
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC **SHAP in simple words:**
# MAGIC - A **positive** SHAP value pushes the prediction toward **high risk (1)**.
# MAGIC - A **negative** SHAP value pushes toward **low risk (0)**.
# MAGIC - Bigger absolute values = stronger influence.
# MAGIC 
# MAGIC > **Teacher Note:** Remind students: SHAP shows **model patterns**, not **causal proof**. Encourage healthy skepticism.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Fairness, Limitations, and Human Responsibility
# MAGIC 
# MAGIC - Models can still be **wrong**. Data may be **biased** or missing key information.
# MAGIC - **False negatives:** miss high-risk people → possible harm.
# MAGIC - **False positives:** worry or cost for people who are actually low risk.
# MAGIC - Keep **humans in the loop** for high-stakes choices.
# MAGIC - Check performance for **different groups** (age, gender, etc.) when data is available.
# MAGIC 
# MAGIC > **Teacher Note:** Ask: “Where should humans double-check the model’s decisions? What data might be missing that could make this unfair?”

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Ethics + Gospel Principle Reflection (Short)
# MAGIC 
# MAGIC - We value **honesty, transparency, and accountability** in AI work.
# MAGIC - Doctrine and Covenants 93:24 teaches that **truth** is knowledge of things as they are, were, and are to come.
# MAGIC - Be honest about model limits and risks. Do not exaggerate performance.
# MAGIC - As disciples of Jesus Christ, use AI tools to **protect and bless others**, not to harm or mislead.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Wrap-Up
# MAGIC 
# MAGIC Today we:
# MAGIC - Trained **two models** (Logistic Regression and Random Forest).
# MAGIC - Compared **metrics** and viewed a **confusion matrix**.
# MAGIC - Explored **feature importance** and **SHAP** for explanations.
# MAGIC - Reflected on **fairness, ethics, and stewardship**.
# MAGIC 
# MAGIC **Connection to coursework:**
# MAGIC - Supports **Lab 5.4: Model Interpretation Report** (students explain their own models).
# MAGIC - Prepares for Week 6–7 dashboards and data stories.
# MAGIC 
# MAGIC > **Teacher Note:** Suggested follow-up: Ask students to pick one feature and argue how it could be measured better in real life. Invite them to write a short reflection on how to communicate model limits to a non-technical audience.
