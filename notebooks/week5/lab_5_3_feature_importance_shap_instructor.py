# Databricks notebook source
# MAGIC %md
# MAGIC # CSAI-382 — Instructor Demo Notebook — Week 5 Lab 5.3 (Feature Importance & SHAP)
# MAGIC 
# MAGIC Welcome! This instructor-facing Databricks notebook guides your **live demo** for Lab 5.3 on feature importance and SHAP explainability. The tone is warm, clear, and ESL-friendly so you can teach with confidence.

# COMMAND ----------
# MAGIC %md
# MAGIC ## SECTION 0 — Instructor Notes
# MAGIC 
# MAGIC **Purpose:** Show students how to load a saved pipeline + model, explore global feature importance, and use SHAP for both global and local explanations. The goal is to **model good thinking, not just code**.
# MAGIC 
# MAGIC **Common misunderstandings:**
# MAGIC - Thinking the feature pipeline is optional. Remind them it keeps transformations consistent.
# MAGIC - Confusing model weights with causation. Feature importance shows association, not proof.
# MAGIC - Mixing up **global** explanations (overall patterns) with **local** explanations (one prediction).
# MAGIC 
# MAGIC **What to prepare beforehand:**
# MAGIC - Ensure the **feature pipeline** file and **trained model** are saved in `/dbfs/FileStore` (see paths below).
# MAGIC - Confirm the transformed train/test sets exist (so the demo focuses on explainability, not preprocessing).
# MAGIC - Test that `shap` is installed in the cluster.
# MAGIC 
# MAGIC **Explainability + ethical AI reminders:**
# MAGIC - Be transparent about what the model can and cannot explain.
# MAGIC - Emphasize humility: explanations help us question and improve models.
# MAGIC - Encourage students to look for bias or odd signals, not just accuracy.
# MAGIC 
# MAGIC **Gospel connection (use in class as desired):**
# MAGIC - *"The light shineth in darkness; and the darkness comprehended it not"* (John 1:5). Just as light reveals what was hidden, explainability sheds light on model decisions.
# MAGIC - *"Let your light so shine before men"* (Matthew 5:16). When we explain our models with integrity, we help others see clearly and act wisely.

# COMMAND ----------
# MAGIC %md
# MAGIC ## SECTION 1 — Load Model, Pipeline, and Data
# MAGIC 
# MAGIC ### 1.1 Why this matters
# MAGIC - **Reproducibility:** Loading the same pipeline + model ensures we can recreate results.
# MAGIC - **Pipeline vs. Model (ESL-friendly):** The **pipeline** cleans and reshapes data. The **model** makes predictions. Think of the pipeline as a **kitchen prep station** and the model as the **chef**.
# MAGIC - The pipeline usually transforms raw features (scaling, encoding). The transformed arrays are ready for the model.

# COMMAND ----------
# Imports and file paths
import joblib
import numpy as np
import pandas as pd

# Load pipeline and model
pipeline = joblib.load("/dbfs/FileStore/stedi_feature_pipeline.pkl")
model = joblib.load("/dbfs/FileStore/stedi_best_model.pkl")

# Load transformed data
X_train = np.load("/dbfs/FileStore/X_train_transformed.npy")
X_test = np.load("/dbfs/FileStore/X_test_transformed.npy")
y_test = pd.read_pickle("/dbfs/FileStore/y_test.pkl")

# Quick sanity check of shapes
X_train.shape, X_test.shape

# COMMAND ----------
# MAGIC %md
# MAGIC ### 1.3 Instructor talking points
# MAGIC - "Notice how we load the **pipeline** and **model** separately. The pipeline builds the right inputs, the model predicts."
# MAGIC - "These arrays are already transformed. That means the pipeline handled scaling/encoding before saving."
# MAGIC - "Shape check: rows = samples, columns = engineered features." 

# COMMAND ----------
# MAGIC %md
# MAGIC ## SECTION 2 — Global Feature Importance
# MAGIC 
# MAGIC ### 2.1 What is global importance?
# MAGIC - Shows which features the model considers most useful **on average** across the dataset.
# MAGIC - Random Forest provides `feature_importances_` directly because each tree tracks how much a feature reduces impurity.
# MAGIC - Real-life analogy: When hiring, you might notice that **experience** and **portfolio quality** usually matter most overall.

# COMMAND ----------
# Extract and display top feature importances
import pandas as pd
import numpy as np

feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
importances = model.feature_importances_

importance_df = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values(by="importance", ascending=False)
    .reset_index(drop=True)
)

# Show top 10
importance_df.head(10)

# COMMAND ----------
# MAGIC %md
# MAGIC ### 2.3 Horizontal bar chart (top 10)

# COMMAND ----------
# Visualize top 10 features
import matplotlib.pyplot as plt

top_n = 10
fig, ax = plt.subplots(figsize=(8, 5))
plot_df = importance_df.head(top_n).iloc[::-1]  # reverse for horizontal plot
ax.barh(plot_df["feature"], plot_df["importance"], color="skyblue")
ax.set_xlabel("Importance (higher = more influence)")
ax.set_title(f"Top {top_n} Global Feature Importances")
plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### 2.4 Instructor talking points
# MAGIC - "Global importance tells us what the model leans on **overall**."
# MAGIC - "Ask an intuition check: *Do these top features make sense for predicting stability? Why or why not?*"
# MAGIC - "Remind students: High importance ≠ causation. It's about influence in this model." 

# COMMAND ----------
# MAGIC %md
# MAGIC ## SECTION 3 — SHAP Global Summary Plot
# MAGIC 
# MAGIC ### 3.1 Why SHAP?
# MAGIC - SHAP values show how each feature **pushes** a prediction up or down compared to the baseline.
# MAGIC - Colors usually show feature value (red = higher value, blue = lower). Position shows impact direction.
# MAGIC - The swarm-like plot is a **global view**: many dots per feature, summarizing the whole dataset.

# COMMAND ----------
# SHAP global summary plot
import shap
shap.initjs()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# The model is binary; shap_values[1] corresponds to the positive class
shap.summary_plot(shap_values[1], X_test, feature_names=pipeline.named_steps["preprocess"].get_feature_names_out())

# COMMAND ----------
# MAGIC %md
# MAGIC ### 3.3 Instructor talking points
# MAGIC - "Each dot = one row from the test set. Position shows influence; color shows feature value."
# MAGIC - "Ask: *What surprises you? Which features push predictions higher?*"
# MAGIC - "Explain slowly: top features matter most overall, but direction depends on the dot's side of zero." 

# COMMAND ----------
# MAGIC %md
# MAGIC ## SECTION 4 — SHAP Local Force Plot
# MAGIC 
# MAGIC ### 4.1 Local vs. global
# MAGIC - **Global** = overall trends. **Local** = why **this** prediction happened.
# MAGIC - SHAP force plots show forces pushing the prediction higher or lower for one sample.

# COMMAND ----------
# SHAP force plot for a single example (first row)
sample_index = 0
sample = X_test[sample_index: sample_index + 1]

# Expect two-class output; use class 1 explanation
force_plot = shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][sample_index],
    sample,
    feature_names=pipeline.named_steps["preprocess"].get_feature_names_out(),
)

# Display in notebooks; in Databricks this renders as an iframe/HTML
force_plot

# COMMAND ----------
# MAGIC %md
# MAGIC ### 4.3 Instructor variations
# MAGIC - Change `sample_index` to compare different cases. Ask students to predict which features will dominate.
# MAGIC - Contrast two rows: one with high predicted risk vs. low risk.
# MAGIC - Invite students to narrate the forces: "Which arrows push toward prediction 1? Which push toward 0?" 

# COMMAND ----------
# MAGIC %md
# MAGIC ## SECTION 5 — Model Interpretation Discussion
# MAGIC - Human intuition vs. model intuition: where do they agree or disagree?
# MAGIC - When to trust: patterns that align with domain knowledge and pass sanity checks.
# MAGIC - When to question: if a rarely relevant feature appears highly influential, or if small data changes flip predictions.
# MAGIC - Bias signals: importance driven by proxy variables (e.g., location when it shouldn't matter).
# MAGIC - Responsible AI: explanations help us audit, improve, and communicate models honestly.

# COMMAND ----------
# MAGIC %md
# MAGIC ## SECTION 6 — Ethical + Gospel Perspective
# MAGIC - **Transparency:** Be clear about how the model makes decisions.
# MAGIC - **Accountability:** Own the outcomes; do not hide behind the algorithm.
# MAGIC - **Humility:** Models are tools, not oracles. We are stewards of the people affected.
# MAGIC - Scripture connections for instructors:
# MAGIC   - John 1:5 — light reveals what was hidden; SHAP brings light to model behavior.
# MAGIC   - Matthew 5:16 — let our "light" of honesty and clarity bless others.
# MAGIC - Short read-aloud: "As we explain our models with openness, we invite trust and protect those we serve. Like light in the dark, clear explanations help everyone see the path forward."

# COMMAND ----------
# MAGIC %md
# MAGIC ## SECTION 7 — Wrap-Up
# MAGIC - Emphasize: consistent pipelines, global feature checks, and SHAP for both **global** and **local** insights.
# MAGIC - Student outcome for Lab 5.3: they should replicate these steps and interpret their own model's explanations.
# MAGIC - Encourage ESL learners: pause often, define terms twice, and invite simple questions. 🌟
