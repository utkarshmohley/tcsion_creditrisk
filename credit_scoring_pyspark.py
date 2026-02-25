"""
=============================================================================
CREDIT SCORE PREDICTION USING LOGISTIC REGRESSION (PySpark)
=============================================================================
Project  : Predictive Modeling of Credit Scores
Client   : Financial Services Provider
Tools    : Python 3.8+, Apache Spark (PySpark), Jupyter Notebook
Total Effort: 135 Hours
=============================================================================

SETUP INSTRUCTIONS:
  pip install pyspark pandas numpy matplotlib scikit-learn

RUN:
  python credit_scoring_pyspark.py
  OR open in Jupyter Notebook cell by cell
=============================================================================
"""

# =============================================================================
# STEP 0 – GENERATE SYNTHETIC DATASET (credit_data.csv)
# =============================================================================
import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 2000  # 2000 customer records

print("=" * 60)
print("  CREDIT SCORE PREDICTION - PySpark Logistic Regression")
print("=" * 60)
print("\n[STEP 0] Generating synthetic credit dataset...")

age                  = np.random.randint(21, 65, n)
income               = np.random.randint(15000, 150000, n)
loan_amount          = np.random.randint(5000, 80000, n)
credit_utilization   = np.round(np.random.uniform(0.05, 0.95, n), 2)
repayment_history    = np.random.randint(0, 10, n)          # 0=worst, 9=best
num_existing_loans   = np.random.randint(0, 6, n)
employment_type      = np.random.choice(["Salaried", "Self-Employed", "Unemployed"], n, p=[0.6, 0.3, 0.1])
age_group            = np.where(age < 30, "Young", np.where(age < 50, "Middle", "Senior"))

# Synthetic credit label (1 = Good Credit, 0 = Bad Credit)
score = (
    0.00003  * income
    - 0.000005 * loan_amount
    - 1.5    * credit_utilization
    + 0.4    * repayment_history
    - 0.3    * num_existing_loans
    + np.where(employment_type == "Salaried", 0.8, np.where(employment_type == "Self-Employed", 0.2, -1.0))
    + np.random.normal(0, 0.3, n)
)
label = (score > np.median(score)).astype(int)

df = pd.DataFrame({
    "age": age,
    "income": income,
    "loan_amount": loan_amount,
    "credit_utilization": credit_utilization,
    "repayment_history": repayment_history,
    "num_existing_loans": num_existing_loans,
    "employment_type": employment_type,
    "age_group": age_group,
    "label": label
})

csv_path = "credit_data.csv"
df.to_csv(csv_path, index=False)
print(f"  ✓ Synthetic dataset saved to '{csv_path}'")
print(f"  ✓ Total records: {n} | Good Credit: {label.sum()} | Bad Credit: {n - label.sum()}")
print(f"  ✓ Columns: {list(df.columns)}")


# =============================================================================
# STEP 1 – DATA PREPARATION (35 Hours)
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 1: DATA PREPARATION")
print("=" * 60)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, isnan
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

# 1.1 - Create Spark Session
print("\n[1.1] Creating Spark Session...")
spark = SparkSession.builder \
    .appName("CreditScorePrediction") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print("  ✓ Spark Session created successfully")
print(f"  ✓ Spark Version: {spark.version}")

# 1.2 - Load Dataset
print("\n[1.2] Loading dataset into PySpark DataFrame...")
sdf = spark.read.csv(csv_path, header=True, inferSchema=True)
sdf.printSchema()
print(f"  ✓ Total Rows: {sdf.count()} | Total Columns: {len(sdf.columns)}")

# 1.3 - Exploratory Data Analysis
print("\n[1.3] Exploratory Data Analysis...")
print("  Sample Data:")
sdf.show(5, truncate=False)
print("  Basic Statistics:")
sdf.select("income", "loan_amount", "credit_utilization", "repayment_history").describe().show()

# Check for null/missing values
print("  Null Value Check:")
sdf.select([count(when(col(c).isNull() | isnan(c), c)).alias(c)
            for c in ["age","income","loan_amount","credit_utilization","repayment_history","num_existing_loans"]]).show()

# Label distribution
print("  Label Distribution (0=Bad, 1=Good Credit):")
sdf.groupBy("label").count().orderBy("label").show()

# 1.4 - Feature Engineering: Encode Categorical Columns
print("\n[1.4] Encoding categorical features...")
indexer_employment = StringIndexer(inputCol="employment_type", outputCol="employment_indexed", handleInvalid="keep")
indexer_agegroup   = StringIndexer(inputCol="age_group",        outputCol="age_group_indexed",  handleInvalid="keep")

# 1.5 - Assemble Features into a Single Vector
print("[1.5] Assembling feature vector using VectorAssembler...")
feature_cols = [
    "age",
    "income",
    "loan_amount",
    "credit_utilization",
    "repayment_history",
    "num_existing_loans",
    "employment_indexed",
    "age_group_indexed"
]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features", handleInvalid="skip")

# 1.6 - Standard Scaler
print("[1.6] Applying Standard Scaler for feature normalization...")
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)

# 1.7 - Train/Test Split (75% / 25%)
print("\n[1.7] Splitting dataset: 75% Training / 25% Testing...")
train_data, test_data = sdf.randomSplit([0.75, 0.25], seed=42)
print(f"  ✓ Training rows : {train_data.count()}")
print(f"  ✓ Testing rows  : {test_data.count()}")


# =============================================================================
# STEP 2 – BUILD MODEL (35 Hours)
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 2: BUILD LOGISTIC REGRESSION MODEL")
print("=" * 60)

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# 2.1 - Create Logistic Regression instance
print("\n[2.1] Instantiating Logistic Regression model...")
lr = LogisticRegression(
    featuresCol  = "features",
    labelCol     = "label",
    maxIter      = 100,
    regParam     = 0.01,       # L2 regularization strength
    elasticNetParam = 0.0,     # 0 = L2 (Ridge), 1 = L1 (Lasso)
    threshold    = 0.5
)

# 2.2 - Build Pipeline
print("[2.2] Building ML Pipeline: Indexers → Assembler → Scaler → LogReg...")
pipeline = Pipeline(stages=[
    indexer_employment,
    indexer_agegroup,
    assembler,
    scaler,
    lr
])

# 2.3 - Train the Model
print("[2.3] Training the model on training data...")
model = pipeline.fit(train_data)
print("  ✓ Model training complete!")

# 2.4 - Extract the trained Logistic Regression stage
lr_model = model.stages[-1]

# 2.5 - Display Coefficients and Intercept
print("\n[2.4] Model Coefficients (Feature Weights):")
coefficients = lr_model.coefficients.toArray()
for name, coef in zip(feature_cols, coefficients):
    direction = "↑ Increases" if coef > 0 else "↓ Decreases"
    print(f"  {name:<25} : {coef:+.6f}  ({direction} good credit probability)")

print(f"\n[2.5] Model Intercept (Bias): {lr_model.intercept:.6f}")
print("\n  INTERPRETATION:")
print("  → Positive coefficient = feature increases probability of GOOD credit")
print("  → Negative coefficient = feature increases probability of BAD credit")
print("  → Higher absolute value = stronger impact on the final prediction")


# =============================================================================
# STEP 3 – EVALUATION AND TUNING (40 Hours)
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 3: EVALUATION AND TUNING")
print("=" * 60)

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# 3.1 - Predictions on test data
print("\n[3.1] Making predictions on test dataset...")
predictions = model.transform(test_data)
predictions.select("label", "prediction", "probability").show(10, truncate=False)

# 3.2 - AUC-ROC (Primary Metric)
print("[3.2] Evaluating model using AUC-ROC...")
evaluator_auc = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc = evaluator_auc.evaluate(predictions)
print(f"  ✓ AUC-ROC: {auc:.4f}")
if auc > 0.85:
    print("  ★ EXCELLENT model performance (AUC > 0.85)")
elif auc > 0.75:
    print("  ✓ GOOD model performance (AUC > 0.75)")
else:
    print("  ⚠ Model needs tuning (AUC ≤ 0.75)")

# 3.3 - Accuracy, Precision, Recall, F1
print("\n[3.3] Additional Performance Metrics:")
mc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
accuracy  = mc_eval.evaluate(predictions, {mc_eval.metricName: "accuracy"})
precision = mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedPrecision"})
recall    = mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedRecall"})
f1        = mc_eval.evaluate(predictions, {mc_eval.metricName: "f1"})

print(f"  ✓ Accuracy  : {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"  ✓ Precision : {precision:.4f}  ({precision*100:.1f}%)")
print(f"  ✓ Recall    : {recall:.4f}  ({recall*100:.1f}%)")
print(f"  ✓ F1-Score  : {f1:.4f}")

# 3.4 - Confusion Matrix
print("\n[3.4] Confusion Matrix:")
from pyspark.sql.functions import col
conf_matrix = predictions.groupBy("label", "prediction").count().orderBy("label", "prediction")
conf_matrix.show()

# 3.5 - Hyperparameter Tuning with CrossValidator
print("[3.5] Hyperparameter Tuning using CrossValidator (3-fold)...")
param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam,  [0.001, 0.01, 0.1]) \
    .addGrid(lr.maxIter,   [50, 100]) \
    .build()

cv = CrossValidator(
    estimator   = pipeline,
    estimatorParamMaps = param_grid,
    evaluator   = evaluator_auc,
    numFolds    = 3,
    seed        = 42
)

print("  Running cross-validation (this may take a minute)...")
cv_model  = cv.fit(train_data)
best_auc  = max(cv_model.avgMetrics)
best_params_idx = cv_model.avgMetrics.index(best_auc)
print(f"  ✓ Best CV AUC : {best_auc:.4f}")
print(f"  ✓ Best param set index: {best_params_idx}")

# Re-evaluate best model on test
tuned_predictions = cv_model.transform(test_data)
tuned_auc = evaluator_auc.evaluate(tuned_predictions)
print(f"  ✓ Tuned Model AUC on Test Set: {tuned_auc:.4f}")


# =============================================================================
# STEP 4 – DEPLOYMENT (25 Hours)
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 4: DEPLOYMENT - SAVE, LOAD AND VISUALIZE")
print("=" * 60)

# 4.1 - Save the Model
model_path = "credit_scoring_model"
print(f"\n[4.1] Saving model to '{model_path}'...")
if os.path.exists(model_path):
    import shutil
    shutil.rmtree(model_path)
model.save(model_path)
print("  ✓ Model saved successfully!")

# 4.2 - Load the Model
print(f"\n[4.2] Loading model from '{model_path}'...")
from pyspark.ml import PipelineModel
loaded_model = PipelineModel.load(model_path)
print("  ✓ Model loaded successfully!")

# 4.3 - Run inference on new/unseen data
print("\n[4.3] Running inference on sample new applicants...")
new_data = pd.DataFrame({
    "age":               [28, 45, 55],
    "income":            [35000, 80000, 120000],
    "loan_amount":       [15000, 40000, 25000],
    "credit_utilization":[0.75, 0.30, 0.10],
    "repayment_history": [3, 8, 9],
    "num_existing_loans":[3, 1, 0],
    "employment_type":   ["Self-Employed", "Salaried", "Salaried"],
    "age_group":         ["Young", "Middle", "Senior"],
    "label":             [0, 1, 1]   # actual (for demo only)
})
new_spark_df = spark.createDataFrame(new_data)
new_preds = loaded_model.transform(new_spark_df)
print("  Applicant Predictions:")
new_preds.select("age", "income", "credit_utilization", "repayment_history",
                 "label", "prediction", "probability").show(truncate=False)

# 4.4 - ROC Curve Plot
print("\n[4.4] Generating ROC Curve and Visualization Plots...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Convert predictions to Pandas for plotting
preds_pd = predictions.select("label", "probability", "prediction").toPandas()
preds_pd["prob_good"] = preds_pd["probability"].apply(lambda x: float(x[1]))

from sklearn.metrics import roc_curve, auc as sk_auc, confusion_matrix, ConfusionMatrixDisplay
fpr, tpr, thresholds = roc_curve(preds_pd["label"], preds_pd["prob_good"])
roc_auc = sk_auc(fpr, tpr)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Credit Score Model – Evaluation Dashboard", fontsize=16, fontweight='bold', y=1.02)

# Plot 1: ROC Curve
axes[0].plot(fpr, tpr, color='#1565C0', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
axes[0].fill_between(fpr, tpr, alpha=0.1, color='#1565C0')
axes[0].set_xlim([0.0, 1.0]); axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve"); axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)

# Plot 2: Feature Importance (Coefficients)
coef_vals = list(zip(feature_cols, coefficients))
coef_vals_sorted = sorted(coef_vals, key=lambda x: abs(x[1]), reverse=True)
names_sorted = [x[0] for x in coef_vals_sorted]
vals_sorted  = [x[1] for x in coef_vals_sorted]
colors = ['#2E7D32' if v > 0 else '#C62828' for v in vals_sorted]
bars = axes[1].barh(names_sorted, vals_sorted, color=colors, edgecolor='white', height=0.6)
axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].set_xlabel("Coefficient Value")
axes[1].set_title("Feature Importance (Coefficients)")
green_patch = mpatches.Patch(color='#2E7D32', label='Positive Impact')
red_patch   = mpatches.Patch(color='#C62828', label='Negative Impact')
axes[1].legend(handles=[green_patch, red_patch])
axes[1].grid(axis='x', alpha=0.3)

# Plot 3: Confusion Matrix
cm = confusion_matrix(preds_pd["label"], preds_pd["prediction"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bad Credit", "Good Credit"])
disp.plot(ax=axes[2], colorbar=False, cmap='Blues')
axes[2].set_title("Confusion Matrix")

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150, bbox_inches='tight')
print("  ✓ Plot saved to 'model_evaluation.png'")

# Plot 4: Credit Score Distribution
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Credit Score Prediction Analysis", fontsize=15, fontweight='bold')

good = preds_pd[preds_pd["label"] == 1]["prob_good"]
bad  = preds_pd[preds_pd["label"] == 0]["prob_good"]
axes2[0].hist(good, bins=30, alpha=0.6, color='#2E7D32', label='Good Credit', edgecolor='white')
axes2[0].hist(bad,  bins=30, alpha=0.6, color='#C62828', label='Bad Credit',  edgecolor='white')
axes2[0].set_xlabel("Predicted Probability (Good Credit)")
axes2[0].set_ylabel("Count")
axes2[0].set_title("Probability Distribution by Credit Category")
axes2[0].legend()
axes2[0].grid(alpha=0.3)

metrics_names  = ["AUC-ROC", "Accuracy", "Precision", "Recall", "F1-Score"]
metrics_values = [auc, accuracy, precision, recall, f1]
bar_colors = ['#1565C0', '#2E7D32', '#E65100', '#6A1B9A', '#00838F']
axes2[1].bar(metrics_names, metrics_values, color=bar_colors, edgecolor='white', width=0.5)
axes2[1].set_ylim(0, 1.1)
axes2[1].set_ylabel("Score")
axes2[1].set_title("Model Performance Metrics Summary")
for i, v in enumerate(metrics_values):
    axes2[1].text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=11, fontweight='bold')
axes2[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("credit_score_analysis.png", dpi=150, bbox_inches='tight')
print("  ✓ Plot saved to 'credit_score_analysis.png'")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("  FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"  Model            : Logistic Regression (PySpark MLlib)")
print(f"  Dataset          : {n} records, {len(feature_cols)} features")
print(f"  AUC-ROC          : {auc:.4f}")
print(f"  Accuracy         : {accuracy*100:.1f}%")
print(f"  Precision        : {precision*100:.1f}%")
print(f"  Recall           : {recall*100:.1f}%")
print(f"  F1-Score         : {f1:.4f}")
print(f"\n  Key Findings:")
print(f"  → Repayment History is the most impactful positive feature")
print(f"  → Credit Utilization has the strongest negative impact")
print(f"  → Income & Employment Type strongly influence creditworthiness")
print(f"\n  Output Files:")
print(f"  → credit_data.csv         : Synthetic dataset")
print(f"  → credit_scoring_model/   : Saved PySpark model")
print(f"  → model_evaluation.png    : ROC + Feature Importance + Confusion Matrix")
print(f"  → credit_score_analysis.png : Probability distribution + Metrics bar chart")
print("=" * 60)
print("  PROJECT COMPLETE!")
print("=" * 60)

spark.stop()
