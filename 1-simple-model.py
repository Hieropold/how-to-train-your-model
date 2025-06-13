# This script demonstrates a complete machine learning workflow using XGBoost for binary classification
# We'll build a model to predict if a user will need hints in the next 3 days based on historical data

# First install required packages:
#  python -m pip install pandas xgboost matplotlib scikit-learn joblib shap

# Import required libraries:
# - pandas: for data manipulation and analysis
# - sklearn: for machine learning utilities (train/test splitting, metrics)
# - xgboost: implementation of gradient boosting for our model
# - matplotlib: for creating visualizations
# - joblib: for saving/loading model files
# - shap: for model interpretability
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import joblib
import shap

# === CONFIGURATION SECTION ===
# Define important constants used throughout the script
CSV_PATH = "training_data.csv"           # Path to our input data
TARGET_COLUMN = "label_3"                # The column we want to predict (whether user needs hints in 3 days)
RANDOM_SEED = 42                         # Fixed random seed for reproducibility
SAVE_METRICS_TO = "run_metrics.json"     # Where to save our model's performance metrics
SHAP_SAMPLE_SIZE = 1000                  # Number of samples to use for SHAP value calculation
EXPERIMENT_NAME = "hint_lover_xgb"       # Name of this experiment for tracking purposes

print("🔍 Loading and preprocessing data...")
# === DATA LOADING AND CLEANING SECTION ===
# Read the CSV file into a pandas DataFrame
df = pd.read_csv(CSV_PATH)
# Convert all column names to lowercase for consistency
df.columns = [col.lower() for col in df.columns]
# Replace any missing values (NaN) with 0
df.fillna(0, inplace=True)

# Define columns that shouldn't be used as features
# These include IDs, timestamps, and target variables for different time periods
NON_FEATURE_COLUMNS = ["event_id", "user_id", "derived_tstamp", 
                      "has_managed_hint_next_1_days", "has_managed_hint_next_3_days", 
                      "has_managed_hint_next_7_days", "label_1", "label_3", "label_7"]
# Create list of feature columns: exclude non-feature columns and text columns (object dtype)
feature_columns = [col for col in df.columns if col not in NON_FEATURE_COLUMNS and df[col].dtype != "object"]

# Split data into features (X) and target variable (y)
X = df[feature_columns]     # Features: all numeric columns except those in NON_FEATURE_COLUMNS
y = df[TARGET_COLUMN]       # Target: whether user needed hints in next 3 days

print(f"✅ Loaded {len(df)} rows with {len(feature_columns)} features")

# === DATA SPLITTING SECTION ===
print("📊 Splitting data into train/validation/test sets...")
# First split: 60% train, 40% temporary set (which will be split again)
# stratify=y ensures the same proportion of examples for each class in the split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=RANDOM_SEED, stratify=y)
# Second split: split the temporary set into validation and test sets (20% each of original data)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp)

# Print the size of each dataset for verification
print(f"📁 Training set: {len(X_train)} rows")
print(f"📁 Validation set: {len(X_val)} rows")
print(f"📁 Test set: {len(X_test)} rows")

# === MODEL TRAINING SECTION ===
print("🚀 Training XGBoost model...")
# Initialize XGBoost classifier with carefully chosen hyperparameters:
# - n_estimators: number of trees to build (200)
# - max_depth: maximum depth of each tree (4) - prevents overfitting
# - learning_rate: how much to correct errors with each tree (0.1)
# - subsample: fraction of samples to use for each tree (0.8) - prevents overfitting
# - colsample_bytree: fraction of features to use for each tree (0.8) - prevents overfitting
model = XGBClassifier(
  n_estimators=200,
  max_depth=4,
  learning_rate=0.1,
  subsample=0.8,
  colsample_bytree=0.8,
  random_state=RANDOM_SEED,
  eval_metric="logloss",        # Use logistic loss as evaluation metric
  use_label_encoder=False,      # Don't encode labels (they're already binary)
  objective="binary:logistic"   # We're doing binary classification
)
# Train the model, using validation set to monitor performance
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=10)
print("✅ Model training complete.")

# === MODEL EVALUATION SECTION ===
print("📈 Evaluating model on validation set...")
# Get predictions: both class predictions (0 or 1) and probability predictions
y_pred = model.predict(X_val)                  # Get class predictions (0 or 1)
y_prob = model.predict_proba(X_val)[:, 1]      # Get probability predictions for class 1

# Calculate various performance metrics:
# ROC AUC: Area under the Receiver Operating Characteristic curve
# Higher values (closer to 1.0) indicate better model performance
roc_auc = roc_auc_score(y_val, y_prob)
# Confusion matrix: shows true positives, false positives, true negatives, false negatives
cm = confusion_matrix(y_val, y_pred).tolist()
# Detailed classification report with precision, recall, F1-score for each class
cr = classification_report(y_val, y_pred, output_dict=True)

# Print all metrics
print("roc_auc", roc_auc)
print("confusion_matrix", cm)
print('classification_report', cr)
print("f1_class_0", cr["0"]["f1-score"])   # F1 score for class 0 (no hints needed)
print("f1_class_1", cr["1"]["f1-score"])   # F1 score for class 1 (hints needed)
print("accuracy", cr["accuracy"])           # Overall accuracy

# === VISUALIZATION SECTION ===
# Create and save confusion matrix visualization
plt.figure()
plt.imshow(confusion_matrix(y_val, y_pred), cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.tight_layout()
plt.savefig("confusion_matrix.png")

# Save the trained model to disk for later use
model_path = "model/hint_lover_model.pkl"
joblib.dump(model, model_path)

# Create and save feature importance distribution plot
# This shows how important each feature is for making predictions
plt.figure()
plt.hist(model.feature_importances_, bins=50)
plt.axvline(0.002, color='green', linestyle='--', label="0.002")  # Reference lines for importance thresholds
plt.axvline(0.005, color='red', linestyle='--', label="0.005")
plt.title("Feature Importance Distribution")
plt.xlabel("Importance Score")
plt.ylabel("Feature Count")
plt.legend()
plt.tight_layout()
plt.savefig("reports/feature_importance.png")

# Create and save ROC curve
# ROC curve shows the tradeoff between true positive rate and false positive rate
fpr, tpr, _ = roc_curve(y_val, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--')   # Diagonal line representing random performance
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.grid()
plt.tight_layout()
plt.savefig("reports/roc_curve.png")

# Create and save Precision-Recall curve
# Shows the tradeoff between precision (accuracy of positive predictions)
# and recall (ability to find all positive cases)
precision, recall, _ = precision_recall_curve(y_val, y_prob)
ap_score = average_precision_score(y_val, y_prob)   # Calculate average precision
plt.figure()
plt.plot(recall, precision, label=f"AP = {ap_score:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.tight_layout()
plt.savefig("reports/pr_curve.png")

# === MODEL INTERPRETABILITY SECTION ===
# Calculate SHAP (SHapley Additive exPlanations) values
# SHAP values help us understand how each feature contributes to predictions
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Create SHAP summary plot showing the most important features
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=30, show=False)

# Find and store the 30 most important features based on SHAP values
mean_abs_shap_values = shap_values.abs.mean(axis=0)
important_features = mean_abs_shap_values.nlargest(30)