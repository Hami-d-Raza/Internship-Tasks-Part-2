# ====================================================
# Term Deposit Subscription Prediction (Bank Marketing)
# Cleaned + Robust version of your script
# ====================================================

# Step 1: Import Libraries
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score
import shap
from IPython.display import display  # for notebook display of shap.force_plot

# -------------------------
# Step 2: Load Dataset (robust)
# -------------------------
# Preferred path you had (fix escaping) - but script will try auto-detect as fallback
preferred_path = r"DS_Tasks 2\bank_Dataset.csv"   # change if you know the exact path
preferred_path_alt = r"DS_Tasks 2\bank_Dataset.csv.csv"

def find_csv_in_cwd():
    # return list of csv files in current working dir (non-recursive)
    return glob.glob(os.path.join(os.getcwd(), "*.csv"))

def try_load(path):
    try:
        # Attempt to read with semicolon delimiter (UCI bank dataset uses ';')
        df_try = pd.read_csv(path, sep=';')
        print(f"Loaded: {path}")
        return df_try
    except Exception as e:
        # print short error for debugging and return None
        print(f"Could not load '{path}': {e}")
        return None

# Try preferred paths first
df = None
for p in (preferred_path, preferred_path_alt):
    if os.path.exists(p):
        df = try_load(p)
        if df is not None:
            break

# If not found, scan current dir for likely candidates
if df is None:
    csvs = find_csv_in_cwd()
    print("CSV files in current directory:", csvs)
    # Look for files containing 'bank' or 'bank-additional' in the name
    candidates = [c for c in csvs if ('bank' in os.path.basename(c).lower())]
    if len(candidates) == 1:
        df = try_load(candidates[0])
    elif len(candidates) > 1:
        print("Multiple candidate bank CSVs found; using the first one:", candidates[0])
        df = try_load(candidates[0])
    else:
        # Try all csvs until one loads with ';' sep
        for c in csvs:
            df = try_load(c)
            if df is not None:
                break

if df is None:
    raise FileNotFoundError(
        "No suitable CSV loaded. Put the dataset (bank-additional-full.csv or bank_Dataset.csv) "
        "in the current folder or set `preferred_path` to the exact file path."
    )

print("\nDataset shape:", df.shape)
print(df.head(3))

# -------------------------
# Step 3: Basic sanity + target checks
# -------------------------
if 'y' not in df.columns:
    raise KeyError("Target column 'y' not found in the dataset.")

print("\nOriginal target distribution:\n", df['y'].value_counts())

# Map target to binary (safe even if already binary)
df['y'] = df['y'].map({'yes': 1, 'no': 0}).astype(int)

# -------------------------
# Step 4: Encode Features
# -------------------------
# Find categorical columns robustly
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
# Remove 'y' if present (shouldn't be string anymore, but safe)
if 'y' in categorical_cols:
    categorical_cols.remove('y')

print("\nCategorical columns found:", categorical_cols)

# One-hot encode categorical variables (drop_first to avoid dummy trap)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nData after encoding shape:", df_encoded.shape)

# -------------------------
# Step 5: Train-Test Split
# -------------------------
X = df_encoded.drop('y', axis=1)
y = df_encoded['y']

# Optional: check for NaNs
if X.isnull().any().any():
    print("Warning: NaNs found in features - filling with 0 (you might want smarter imputation).")
    X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)
print("Train positive ratio:", y_train.mean(), " Test positive ratio:", y_test.mean())

# -------------------------
# Step 6: Train Models
# -------------------------
# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, solver='liblinear')
log_reg.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# -------------------------
# Step 7: Evaluation Function
# -------------------------
def evaluate_model(model, X_test, y_test, name):
    # If predict_proba not available, use decision_function (rare)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # fallback: scale decision function to [0,1]
        from sklearn.preprocessing import MinMaxScaler
        scores = model.decision_function(X_test)
        y_proba = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).ravel()
    y_pred = (y_proba >= 0.5).astype(int)  # probability threshold 0.5

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{name} Confusion Matrix:\n{cm}")
    print(f"\n{name} Classification Report:\n{classification_report(y_test, y_pred)}")
    f1 = f1_score(y_test, y_pred)
    print(f"{name} F1-Score: {f1:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

# -------------------------
# Step 8: Evaluate Models & Plot ROC
# -------------------------
plt.figure(figsize=(7,7))
evaluate_model(log_reg, X_test, y_test, "Logistic Regression")
evaluate_model(rf, X_test, y_test, "Random Forest")

plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

# -------------------------
# Step 9: Explain Predictions with SHAP (Random Forest)
# -------------------------
# Note: This works best in a Jupyter/Colab environment
explainer = shap.TreeExplainer(rf)
# compute shap values for the positive class (rf is tree-based)
shap_values = explainer.shap_values(X_test)

# shap_values might be a list (one per output class); for binary classifiers shap_values[1] is positive class
pos_index = 1 if isinstance(shap_values, list) and len(shap_values) > 1 else 0

print("\nExplaining first 5 test rows with SHAP:")
shap.initjs()
for i in range(5):
    # display returns the interactive force-plot in notebooks
    display(shap.force_plot(explainer.expected_value[pos_index], shap_values[pos_index][i], X_test.iloc[i]))

# Global summary plot (desktop/notebook)
shap.summary_plot(shap_values[pos_index], X_test)
