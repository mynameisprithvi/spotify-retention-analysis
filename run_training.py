import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from scripts.data_ingest import load_raw
from scripts.preprocessing import preprocess
from scripts.train import build_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_OUTPUT = BASE_DIR / "models" / "retention_model.joblib"

# 1) Load
df = load_raw(BASE_DIR)

# 2) Preprocess
df = preprocess(df)

# 3) Feature selection
feature_cols = [
    "danceability", "energy", "valence", "loudness",
    "speechiness", "acousticness", "tempo", "liveness"
]

X = df[feature_cols]
y = df["short_lived"]

# 4) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5) Train
pipeline = build_model(model_type="rf")
pipeline.fit(X_train, y_train)

# 6) Evaluate
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, y_pred_proba)

print(f"ROC-AUC: {roc:.3f}")

# 7) Save
MODEL_OUTPUT.parent.mkdir(exist_ok=True)
joblib.dump(pipeline, MODEL_OUTPUT)

print("Model saved.")