import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

from scripts.data_ingest import load_raw, save_raw_parquet
from scripts.preprocessing import load_parquet, preprocess, save_processed
from scripts.train import build_training_pipeline


BASE_DIR = Path(__file__).resolve().parent
MODEL_OUTPUT = BASE_DIR / "models" / "discount_model.joblib"


# 1) Ingest
df_raw = load_raw(BASE_DIR)
save_raw_parquet(df_raw, BASE_DIR)

# 2) Preprocess
df_loaded = load_parquet(BASE_DIR)
df_processed = preprocess(df_loaded)
save_processed(df_processed, BASE_DIR)

# 3) Prepare features
target = "Discount"
X = df_processed.drop(columns=[target])
y = df_processed[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) Train model
pipeline = build_training_pipeline(X_train)
pipeline.fit(X_train, y_train)

# 5) Save model
MODEL_OUTPUT.parent.mkdir(exist_ok=True)
joblib.dump(pipeline, MODEL_OUTPUT)

print("Full ML pipeline complete. Model saved.")