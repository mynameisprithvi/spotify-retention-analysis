from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(model_type="logistic"):

    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

    elif model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42
        )
        pipeline = Pipeline([
            ("model", model)
        ])

    else:
        raise ValueError("model_type must be 'logistic' or 'rf'")

    return pipeline