from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=18,
        n_jobs=-1,
        random_state=42
    )

    weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    model.fit(X_train, y_train, sample_weight=weights)
    return model

def predict_with_confidence(model, X):

    probs = model.predict_proba(X)
    preds = np.argmax(probs, axis=1)

    sorted_probs = np.sort(probs, axis=1)
    confidence = sorted_probs[:, -1] - sorted_probs[:, -2]

    return preds, probs, confidence