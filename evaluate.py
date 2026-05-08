from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def evaluate(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0),
        "report": classification_report(y_true, y_pred, zero_division=0),
        "confusion matrix": confusion_matrix(y_true, y_pred)
    }
