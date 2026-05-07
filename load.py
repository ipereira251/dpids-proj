import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def clean_labels(labels):
    return labels.astype(str).str.replace("�", "", regex=False).str.strip()

def map_labels(label):
    label = str(label).lower()

    if "benign" in label:
        return "BENIGN"
    elif "dos" in label or "ddos" in label:
        return "DOS"
    elif "portscan" in label:
        return "SCAN"
    elif "web attack" in label:
        return "WEB"
    elif "infiltration" in label:
        return "INF"
    else:
        return "OTHER"

def load_dataset(train_files, test_files, label_col):
    dfs = []
    for f in train_files + test_files:

        df = pd.read_csv(f, low_memory=False)

        df.columns = df.columns.str.strip()

        # keep only label, numeric
        numeric = df.select_dtypes(include=[np.number])

        if label_col not in df.columns:
            continue

        df = pd.concat([numeric, df[[label_col]]], axis=1)

        # small per-file
        df = df.sample(frac=0.2, random_state=42)

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True, copy=False)

    # labels
    labels = clean_labels(df[label_col]).apply(map_labels)

    X = df.select_dtypes(include=[np.number]).astype(np.float32)

    X[np.isinf(X)] = np.nan
    valid = ~X.isna().any(axis=1)

    X = X[valid]
    labels = labels[valid]

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # encode
    le = LabelEncoder()
    le.fit(y_train)

    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    # scale
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, le