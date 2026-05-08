from load import load_dataset
from privacy import add_gaussian_noise
from classify import train_model, predict_with_confidence
from mitigate import mitigation_layer
from evaluate import evaluate
import numpy as np

def run_experiment(train_files, test_files, label_col, sigma=0.2):
    X_train, X_test, y_train, y_test, le = load_dataset(
        train_files,
        test_files,
        label_col
    )

    benign_label = np.where(le.classes_ == "BENIGN")[0][0]

    # clean
    model_clean = train_model(X_train, y_train)
    preds_clean, probs_clean, conf_clean = predict_with_confidence(model_clean, X_test)
    results_clean = evaluate(y_test, preds_clean)

    # dp simulation
    X_train_noisy = add_gaussian_noise(X_train, sigma)
    model_noisy = train_model(X_train_noisy, y_train)

    preds_noisy, probs_noisy, conf_noisy = predict_with_confidence(model_noisy, X_test)
    results_noisy = evaluate(y_test, preds_noisy)

    # mitigation
    preds_mitigated = mitigation_layer(
        preds_noisy,
        probs_noisy,
        conf_noisy,
        benign_label=benign_label
    )

    results_mitigated = evaluate(y_test, preds_mitigated)

    # results
    print("\n=== RESULTS ===")
    print("Clean:\n")
    print(results_clean["report"])
    print(results_clean["confusion matrix"])

    print("\nNoisy:\n")
    print(results_noisy["report"])
    print(results_noisy["confusion matrix"])

    print("\nMitigated:\n")
    print(results_mitigated["report"])
    print(results_mitigated["confusion matrix"])

if __name__ == "__main__":

    files = [
        "data/Monday-WorkingHours.pcap_ISCX.csv",
        "data/Tuesday-WorkingHours.pcap_ISCX.csv",
        "data/Wednesday-workingHours.pcap_ISCX.csv",
        "data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "data/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "data/Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    ]

    run_experiment(files, [], "Label", sigma=0.2)
