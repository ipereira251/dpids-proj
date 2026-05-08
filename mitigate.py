import numpy as np

def mitigation_layer(
    preds,
    probs,
    confidence,
    benign_label,
    entropy_threshold=1.3, #fewer interventions
    confidence_threshold=0.35, #more conservative intervention
    alpha=0.25  # strength of minority bias, decrease for less agressive attack bias
):

    refined = preds.copy()
    eps = 1e-9

    entropy = -np.sum(probs * np.log(probs + eps), axis=1)

    # estimate class frequencies from predictions
    class_counts = np.bincount(preds)
    class_counts = class_counts + 1e-6  # avoid divide by zero

    # minority boost weights (inverse frequency, normalized)
    class_weights = (1.0 / class_counts)
    class_weights = class_weights / np.max(class_weights)

    for i in range(len(preds)):

        uncertain = (
            confidence[i] < confidence_threshold
            and entropy[i] > entropy_threshold
        )

        if not uncertain:
            continue

        # weighted probability adjustment 
        adjusted = probs[i] * (1 + alpha * class_weights)

        # normalize back to probability distribution
        adjusted = adjusted / (np.sum(adjusted) + eps)

        refined[i] = np.argmax(adjusted)

        # benign safety constraint
        if refined[i] != benign_label and probs[i][benign_label] > 0.6:
            refined[i] = benign_label

    return refined
