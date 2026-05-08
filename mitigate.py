import numpy as np

def mitigation_layer(
    preds,
    probs,
    confidence,
    benign_label,
    entropy_threshold=0.9, #fewer interventions
    confidence_threshold=0.35, #more conservative intervention
    alpha=1.5,  # strength of minority bias, decrease for less agressive attack bias
    debug=True
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

    trigger_count = 0
    changed_preds = 0
    
    for i in range(len(preds)):

        uncertain = (
            confidence[i] < confidence_threshold
            and entropy[i] > entropy_threshold
        )

        if not uncertain:
            continue

        trigger_count += 1


        old_pred = preds[i]

        # weighted probability adjustment 
        adjusted = probs[i] ** 0.6 #(1 + alpha * class_weights)

        # normalize back to probability distribution
        adjusted = adjusted * (1 + alpha * class_weights)
        
        adjusted = adjusted / (np.sum(adjusted) + eps)

        new_pred = np.argmax(adjusted)

        # benign safety constraint
        if new_pred != benign_label and probs[i][benign_label] > 0.75:
            new_pred = benign_label

        refined[i] = new_pred

        if new_pred != old_pred:
            changed_preds += 1

        if debug:
            print("mitigation debug")
            print("mitigations triggered", trigger_count)
            print("preds changed", changed_preds)
            print("\n")

    return refined
