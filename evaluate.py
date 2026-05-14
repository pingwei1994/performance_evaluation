import pandas as pd
import numpy as np
from sklearn.metrics import classification_report , confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, normalize=False):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    labels = np.unique(y_true)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""), fontsize=16)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                fontsize=14,
                fontweight="bold" if i == j else "normal"
            )

    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor")
    ax.tick_params(which="minor", bottom=False, left=False)

    return fig


def main():
    annotated = pd.read_csv('data/all_annotations.csv')[['report_id', 'standard_drug', 'standard_symptom','final_label']]
    annotated.dropna(subset=['final_label'], inplace=True)
    predicted = pd.read_csv('data/LLM_df.csv')
    if annotated.duplicated(subset=['report_id', 'standard_drug', 'standard_symptom']).any():
        return 'all_annotations.csv contains duplications'
    else:
        annotated['final_label'] = annotated['final_label'].apply(lambda x: 1 if x=='Positive' else (0 if x=='Negative' else 2))
        evaluate = predicted.merge(annotated, on=['report_id', 'standard_drug', 'standard_symptom'], how='inner')
        return classification_report(evaluate['final_label'], evaluate['pred']), evaluate, plot_confusion_matrix(evaluate['final_label'], evaluate['pred'])


if __name__ == "__main__":
    main()


