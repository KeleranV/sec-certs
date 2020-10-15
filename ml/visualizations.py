from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn')


# So far I didn't figure out how to compute this for cross-validation case (technically it's a mess, mentally ok).
# Result: We will compute them on train set, obtaining biased (but still informative) results.
def plot_feature_importances(model, x_test, y_test):
    result = permutation_importance(model, x_test, y_test, n_repeats=20,
                                    random_state=42, n_jobs=8)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=x_test.columns[sorted_idx])
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.show()


def print_confusion_matrix(confusion_matrix, class_names, figsize=(13, 10), fontsize=14, filepath=None):
    # Copy pasted: https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        sns.heatmap(df_cm, annot=True, fmt=".1%", cmap='Greens')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if filepath is not None:
        fig.savefig(filepath, bbox_inches='tight', pad_inches=0.1, dpi=300)


def cross_validate_plot_cm(model, x_train, y_train, target_var_dict, n_splits=10, filepath=None):
    labels = list(target_var_dict.keys())
    range_labels = list(target_var_dict.values())

    kf = StratifiedKFold(n_splits=n_splits, random_state=42)
    cm_dim = len(range_labels)
    cm_sum = np.zeros((cm_dim, cm_dim))
    for train_index, test_index in kf.split(x_train, y_train):
        model.fit(x_train.iloc[train_index], y_train.iloc[train_index])
        cm = confusion_matrix(y_train.iloc[test_index], model.predict(x_train.iloc[test_index]), labels=range_labels)
        cm_sum = np.add(cm_sum, cm)

    # normalize
    cm_sum = cm_sum.astype('float') / cm_sum.sum(axis=1)[:, np.newaxis]
    # fill-in missing values (if class with 0 instances)
    cm_sum = np.nan_to_num(cm_sum)
    print_confusion_matrix(cm_sum, labels, filepath=filepath)
