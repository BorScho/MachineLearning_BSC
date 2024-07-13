from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib as plt
import numpy as np
import pandas as pd

def plot_roc(y_true, y_score, label_name, ax):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr)
    ax.plot([0,1],[0,1], color="grey", linestyle="--")
    ax.set_ylabel("TPR")
    ax.set_xlabel("FPR")
    ax.set_title(f"{label_name}: AUC = {roc_auc_score(y_true, y_score):.4f}")

def true_preds(preds, index_df):
    result =  pd.DataFrame(
        {
            "h1n1_vaccine": preds[0][:,1],
            "seasonal_vaccine": preds[1][:,1],
        },
        index=index_df
    )
    return result

def plot_roc_h1n1_and_seasonal(y_eval, y_preds):
    fig, ax = plt.subplots(1,2,figsize=(7, 3.5))

    plot_roc(
        y_eval["h1n1_vaccine"],
        y_preds["h1n1_vaccine"],
        "h1n1_vaccine",
        ax=ax[0]
    )

    plot_roc(
        y_eval["seasonal_vaccine"],
        y_preds["seasonal_vaccine"],
        "seasonal_vaccine",
        ax=ax[1]
    )

    fig.tight_layout()

def train_fullset_and_save(estimator, X_test_df, X_train_df, y_train_df,submission_data_filepath):
    submission_df = pd.read_csv("submission_format.csv", index_col="respondent_id")
    np.testing.assert_array_equal(submission_df.index.values, X_test_df.index.values)
    
    estimator.fit(X_train_df, y_train_df)
    test_preds = estimator.predict_proba(X_test_df)

    submission_df["h1n1_vaccine"]=test_preds[0][:,1]
    submission_df["seasonal_vaccine"]=test_preds[1][:,1]
    
    submission_df.to_csv(submission_data_filepath, index=True)