import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import joblib

CONFIG_PATH = "config.json"


def load_data(churn_fn, web_visits_fn, claims_fn, app_use_fn, CV_group="Train"):
    """Load data and split into Train/CV groups.
    Args:
        churn_fn (str): File path for churn data.
        web_visits_fn (str): File path for web visits data.
        claims_fn (str): File path for claims data.
        app_use_fn (str): File path for app usage data.
        CV_group (str): Which group to return ("Train", "CV", or "All").

    Returns:
        churn_df (pd.DataFrame): Churn data for the specified CV group (see docs/schema_churn_labels.md).
        web_visits_df (pd.DataFrame): Web visits data for the specified CV group (see docs/schema_web_visits.md).
        claims_df (pd.DataFrame): Claims data for the specified CV group (see docs/schema_claims.md).
        app_use_df (pd.DataFrame): App usage data for the specified CV group (see docs/schema_app_use.md).
    """

    full_churn_df = pd.read_csv(churn_fn, index_col=0, header=0)
    full_web_visits_df = pd.read_csv(web_visits_fn, index_col=0, header=0)
    full_claims_df = pd.read_csv(claims_fn, index_col=0, header=0)
    full_app_use_df = pd.read_csv(app_use_fn, index_col=0, header=0)

    # convert timestamp columns to datetime
    full_app_use_df["timestamp"] = pd.to_datetime(full_app_use_df["timestamp"])
    full_churn_df["signup_date"] = pd.to_datetime(full_churn_df["signup_date"])
    full_web_visits_df["timestamp"] = pd.to_datetime(full_web_visits_df["timestamp"])
    full_claims_df["diagnosis_date"] = pd.to_datetime(full_claims_df["diagnosis_date"])

    if CV_group == "Test":
        # always take full matrices for test group (no train-CV split)
        return full_churn_df, full_web_visits_df, full_claims_df, full_app_use_df

    # Extract unique subjects
    subjects = full_churn_df.index.unique()

    # test_size=0.2 means 20% of subjects go to CV
    train_ids, cv_ids = train_test_split(subjects, test_size=0.2, random_state=42)

    # train-CV split
    # churn
    train_churn_df = full_churn_df.loc[train_ids]
    cv_churn_df = full_churn_df.loc[cv_ids]
    # web visits
    train_web_visits_df = full_web_visits_df.loc[
        full_web_visits_df.index.intersection(train_ids)
    ]
    cv_web_visits_df = full_web_visits_df.loc[
        full_web_visits_df.index.intersection(cv_ids)
    ]
    # claims
    train_claims_df = full_claims_df.loc[full_claims_df.index.intersection(train_ids)]
    cv_claims_df = full_claims_df.loc[full_claims_df.index.intersection(cv_ids)]
    # app usage
    train_app_use_df = full_app_use_df.loc[
        full_app_use_df.index.intersection(train_ids)
    ]
    cv_app_use_df = full_app_use_df.loc[full_app_use_df.index.intersection(cv_ids)]

    # Verification (Optional but good for homework)
    print(f"Train subjects: {train_churn_df.index.nunique()}")
    print(f"CV subjects:    {cv_churn_df.index.nunique()}")

    # get all dfs for the specified CV group
    group_mapping = {
        "Train": (
            train_churn_df,
            train_web_visits_df,
            train_claims_df,
            train_app_use_df,
        ),
        "CV": (cv_churn_df, cv_web_visits_df, cv_claims_df, cv_app_use_df),
        "All": (full_churn_df, full_web_visits_df, full_claims_df, full_app_use_df),
    }
    if CV_group not in group_mapping:
        raise ValueError(f"Unknown CV_group: {CV_group}")
    churn_df, web_visits_df, claims_df, app_use_df = group_mapping.get(CV_group)

    return churn_df, web_visits_df, claims_df, app_use_df


def precision_recall_plot(y_test, y_proba, fig_path=None, title=None):
    """precision-recall curve"""

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    plt.figure(figsize=(6, 6))
    plt.plot(recalls, precisions, linewidth=2)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    if fig_path is not None:
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")


def align_features_and_labels(feat_df, churn_df, target_col="churn"):

    data_df = feat_df.join(churn_df[[target_col]], how="inner", validate="one_to_one")
    X = data_df.drop(columns=target_col)
    y = data_df[target_col]

    return X, y


def slope(x):
    """Calculate the slope of a sequence of values using linear regression."""
    if len(x) < 3:
        return np.nan
    return np.polyfit(range(len(x)), x, 1)[0]


def load_model(model_type="xgb"):

    model_path = get_model_path(model_type=model_type)
    model = joblib.load(model_path)

    print(f"Model loaded from {model_path}")
    if "feature_selection" in model_type:
        # load importance and selected features if feature selection was used
        selected_path = model_path.replace(".joblib", "_selected_features.pkl")
        importance_path = model_path.replace(".joblib", "_feature_importance.csv")
        if os.path.exists(selected_path):
            selected_features = joblib.load(selected_path)
            print(f"Selected features loaded from {selected_path}")
        else:
            print(f"Selected features file not found at {selected_path}")

        if os.path.exists(importance_path):
            importance_df = pd.read_csv(importance_path)
            print(f"Feature importance loaded from {importance_path}")
        else:
            print(f"Feature importance file not found at {importance_path}")

    return model


def get_model_path(model_type="xgb"):

    config_dict = pd.read_json(CONFIG_PATH, typ="series").to_dict()

    if "xgb" in model_type:
        model_file_name = (
            model_type + config_dict["output_file_names"]["model_name_suffix"]
        )
    else:  # raise value exception: only xgb supports saving
        raise ValueError(
            f"Model type {model_type} not supported for saving. Only 'xgb' is supported."
        )

    model_dir = config_dict["model_dir"]
    # create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}{model_file_name}"

    return model_path


# Define time of day categories
def time_of_day(hour):
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "noon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"
