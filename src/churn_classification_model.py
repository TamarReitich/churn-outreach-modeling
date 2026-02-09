import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from create_feat_df import build_feat_df
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from xgboost import XGBClassifier
import joblib
import os
from utils import (
    precision_recall_plot,
    align_features_and_labels,
    load_model,
    get_model_path,
)
import argparse
import warnings

CONFIG_PATH = "config.json"


def main():

    args = parse_args()
    print(f"Model type: {args.model_type}")
    print(f"Train model: {args.train_model}")

    load_saved_model = not args.train_model
    run_churn_modeling_pipeline(
        model_type=args.model_type, load_saved_model=load_saved_model
    )


### main pipeline functions ###


def run_churn_modeling_pipeline(
    model_type="logistic_regression", load_saved_model=False
):

    # load training and cv data
    train_feat_df, train_churn_df = build_feat_df("Train")
    CV_feat_df, CV_churn_df = build_feat_df("CV")

    X_train, y_train = align_features_and_labels(train_feat_df, train_churn_df)
    X_CV, y_CV = align_features_and_labels(CV_feat_df, CV_churn_df)

    if not load_saved_model:
        # fit model
        model = fit_churn_model(X_train, y_train, model_type=model_type)
    else:
        # load model
        model = load_model(model_type=model_type)

    # evaluate model on CV set and return AUC
    p_train = predict_churn_probability(model, X_train, model_type=model_type)
    p_CV = predict_churn_probability(model, X_CV, model_type=model_type)

    train_metrics = evaluate_model(p_train, y_train)
    cv_metrics = evaluate_model(p_CV, y_CV)

    # print erformance
    print_model_performance(train_metrics, dataset_name="Train")
    print_model_performance(cv_metrics, dataset_name="CV")

    # evaluate model for outreach = 0 and outreach = 1 separately
    X_CV_subset, y_CV_subset, p_CV_subset, subset_metrics = [], [], [], []
    for outreach_value in [0, 1]:
        print(f"\nEvaluating model performance for outreach = {outreach_value}...")
        X_CV_subset.append(X_CV[X_CV["outreach"] == outreach_value])
        y_CV_subset.append(y_CV[X_CV["outreach"] == outreach_value])
        p_CV_subset.append(
            predict_churn_probability(
                model, X_CV_subset[outreach_value], model_type=model_type
            )
        )
        subset_metrics.append(
            evaluate_model(p_CV_subset[outreach_value], y_CV_subset[outreach_value])
        )
        print_model_performance(
            subset_metrics[outreach_value],
            dataset_name=f"CV (Outreach={outreach_value})",
        )

    # plot precision-recall curve
    pr_curve_path = get_model_path(model_type=model_type).replace(
        ".joblib", "_precision_recall_curve_CV_group.png"
    )
    precision_recall_plot(
        y_CV,
        p_CV,
        fig_path=pr_curve_path,
        title=f"Precision-Recall Curve for CV Set (Model: {model_type})",
    )


def fit_churn_model(X, y, model_type="logistic_regression"):
    """Train a churn classification model using the provided feature dataframe and churn labels.

    Args:
        feat_df (pd.DataFrame): DataFrame containing features for each member_id.
        churn_df (pd.DataFrame): DataFrame containing churn labels for each member_id.

    Returns:
        model: Trained classification model.
    """
    # Train a simple logistic regression model for "baseline" performance
    if model_type == "logistic_regression":

        # Handle missing values (if any) - for simplicity, we'll fill with median
        X_filled = X.fillna(X.median())

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_filled, y)

    elif "xgb" in model_type:
        # use XGBoost for potentially better performance
        # Main training flow
        # calculate imbalance ratio for scale_pos_weight
        num_pos = sum(y)
        num_neg = len(y) - num_pos
        scale_pos_weight = (
            num_neg / num_pos if num_pos > 0 else 1
        )  # handle class imbalance

        print(f"Class imbalance - Positive: {num_pos}, Negative: {num_neg}")
        print(f"Scale pos weight: {scale_pos_weight:.2f}\n")

        if "feature_selection" in model_type:
            # Step 1: Train initial model for feature selection
            selector_model = train_for_feature_selection(X, y, scale_pos_weight)

            # Step 2: Select features based on importance
            X_selected, selected_features, importance_df = (
                select_features_by_importance(X, y, selector_model, threshold=0.01)
            )
        else:
            # If no feature selection, use all features
            X_selected = X
            selected_features = X.columns.tolist()
            importance_df = None

        # Step 3: Train final model on selected features
        model = train_final_model(X_selected, y, scale_pos_weight)

        # save the model
        save_model(
            model,
            model_type=model_type,
            selected_features=selected_features,
            importance_df=importance_df,
        )
    return model


def train_final_model(X, y, scale_pos_weight):
    """Train final model with grid search and early stopping"""

    # config model for grid search
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )

    # param grid for tuning - focus on regularization and tree complexity to reduce overfitting
    param_grid = {
        "max_depth": [2, 3],              # Keep shallow
        "learning_rate": [0.01, 0.03],    
        "subsample": [0.7, 0.8],          # Reduced options
        "colsample_bytree": [0.7, 0.8],   # Reduced options
        "min_child_weight": [15, 25],     # Higher values for more regularization
        "reg_alpha": [3.0, 5.0],          # L1 regularization
        "reg_lambda": [7.0, 10.0],        # L2 regularization
        "gamma": [0.5, 1.0]               # enforce pruning
    }

    # we want to optimize for positive churn,
    # use unweigted f1

    scoring = "f1"

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring=scoring,  # weighted F1 for class imbalance
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=1,
    )

    print("\nStarting grid search on selected features...")
    # Hyperparameter tuning with grid search
    grid.fit(X, y)

    print(f"\nBest parameters: {grid.best_params_}")
    print(f"Best CV score: {grid.best_score_:.4f}")

    # Get best parameters
    best_params = grid.best_estimator_.get_params()
    best_params.pop("early_stopping_rounds", None)  # Remove if exists

    # After grid search, retrain with early stopping for additional overfitting control
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Create final model with best params + early stopping
    print("\nTraining final model with early stopping...")
    model = XGBClassifier(**best_params, early_stopping_rounds=20)

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    return model


def predict_churn_probability(
    model, X, model_type="logistic_regression", return_df=False
):

    X = X.copy()
    # evaluare the model
    if model_type == "logistic_regression":

        # Handle missing values (if any) - for simplicity, we'll fill with median
        X = X.fillna(X.median())

    if "feature_selection" in model_type:
        # load selected features if feature selection was used
        model_path = get_model_path(model_type=model_type)
        selected_path = model_path.replace(".joblib", "_selected_features.pkl")
        if os.path.exists(selected_path):
            selected_features = joblib.load(selected_path)
            print(f"Using selected features from {selected_path}")
            X = X[selected_features]
        else:
            print(
                f"Selected features file not found at {selected_path}. Using all features."
            )

    p = model.predict_proba(X)[:, 1]
    print(f"Predicted probabilities using {X.shape[1]} features for {len(p)} samples.")
    if return_df:
        return pd.DataFrame({"member_id": X.index, "predicted_churn_probability": p})

    return p


def evaluate_model(p, y):

    # AUC
    auc = roc_auc_score(y, p)

    # For classification report and confusion matrix, we need binary predictions
    y_pred = (p >= 0.5).astype(int)

    class_report = classification_report(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)

    metrics = {
        "AUC": auc,
        "Classification Report": class_report,
        "Confusion Matrix": conf_matrix,
    }

    return metrics


### saving and loading related model functions ###
def save_model(model, model_type="xgb", selected_features=None, importance_df=None):
    model_path = get_model_path(model_type=model_type)

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    if selected_features is not None:
        selected_path = model_path.replace(".joblib", "_selected_features.pkl")
        joblib.dump(selected_features, selected_path)
    if importance_df is not None:
        importance_path = model_path.replace(".joblib", "_feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)


### feature selection functions  ###
def train_for_feature_selection(X, y, scale_pos_weight):
    """Train a simple model to get feature importances"""
    print("Training initial model for feature selection...")

    # Simple model for feature importance - use regularization to align with later grid search
    selector_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=2,
        learning_rate=0.01,
        subsample=0.7,
        min_child_weight=15,
        reg_alpha=3.0,
        reg_lambda=7.0,
        gamma=0.5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )

    selector_model.fit(X, y)
    return selector_model


def select_features_by_importance(X, y, model, threshold=0.005):
    """Select features based on importance threshold"""
    # Get feature importances
    importance_df = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    print(f"\nTop 20 Features:")
    print(importance_df.head(20).to_string(index=False))

    # Select features above threshold
    selected_features = importance_df[importance_df["importance"] > threshold][
        "feature"
    ].tolist()

    print(f"\n{'='*60}")
    print(f"Features selected: {len(selected_features)} out of {len(X.columns)}")
    print(f"Threshold: {threshold}")
    print(f"{'='*60}\n")

    X_selected = X[selected_features]

    return X_selected, selected_features, importance_df


# print performance
def print_model_performance(metrics, dataset_name):
    print("\n" + "=" * 60)
    print(f"MODEL PERFORMANCE ON {dataset_name.upper()} SET")
    print("=" * 60)
    print(f"AUC: {metrics['AUC']:.4f}")
    print(f"\nClassification Report:\n{metrics['Classification Report']}")
    print(f"\nConfusion Matrix:\n{metrics['Confusion Matrix']}")


def parse_args():
    """
    Parse command-line arguments.

    Example usage:
        python churn_classification_model.py -m xgb --train_model
        python churn_classification_model.py --model_type xgb_feature_selection

    Arguments:
        -m, --model_type      Model type to use ("xgb", "xgb_feature_selection"), default: "xgb"
        -t, --train_model     Flag to train the model (default: False)
    """
    parser = argparse.ArgumentParser(description="Churn Classification Model Runner")
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        choices=["xgb", "xgb_feature_selection"],
        default="xgb_feature_selection",
        help="Model type to use ('xgb' or 'xgb_feature_selection'), default: 'xgb'",
    )
    parser.add_argument(
        "-t", "--train_model", action="store_true", help="Flag to train the model"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Silence FutureWarnings
    print("Silencing FutureWarnings for cleaner output...")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    print("Starting churn modeling...")
    main()
