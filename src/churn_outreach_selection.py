from create_feat_df import build_feat_df
from churn_classification_model import predict_churn_probability
from utils import (
    precision_recall_plot,
    align_features_and_labels,
    load_model,
    get_model_path,
)
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import warnings

CONFIG_PATH = "config.json"


def main():

    args = parse_args()
    print(f"Model type: {args.model_type}")

    deterimine_outreach(model_type=args.model_type)


def deterimine_outreach(model_type="xgb"):
    """Determine which members to target for outreach based on predicted churn probabilities with and without outreach."""
    # load test and cv data
    test_feat_df, test_churn_df = build_feat_df("Test")

    X_test, y_test = align_features_and_labels(test_feat_df, test_churn_df)

    # load model
    model = load_model(model_type=model_type)

    X_test_outreach_0 = X_test.copy().fillna(0)
    X_test_outreach_1 = X_test.copy().fillna(1)

    # evaluate model on test data for outreach = 0 and outreach = 1
    # replace missing outreach values
    p_test_no_outreach_df = predict_churn_probability(
        model, X_test_outreach_0, model_type=model_type, return_df=True
    )
    p_test_outreach_df = predict_churn_probability(
        model, X_test_outreach_1, model_type=model_type, return_df=True
    )

    p_test_no_outreach = p_test_no_outreach_df["predicted_churn_probability"].values
    p_test_outreach = p_test_outreach_df["predicted_churn_probability"].values

    # examine the probability of churn with and without outreach for each test sample
    # interestef in pariticipants that have higher churn probability without outreach, as they are the ones that would benefit most from outreach
    p_diff = p_test_no_outreach - p_test_outreach
    sorted_diff = np.sort(p_diff)[::-1]

    uplift_thresh = 0.01

    # create a CSV file containing a sorted list of the top 'n' members for outreach.
    # This file includes member_id, a prioritization score (p_diff), and the member's rank.
    priority_df = pd.DataFrame(
        {"member_id": X_test.index, "prioritization_uplift_score": p_diff}
    )
    # screen for members with p_no_outreach > 0.5 and p_diff > uplift_thresh
    priority_df = priority_df[(p_test_no_outreach > 0.5) & (p_diff > uplift_thresh)]
    priority_df = priority_df.sort_values(
        by="prioritization_uplift_score", ascending=False
    ).reset_index(drop=True)
    priority_df["rank"] = priority_df.index + 1
    priority_df.to_csv(
        get_model_path(model_type=model_type).replace(
            ".joblib", "_outreach_prioritization.csv"
        ),
        index=False,
    )

    # plot for stakeholders: who should we target for outreach?
    plot_targeting_map(
        p0=p_test_no_outreach,
        p1=p_test_outreach,
        p0_thresh=0.5,
        uplift_thresh=uplift_thresh,
        figsize=(9, 7),
    )
    fig_path = get_model_path(model_type=model_type).replace(
        ".joblib", "_outreach_targeting_map.png"
    )
    plt.savefig(fig_path)

    # plot the difference from highest to lowest
    # 2 subplots: full range and trimmed to n
    n = 2000
    # subplot 1: relevant participants with highest predicted outreach effect
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    axs = axs.flatten()
    axs[0].plot(sorted_diff[:n])
    axs[0].set_title(
        "Difference in Predicted Churn Probability (Outreach vs No Outreach)"
    )
    axs[0].axhline(0, color="black", linestyle="--")
    axs[0].set_xlabel("# Test Samples (sorted by difference)")
    axs[0].set_ylabel("P(churn | No Outreach) - P(churn | Outreach)")

    # subplot 2: all participants
    axs[1].plot(sorted_diff)
    axs[1].axhline(0, color="black", linestyle="--")
    axs[1].set_ylabel("P(churn | No Outreach) - P(churn | Outreach)")

    # subplot 3:
    # plot the distribution of the difference vs the probability of churn without outreach
    axs[2].scatter(p_test_no_outreach, p_diff, alpha=0.5)
    axs[2].set_xlabel("P(churn | no outreach)")
    axs[2].set_ylabel("P(churn | No Outreach) - P(churn | Outreach)")

    # plot the difference from highest to lowest only for high churn users
    high_churn_mask = p_test_no_outreach > 0.5
    sorted_diff_high_churn = np.sort(p_diff[high_churn_mask])[::-1]
    axs[3].plot(sorted_diff_high_churn[:n])
    axs[3].axhline(0.05, color="black", linestyle="--")
    axs[3].set_title("Difference in Predicted Churn Probability for High-Risk Users")
    axs[3].set_ylabel("P(churn | no outreach) - P(churn | outreach)")
    axs[3].set_xlabel("# Test Samples (sorted by difference)")
    axs[3].set_ylabel("P(churn | No Outreach) - P(churn | Outreach)")

    # save figure
    fig_path = get_model_path(model_type=model_type).replace(
        ".joblib", "_outreach_effect.png"
    )
    plt.savefig(fig_path)


def plot_targeting_map(p0, p1, p0_thresh=0.5, uplift_thresh=0.01, figsize=(9, 7)):
    """
    Plot a stakeholder-friendly targeting map for outreach decisions.

    Parameters
    ----------
    p0 : array-like
        P(churn | no outreach)
    p1 : array-like
        P(churn | outreach)
    p0_thresh : float
        Baseline churn risk threshold
    uplift_thresh : float
        Minimum meaningful churn reduction
    figsize : tuple
        Figure size
    """

    uplift = p0 - p1

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=figsize)

    # Density-aware scatter
    sns.scatterplot(x=p0, y=uplift, s=15, alpha=0.35, edgecolor=None, ax=ax)

    # Decision boundaries
    ax.axvline(p0_thresh, linestyle="--", linewidth=1.5, label="At-risk threshold")
    ax.axhline(uplift_thresh, linestyle="--", linewidth=1.5, label="Minimum impact")

    # Shaded outreach region
    ax.fill_between(
        [p0_thresh, 1.0],
        uplift_thresh,
        uplift.max(),
        alpha=0.15,
        label="Outreach applied",
    )

    # Labels & title
    ax.set_xlabel("Churn risk without outreach")
    ax.set_ylabel("Expected churn reduction from outreach")
    ax.set_title("Who We Target for Outreach")

    ax.set_xlim(0, 1)
    ax.set_ylim(min(uplift.min(), -0.01), uplift.max() * 1.05)

    ax.legend(frameon=False)
    sns.despine()
    plt.tight_layout()


def parse_args():
    """
    Example usage:
        python churn_classification_model.py --model_type xgb_feature_selection

    Arguments:
        -m, --model_type      Model type to use ("xgb", "xgb_feature_selection"), default: "xgb"
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
    return parser.parse_args()


if __name__ == "__main__":

    # Silence FutureWarnings
    print("Silencing FutureWarnings for cleaner output...")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    print("Starting churn outreach selection process...")
    main()
