# create a script to test use of basic python data science tools like numpy and pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import circmean, circstd
from utils import load_data, slope

CONFIG_PATH = "config.json"


def build_feat_df(CV_group):

    # verify that CV_group is valid for exploration
    valid_groups = ["Train", "CV", "All", "Test"]
    if CV_group not in valid_groups:
        raise ValueError(
            f"Invalid CV_group '{CV_group}'. Must be one of {valid_groups}."
        )

    # get paths from json config file
    config_dict = pd.read_json(CONFIG_PATH, typ="series").to_dict()

    churn_fn = config_dict[CV_group]["data"]["churn"]
    web_visits_fn = config_dict[CV_group]["data"]["web_visits"]
    claims_fn = config_dict[CV_group]["data"]["claims"]
    app_use_fn = config_dict[CV_group]["data"]["app_usage"]

    churn_df, web_visits_df, claims_df, app_use_df = load_data(
        churn_fn, web_visits_fn, claims_fn, app_use_fn, CV_group=CV_group
    )

    # Extract claim features
    claim_features_df = get_claim_features(claims_df.copy())

    # extract web visit features
    web_visits_features_df = get_web_visits_features(web_visits_df.copy())

    # extract signup features from churn_df
    signup_date_feature_df = get_signup_features(churn_df.copy())

    # extract app use features
    app_use_features_df = get_app_use_features(app_use_df.copy())

    if (
        not CV_group == "Test"
    ):  # for testing, the difference in probabilities between outreach = 1/0 will be exmined
        # add outreach features for Train and CV groups
        outreach_features_df = churn_df[["outreach"]]
    else:  # for test set, outreach is not available, so add a placeholder column with all values set to NaN
        outreach_features_df = pd.DataFrame(index=churn_df.index, columns=["outreach"])

    # Merge all dataframes on member_id
    # verify that member_id is the index for all dataframes before merging
    if not all(
        df.index.name == "member_id"
        for df in [
            outreach_features_df,
            claim_features_df,
            web_visits_features_df,
            signup_date_feature_df,
            app_use_features_df,
        ]
    ):
        raise ValueError(
            "All dataframes must have 'member_id' as the index before merging."
        )

    feat_df = (
        outreach_features_df.join(claim_features_df, how="left")
        .join(web_visits_features_df, how="left")
        .join(signup_date_feature_df, how="left")
        .join(app_use_features_df, how="left")
    )

    if not CV_group == "Test":
        churn_df = churn_df[["churn"]]
    else:
        churn_df = pd.DataFrame(index=churn_df.index, columns=["churn"])

    return feat_df, churn_df


def get_claim_features(claims_df):
    """Extract features from claims data.
    Current features:
        - Number of claims for each ICD code (e.g. num_claims_icd_123)
        - Days since last claim for each ICD code (e.g. days_since_claim_icd_123)

        In later versions, can explore additional possible features.
    Args:
        claims_df (pd.DataFrame): Claims data for a set of subjects (see docs/schema_claims.md).

    Returns:
        claim_features_df (pd.DataFrame): DataFrame where each row corresponds to a subject and columns are claim-based

    """

    # get the number of claims per ICD code for each member
    member_icd_matrix = (
        claims_df.reset_index()
        .groupby(["member_id", "icd_code"])
        .size()
        .unstack(fill_value=0)
    )

    # add the date of the last claim for each claim as a feature
    ref_date = claims_df["diagnosis_date"].max()
    claims_df["days_since_claim"] = (
        (ref_date - claims_df["diagnosis_date"]).dt.days
    ).astype(int)

    # unstack min days since claim for each ICD code to get feature columns
    # fill missing values with a large number (e.g. 1000) to indicate no claims for that ICD code
    member_icd_date_matrix = (
        claims_df.reset_index()
        .groupby(["member_id", "icd_code"])["days_since_claim"]
        .min()
        .unstack(fill_value=1000)
    )

    # combine the two matrices - add suffixes to distinguish between count and days since claim features
    claim_features_df = member_icd_matrix.add_suffix("_count").join(
        member_icd_date_matrix.add_suffix("_days_since_claim"), how="outer"
    )

    return claim_features_df


def get_signup_features(churn_df):
    """Extract features from churn data.
    Args:
        churn_df (pd.DataFrame): Churn data for a set of subjects (see docs/schema_churn_labels.md).

    Returns:
        signup_features_df (pd.DataFrame): DataFrame where each row corresponds to a subject and a column is days since signup.

    In later versions, can explore additional possible features, e.g. signup months (possible seasonal effects), signup day of week, etc.
    """
    signup_features_df = pd.DataFrame(index=churn_df.index)
    # add day since sign up in int format as a feature
    signup_features_df["days_since_signup"] = (
        churn_df["signup_date"].max() - churn_df["signup_date"]
    ).dt.days.astype(int)

    return signup_features_df


def get_web_visits_features(web_visits_df):
    """Extract features from web visits data.
    Args:
        web_visits_df (pd.DataFrame): Web visits data for a set of subjects (see docs/schema_web_visits.md).

    Returns:
        web_visits_features_df (pd.DataFrame): DataFrame where each row corresponds to a subject and columns are web visit-based features.
    """
    # replace url, title and description by clusters
    # load clusters based on config file
    config_dict = pd.read_json(CONFIG_PATH, typ="series").to_dict()
    cluster_file_name = config_dict["output_file_names"]["web_visit_clusters"]
    data_output_dir = config_dict["Train"]["data_output_dir"]

    try:
        cluster_df = pd.read_csv(
            f"{data_output_dir}/{cluster_file_name}", index_col=0, header=0
        )
    except FileNotFoundError:
        print(
            f"Cluster file {cluster_file_name} not found in {data_output_dir}. Please run the data_exploration.py to generate the cluster file before running this function."
        )

    # create web visit df with cluster labels instead of url, title and description
    web_visits_clusters_df = web_visits_df[["description", "timestamp"]].copy()
    web_visits_clusters_df["cluster_label"] = cluster_df.loc[
        web_visits_df["description"], "cluster_label"
    ].values
    web_visits_clusters_df["app_relevance"] = cluster_df.loc[
        web_visits_df["description"], "app_relevance"
    ].values
    # map app_relevance labels to "relevant" and "not_relevant"
    web_visits_clusters_df["app_relevance"] = web_visits_clusters_df[
        "app_relevance"
    ].map({1: "relevant_websites", 0: "irrelevant_websites"})

    # calculate features:
    # add time from claim in hours
    web_visits_clusters_df["time_since_visit"] = (
        web_visits_clusters_df["timestamp"].max() - web_visits_clusters_df["timestamp"]
    ).dt.total_seconds() / 3600

    # # unstack min days since claim for each ICD code to get feature columns
    # # fill missing values with a large number (e.g. 1000) to indicate no claims for that ICD code

    # number of web visits for each cluster label
    # fill missing values with 0 - no visits for that cluster label
    member_n_web_visit_data = (
        web_visits_clusters_df.reset_index()
        .groupby(["member_id", "cluster_label"])
        .size()
        .unstack(fill_value=0)
        .add_prefix("cluster_")
    )
    # add number of web visits for app relevance catefory
    member_n_web_visit_data = member_n_web_visit_data.join(
        web_visits_clusters_df.reset_index()
        .groupby(["member_id", "app_relevance"])
        .size()
        .unstack(fill_value=0)
    )
    member_n_web_visit_data = member_n_web_visit_data.add_suffix("_n_web_visits")

    # take clusters with median n_web visits>2 to add trend features
    median_n_web_visit = member_n_web_visit_data.filter(regex="^cluster_").median()
    high_median_visit_clusters = (
        median_n_web_visit[median_n_web_visit > 2]
        .index.str.replace("_n_web_visits", "")
        .str.replace("cluster_", "")
    )
    filtered_web_visits_clusters_df = web_visits_clusters_df[
        web_visits_clusters_df["cluster_label"]
        .astype(str)
        .isin(high_median_visit_clusters)
    ]

    # add trend features for clusters with high rate of web visits
    web_visit_trend = (
        filtered_web_visits_clusters_df.groupby(["member_id", "cluster_label"])[
            "time_since_visit"
        ]
        .apply(slope)
        .unstack(fill_value=0)
        .add_suffix("_trend")
        .add_prefix("cluster_")
    )
    # fill nans with 0 slope = no trend
    web_visit_trend = web_visit_trend.isna().astype(int).fillna(0)

    # add trend for app relevance category for app relevant and irrelevant websites
    web_visit_trend = web_visit_trend.join(
        web_visits_clusters_df.groupby(["member_id", "app_relevance"])[
            "time_since_visit"
        ]
        .apply(slope)
        .unstack(fill_value=0)
        .add_suffix("_trend")
    )
    # fill nans with 0 slope = no trend
    web_visit_trend = web_visit_trend.fillna(0)

    # time from last web visit for each cluster label
    # fill missing values with 100000 - "inf" time since last visit for that cluster label
    member_last_web_visit_data = (
        web_visits_clusters_df.reset_index()
        .groupby(["member_id", "cluster_label"])["time_since_visit"]
        .min()
        .unstack(fill_value=100000)
        .add_prefix("cluster_")
    )
    # add time from last web visit for app relevance catefory
    member_last_web_visit_data = member_last_web_visit_data.join(
        web_visits_clusters_df.reset_index()
        .groupby(["member_id", "app_relevance"])["time_since_visit"]
        .min()
        .unstack(fill_value=100000),
        how="outer",
        rsuffix="_app_relevance",
    )
    member_last_web_visit_data = member_last_web_visit_data.add_suffix(
        "_time_since_web_visit"
    )

    # join the two dataframes to get the final web visit features dataframe
    web_visits_features_df = member_n_web_visit_data.join(
        member_last_web_visit_data, how="outer"
    ).join(web_visit_trend, how="outer")

    return web_visits_features_df


def get_app_use_features(app_use_df):
    """Extract features from app use data.
    Args:
        app_use_df (pd.DataFrame): App usage data for a set of subjects (see docs/schema_app_use.md).

    Returns:
        app_use_features_df (pd.DataFrame): DataFrame where each row corresponds to a subject and columns are app use-based features.

    Add variability of use time during the day in later versions (circular std)
    """
    # calculate features:
    # add time from app use in hours and time of day of app use
    app_use_df = app_use_df.copy().reset_index("member_id")
    app_use_df["time_since_app_use"] = (
        app_use_df["timestamp"].max() - app_use_df["timestamp"]
    ).dt.total_seconds() / 3600
    app_use_df["hour_of_day"] = (
        app_use_df["timestamp"].dt.hour
        + app_use_df["timestamp"].dt.minute / 60
        + app_use_df["timestamp"].dt.second / 3600
    )

    # time from last app use in hours
    member_last_app_use_data = (
        app_use_df.reset_index()
        .groupby(["member_id"])["time_since_app_use"]
        .min()
        .rename("time_since_last_app_use")
    )

    # number of app uses during observation window
    member_n_app_use_data = (
        app_use_df.reset_index().groupby(["member_id"]).size().rename("n_app_uses")
    )

    # trend in app use time during observation window - slope of time since app use over time (negative slope = more recent app uses, positive slope = less recent app uses)
    # add trend for app relevance category for app relevant and irrelevant websites
    app_use_trend = (
        app_use_df.groupby(["member_id"])["time_since_app_use"]
        .apply(slope)
        .rename("app_use_time_trend")
    ).to_frame()
    # fill nans with 0 slope = no trend
    app_use_trend = app_use_trend.isna().astype(int).fillna(0)

    # variability of use time: std of time between app uses
    std_app_use_data = (
        app_use_df.sort_values(
            ["member_id", "time_since_app_use"]
        )  # sort values to get consecutive app uses
        .groupby("member_id")["time_since_app_use"]
        .diff()  # calculate time difference
        .groupby(app_use_df["member_id"])
        .std()  # calculate std of time differences for each member
        .rename("app_use_time_diff_std")
    )

    # skewness of use time: detects if there are outliers in use time (e.g. one very long time between uses)
    skew_app_use_data = (
        app_use_df.sort_values(
            ["member_id", "time_since_app_use"]
        )  # sort values to get consecutive app uses
        .groupby("member_id")["time_since_app_use"]
        .diff()  # calculate time difference
        .groupby(app_use_df["member_id"])
        .skew()  # calculate std of time differences for each member
        .rename("app_use_time_diff_skew")
    )
    # typical use time of day - circular mean and std of time of day of app use
    mean_hour_day = (
        app_use_df.assign(angle=lambda df: df["hour_of_day"] * 2 * np.pi / 24)
        .groupby("member_id")["angle"]
        .apply(lambda x: circmean(x, high=2 * np.pi, low=0) * 24 / (2 * np.pi))
        .rename("mean_hour_day")
    )

    std_hour_day = (
        app_use_df.assign(angle=lambda df: df["hour_of_day"] * 2 * np.pi / 24)
        .groupby("member_id")["angle"]
        .apply(lambda x: circstd(x, high=2 * np.pi, low=0) * 24 / (2 * np.pi))
        .rename("std_hour_day")
    )
    typical_use_time_df = pd.concat([mean_hour_day, std_hour_day], axis=1)

    # join all df's
    app_use_features_df = pd.concat(
        [
            member_last_app_use_data,
            member_n_app_use_data,
            std_app_use_data,
            skew_app_use_data,
            typical_use_time_df,
            app_use_trend,
        ],
        axis=1,
    )

    return app_use_features_df


if __name__ == "__main__":
    # Example usage of the load_data function

    feat_df, churn_df = build_feat_df(CV_group="Train")
