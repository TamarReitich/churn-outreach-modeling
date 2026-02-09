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
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import circmean, circstd
from utils import time_of_day, load_data
import warnings

CONFIG_PATH = "config.json"

"""Overview:
This script performs exploratory data analysis on healthcare member data to identify drivers of churn. 
It leverages pandas and numpy for data manipulation, scikit-learn and scipy for advanced analytics 
(PCA, hierarchical clustering, circular statistics), and seaborn/matplotlib for visualization.

Key Features
Temporal Analysis: Calculates churn probability by signup date and time-of-day app usage using circular means.
Behavioral Clustering: Uses SentenceTransformer to embed web search descriptions and hierarchical clustering to group similar user interests.
Intervention Impact: Facets all analyses by "outreach" status to visualize the relationship between marketing/clinical interventions and retention.
Medical Context: Correlates churn with specific ICD-10 diagnosis codes and claim volumes."""


def main():
    # run data exploration for train and CV groups
    for CV_group in ["Train", "CV"]:
        print(f"Running data exploration for {CV_group} group...")
        data_exploration(CV_group=CV_group)

    # also run on all data combined to get better power for some analyses (e.g. web visit description clustering)
    print(f"Running data exploration for All group...")
    data_exploration(CV_group="All")


def data_exploration(CV_group="Train"):

    # verify that CV_group is valid for exploration
    valid_groups = ["Train", "CV", "All"]
    if CV_group not in valid_groups:
        raise ValueError(
            f"Invalid CV_group '{CV_group}'. Must be one of {valid_groups}."
        )

    # get paths from json config file
    config_dict = pd.read_json(CONFIG_PATH, typ="series").to_dict()
    # create a directory to save figures
    fig_dir = config_dict[CV_group]["fig_dir"]
    data_output_dir = config_dict[CV_group]["data_output_dir"]

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_output_dir, exist_ok=True)

    churn_fn = config_dict[CV_group]["data"]["churn"]
    web_visits_fn = config_dict[CV_group]["data"]["web_visits"]
    claims_fn = config_dict[CV_group]["data"]["claims"]
    app_use_fn = config_dict[CV_group]["data"]["app_usage"]
    # load data
    churn_df, web_visits_df, claims_df, app_use_df = load_data(
        churn_fn, web_visits_fn, claims_fn, app_use_fn, CV_group=CV_group
    )

    # plot churn probability vs app use, separately for outreach vs no outreach
    plot_churn_vs_app_use(churn_df, app_use_df, fig_dir)

    # plot web visit description embeddings and cluster by similarity
    explore_web_visits(churn_df, web_visits_df, fig_dir, data_output_dir)

    # churn probability with and without outreach
    churn_p_outreach = churn_df[churn_df["outreach"] == 1]["churn"].mean()
    churn_p_no_outreach = churn_df[churn_df["outreach"] == 0]["churn"].mean()

    # plot churn probability vs time, across all and separately for outreach and no outreach
    for outreach_status, churn_p in zip(
        [1, 0], [churn_p_outreach, churn_p_no_outreach]
    ):
        print(f"Churn probability with outreach={outreach_status}: {churn_p:.4f}")
        # plot churn vs month
        plot_churn_vs_time(
            churn_df.copy(),
            fig_dir,
            time_unit="Month",
            separate_outreach=outreach_status,
        )
        # can also run with day/week granularity, but these are noisier and less interpretable than month, so not included in main script

    # visualize claims distribution
    plot_claims_distribution(claims_df, fig_dir, count_patients=True)

    # plot churn probability for patients with vs without specific claims
    plot_churn_by_claims_outreach(churn_df, claims_df, fig_dir)


def plot_churn_vs_app_use(churn_df, app_use_df, fig_dir):
    plot_churn_vs_num_uses(churn_df, app_use_df, fig_dir)
    plot_churn_vs_time_since_last_use(churn_df, app_use_df, fig_dir)
    plot_churn_vs_use_time_of_day(churn_df, app_use_df, fig_dir)


def plot_churn_vs_use_time_of_day(churn_df, app_use_df, fig_dir):

    # Copy data to avoid modifying original
    app_use_df = app_use_df.copy()
    churn_df = churn_df.copy()
    df = app_use_df.join(churn_df[["churn", "outreach"]], how="inner")

    # convert timestamp into hours
    df["time_of_day"] = df["timestamp"].dt.hour.apply(time_of_day)

    # get the most probable time of day for each use
    time_of_day_counts = (
        df.groupby(["member_id", "time_of_day"]).size().reset_index(name="count")
    )
    # get label by time of day with most uses
    ## note - taking the first of ties. a more careful account will treat ties differently
    time_of_day_labels = time_of_day_counts.loc[
        time_of_day_counts.groupby("member_id")["count"].idxmax()
    ][["member_id", "time_of_day"]]

    # plot churn probability by favorite time of day, separately for outreach vs no outreach
    df = time_of_day_labels.merge(
        churn_df[["churn", "outreach"]],
        left_on="member_id",
        right_index=True,
        how="inner",
    )
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="time_of_day", y="churn", hue="outreach", palette="muted")
    plt.title("Churn Probability by Favorite Time of Day for App Use")
    plt.xlabel("Favorite Time of Day for App Use")
    plt.ylabel("Churn Probability")

    # save figure
    plt.savefig(f"{fig_dir}/churn_vs_time_of_day.png", dpi=300)

    # circular mean of time of day for each subject and plot vs churn probability, separately for outreach vs no outreach
    app_use_df = app_use_df.copy()
    churn_df = churn_df.copy()
    df = app_use_df.join(churn_df[["churn", "outreach"]], how="inner")

    # add hour of day column
    df["hour_of_day"] = (
        df["timestamp"].dt.hour
        + df["timestamp"].dt.minute / 60
        + df["timestamp"].dt.second / 3600
    )
    member_means_df = (
        df.assign(angle=lambda df: df["hour_of_day"] * 2 * np.pi / 24)
        .groupby("member_id")["angle"]
        .apply(lambda x: circmean(x, high=2 * np.pi, low=0) * 24 / (2 * np.pi))
        .to_frame(name="mean_hour_day")
    )
    # Define bins (e.g., 1-hour bins)
    bins = np.linspace(0, 24, 25)  # 0-24 hours
    member_means_df["hour_bin"] = pd.cut(
        member_means_df["mean_hour_day"], bins, include_lowest=True
    )
    member_means_df = member_means_df.merge(
        churn_df[["churn", "outreach"]], left_index=True, right_index=True, how="inner"
    )

    # Compute mean churn per bin & outreach
    churn_by_bin = (
        member_means_df.groupby(["hour_bin", "outreach"], observed=False)["churn"]
        .mean()
        .reset_index()
    )

    # Bin midpoint for plotting
    churn_by_bin["hour_mid"] = churn_by_bin["hour_bin"].apply(
        lambda x: x.left + (x.right - x.left) / 2
    )

    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=churn_by_bin, x="hour_mid", y="churn", hue="outreach", marker="o")
    plt.xlabel("Circular mean time of day (hours)")
    plt.ylabel("Mean churn probability")
    plt.title("Churn probability vs per-member circular mean time of day")
    plt.xticks(range(0, 25, 2))

    # save figure
    plt.savefig(f"{fig_dir}/churn_vs_circular_mean_time_of_day.png", dpi=300)


def plot_churn_vs_time_since_last_use(churn_df, app_use_df, fig_dir):
    """Plot churn probability vs time since last app use, separately for outreach vs no outreach.
    Time bins are computed across all subjects (5% quantiles) to ensure same bins for both outreach groups.
    """

    # Copy data to avoid modifying original
    app_use_df = app_use_df.copy()
    churn_df = churn_df.copy()
    df = app_use_df.join(churn_df[["churn", "outreach"]], how="inner")

    # Compute last use per member
    last_use = df.groupby("member_id")["timestamp"].max().reset_index()
    last_use.rename(columns={"timestamp": "last_use"}, inplace=True)
    df = df.merge(last_use, on="member_id", how="left")

    # Outreach occurs 1 hour after last use of the app across all subjects
    df["outreach_time"] = df["last_use"].max() + pd.Timedelta(hours=1)

    # Time since last use (hours)
    df["hours_since_last_use"] = (
        df["outreach_time"] - df["last_use"]
    ).dt.total_seconds() / 3600

    # Compute bin edges across all subjects (5% quantiles)
    n_bins = 20  # 100% / 5%
    bin_edges = (
        df["hours_since_last_use"].quantile(np.linspace(0, 1, n_bins + 1)).values
    )

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Plot for each outreach group using same bins
    for outreach_val in df["outreach"].unique():
        subset = df[df["outreach"] == outreach_val].copy()

        # Bin based on shared edges
        subset["time_bin"] = pd.cut(
            subset["hours_since_last_use"], bins=bin_edges, include_lowest=True
        )

        # Aggregate churn probability per bin
        agg = (
            subset.groupby("time_bin", observed=False)
            .agg(
                churn_prob=("churn", "mean"),
                churn_sem=("churn", "sem"),
                time_mean=("hours_since_last_use", "mean"),
            )
            .reset_index()
        )

        # Plot line and error bars
        plt.errorbar(
            agg["time_mean"],
            agg["churn_prob"],
            yerr=agg["churn_sem"],
            marker="o",
            capsize=3,
            label=f"Outreach={outreach_val}",
        )

    plt.title("Churn Probability vs Time Since Last Use to Outreach")
    plt.xlabel("Hours Since Last Use")
    plt.ylabel("Churn Probability")
    plt.legend(title="Outreach")
    plt.tight_layout()

    # Save figure
    plt.savefig(f"{fig_dir}/churn_vs_time_since_last_use.png", dpi=300)
    plt.close()


def plot_churn_vs_num_uses(churn_df, app_use_df, fig_dir):
    """Plot churn probability vs exact number of app uses, separately for outreach vs no outreach."""

    # Copy data to avoid modifying original
    app_use_df = app_use_df.copy()
    churn_df = churn_df.copy()

    # Count number of uses per member
    usage_counts = app_use_df.groupby("member_id").size().reset_index(name="num_uses")

    # Merge counts with churn/outreach info
    df = churn_df[["churn", "outreach"]].merge(usage_counts, on="member_id", how="left")

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))

    # Plot separately for outreach groups
    for outreach_val in df["outreach"].unique():
        subset = df[df["outreach"] == outreach_val]

        # Aggregate churn probability per exact number of uses
        agg = (
            subset.groupby("num_uses")
            .agg(churn_prob=("churn", "mean"), churn_sem=("churn", "sem"))
            .reset_index()
        )

        # Plot line and error bars
        plt.errorbar(
            agg["num_uses"],
            agg["churn_prob"],
            yerr=agg["churn_sem"],
            marker="o",
            capsize=3,
            label=f"Outreach={outreach_val}",
        )
    # Styling
    max_uses = int(df["num_uses"].max())  # cast to int to fix TypeError
    step = max(1, max_uses // 10)  # set reasonable step
    plt.xticks(range(0, max_uses + 1, step))

    # labels with large font size and title
    plt.title("Churn Probability vs Number of App Uses", fontsize=16, pad=15)
    plt.xlabel("Number of App Uses During Obeservation Window", fontsize=14)
    plt.ylabel("Churn Probability", fontsize=14)
    plt.legend(title="Outreach")
    plt.tight_layout()

    # Save figure
    plt.savefig(f"{fig_dir}/churn_vs_num_uses.png", dpi=300)
    plt.close()


def explore_web_visits(churn_df, web_visits_df, fig_dir, data_output_dir):
    """get embeddings for web visit search queries using SentenceTransformer."""

    def get_embeddings_df(web_visits_df, emb_dir):
        csv_path = f"{emb_dir}/web_visits_description_embeddings.csv"
        if not os.path.exists(csv_path):
            from sentence_transformers import SentenceTransformer, util

            web_visits_df = web_visits_df.copy()
            # use sentence transformer to get embeddings for the web visit descriptions
            model = SentenceTransformer("all-MiniLM-L6-v2")

            unique_description = web_visits_df["description"].unique().tolist()
            embs = model.encode(unique_description)

            emb_df = pd.DataFrame(
                embs, columns=[f"emb_{i}" for i in range(embs.shape[1])]
            )
            # Add descriptions as a column
            emb_df["description"] = unique_description

            # Set 'description' as the index
            emb_df.set_index("description", inplace=True)

            # Check embedding shape (optional)
            print(f"Embedding shape: {embs.shape}")

            # Save to CSV with description as the index
            emb_df.to_csv(csv_path, index=True)
            print(f"Saved embeddings to {csv_path}")

        else:
            # read saved embeddings
            emb_df = pd.read_csv(csv_path, index_col=0)
            print(f"Loaded embeddings shape: {emb_df.shape}")

        return emb_df

    emb_dir = fig_dir
    emb_df = get_embeddings_df(web_visits_df, emb_dir)

    # plot pairwise distances between web visit description embeddings, clustered by similarity
    cluster_web_descriptions(emb_df, fig_dir, data_output_dir)


def explore_web_visits_vs_time(churn_df, web_visits_df, fig_dir):
    all_descriptions = web_visits_df.loc[:, "description"].unique()


def cluster_web_descriptions(emb_df, fig_dir, data_output_dir):
    """Plot pairwise cosine distances between web visit description embeddings."""
    embeddings = emb_df.values
    # Compute pairwise cosine distances
    condensed_dist = pdist(embeddings, metric="cosine")  # 1D condensed
    cosine_distances = squareform(condensed_dist)  # (26, 26)
    # Hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method="average")

    # plot the distance between clusters averages from the linkage matrix
    distance = linkage_matrix[:, 2]  # distance between clusters
    plt.figure(figsize=(10, 6))
    plt.plot(distance, marker="o")
    plt.title("Distance Between Clusters in Hierarchical Clustering")
    plt.xlabel("Cluster Merge Step")
    plt.ylabel("Distance")
    plt.savefig(f"{fig_dir}/web_visit_description_cluster_distances.png", dpi=300)

    # set threshold and get cluster labels based on distance
    threshold = 0.7  # to get reasonable amount of granularity based
    cluster_labels = fcluster(linkage_matrix, t=threshold, criterion="distance")

    # Seaborn clustermap (automatically sorts by similarity)
    g = sns.clustermap(
        cosine_distances,
        row_cluster=True,
        col_cluster=True,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        cmap="viridis",
        xticklabels=emb_df.index,
        yticklabels=emb_df.index,
        figsize=(12, 12),
        cbar_pos=(0.85, 0.05, 0.03, 0.2),
    )
    unique_clusters = np.unique(cluster_labels)
    palette = sns.color_palette("tab10", len(unique_clusters))

    cluster_to_color = dict(zip(unique_clusters, palette))

    fontsize = 16
    # Row labels
    row_order = g.dendrogram_row.reordered_ind
    for label, idx in zip(g.ax_heatmap.get_yticklabels(), row_order):
        label.set_color(cluster_to_color[cluster_labels[idx]])
        label.set_fontsize(fontsize)

    # Column labels
    col_order = g.dendrogram_col.reordered_ind
    for label, idx in zip(g.ax_heatmap.get_xticklabels(), col_order):
        label.set_color(cluster_to_color[cluster_labels[idx]])
        label.set_fontsize(fontsize)

    plt.title("Cosine Distance Heatmap (Clustered)")
    plt.savefig(
        f"{fig_dir}/web_visit_description_cluster_cosine_distances.png", dpi=300
    )

    # save web description, cluster label, and cluster emb
    # create a dataframe with description, cluster label, and description relevance to app use (based on manual review of descriptions in each cluster)
    cluster_df = pd.DataFrame(
        {
            "description": emb_df.index,
        }
    )
    cluster_df.set_index("description", inplace=True)
    cluster_df["cluster_label"] = cluster_labels

    # add semi manual app-relevant binary labels based on manual review of descriptions in each cluster
    # NOTE: app_relevant_clusters chosen via manual semantic inspection
    app_relevant_clusters = [7, 8, 9, 10, 11]
    app_relevance = [
        1 if label in app_relevant_clusters else 0
        for label in cluster_df["cluster_label"]
    ]
    cluster_df["app_relevance"] = app_relevance

    # get file output name from config
    config_dict = pd.read_json(CONFIG_PATH, typ="series").to_dict()
    output_file_names = config_dict["output_file_names"]["web_visit_clusters"]
    cluster_df.to_csv(f"{data_output_dir}/{output_file_names}", index=True)


def plot_claims_distribution(claims_df, fig_dir, count_patients=True):
    """
    Plots ICD code distribution by claim volume or patient prevalence.

    Args:
        claims_df (pd.DataFrame): Data containing 'icd_code' and 'member_id'.
        fig_dir (str): Directory to save the output plot.
        count_patients (bool): If True, counts unique patients per code.
            If False, counts total claim occurrences.
    """

    df = claims_df.copy()
    claims_icd_codes = df["icd_code"].unique()

    # print the list of claims:
    print(f"Unique ICD Codes in Claims: {claims_icd_codes}")

    if not count_patients:
        # number of claims per icd code, sorted descending
        claims_counts = df["icd_code"].value_counts().sort_values(ascending=False)
        counts = claims_counts
        fig_name = f"{fig_dir}/claims_distribution_icd_codes.png"
        title_str = "claims prevalence by ICD Code"
        y_label_str = "Number of Claims"

    else:
        # plot the number of subjects per icd code (removing duplicate claims per subject)
        # Calculate unique subjects per ICD code
        # We reset the index to make the member_id accessible for counting
        subject_counts = (
            df.reset_index()
            .groupby("icd_code")["member_id"]
            .nunique()
            .sort_values(ascending=False)
        )
        counts = subject_counts
        fig_name = f"{fig_dir}/unique_patient_distribution_icd_codes.png"
        title_str = "Patient Prevalence by ICD Code"
        y_label_str = "Number of Unique Patients"

    plt.figure(figsize=(12, 6))
    plt.bar(counts.index.astype(str), counts.values)
    # use log scale
    plt.ylim(counts.values.min() * 0.95, counts.values.max() * 1.05)
    plt.yscale("log")
    plt.title(title_str, fontsize=14, pad=15)
    plt.ylabel(f"{y_label_str} (log scale)", fontsize=12)
    plt.xlabel("ICD Code", fontsize=12)
    plt.xticks(rotation=45)

    # save figure
    plt.savefig(fig_name, dpi=300)


def plot_churn_by_claims_outreach(churn_df, claim_df, fig_dir):
    """
    Plots churn probability by ICD claim status, faceted by outreach.

    Args:
        churn_df (pd.DataFrame): Dataframe with 'churn' and 'outreach' columns.
        claim_df (pd.DataFrame): Dataframe with 'icd_code' column.
        fig_dir (str): Directory to save the visualization.
    """
    # Create boolean matrix for ICD claims per member
    member_icd_matrix = (
        claim_df.reset_index()
        .groupby(["member_id", "icd_code"])
        .size()
        .unstack(fill_value=0)
        > 0
    )

    # Merge with churn and outreach data
    analysis_df = (
        churn_df[["churn", "outreach"]]
        .join(member_icd_matrix, how="left")
        .fillna(False)
    )
    icd_codes = claim_df["icd_code"].unique()

    # Melt the dataframe for Seaborn plotting
    melted_df = analysis_df.melt(
        id_vars=["churn", "outreach"],
        value_vars=icd_codes,
        var_name="icd_code",
        value_name="filed_claim",
    )

    # Calculate means and counts for plotting and labeling
    plot_data = (
        melted_df.groupby(["outreach", "icd_code", "filed_claim"])
        .agg(
            churn_prob=("churn", "mean"),
            patient_count=("churn", "count"),
            churn_sem=("churn", "sem"),  # standard error of the mean for error bars
        )
        .reset_index()
    )

    # Initialize subplots: 2 rows, 1 column
    fig, axes = plt.subplots(2, 1, figsize=(18, 7), sharey=True)
    filed_claim_statuses = [False, True]

    for i, filed_claim_status in enumerate(filed_claim_statuses):
        ax = axes[i]
        subset = plot_data[plot_data["filed_claim"] == filed_claim_status]

        sns.barplot(
            data=subset,
            x="icd_code",
            y="churn_prob",
            hue="outreach",
            ax=ax,
            palette="muted",
        )

        ax.set_title(f"Filed Claim: {filed_claim_status}", fontsize=14)
        ax.set_ylabel("Churn Probability" if i == 0 else "")
        ax.set_xlabel("ICD Code")
        ax.legend(
            title="Outreach",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
        ax.set_ylim(
            plot_data["churn_prob"].min() * 0.95, plot_data["churn_prob"].max() * 1.05
        )

        # add custom error bars from churn_sem
        for i, patch in enumerate(ax.patches[: len(subset)]):
            # patches are in order by x*hue, so make sure subset matches this order
            height = patch.get_height()
            # get the corresponding SEM value
            sem = subset["churn_sem"].iloc[i]
            ax.errorbar(
                x=patch.get_x() + patch.get_width() / 2,
                y=height,
                yerr=sem,
                color="black",
                capsize=3,
            )

    plt.suptitle(
        "Churn Probability by ICD Claim Status and Outreach Intervention", fontsize=16
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{fig_dir}/churn_icd_outreach_facet.png", dpi=300)
    plt.close()

    # plot also 2X2 outreach vs claim status - I10 (hypertension) vs no I10, faceted by outreach
    icd_interest = "I10"  # example ICD code of interest
    plt.figure(figsize=(8, 6))
    subset = plot_data[plot_data["icd_code"] == icd_interest]

    # draw the barplot without error bars
    ax = sns.barplot(
        data=subset,
        x="filed_claim",
        y="churn_prob",
        hue="outreach",
        palette="muted",
        errorbar=None,  # disables automatic error bars
    )

    # add custom error bars from churn_sem
    for i, patch in enumerate(ax.patches[: len(subset)]):
        # patches are in order by x*hue, so make sure subset matches this order
        height = patch.get_height()
        # get the corresponding SEM value
        sem = subset["churn_sem"].iloc[i]
        ax.errorbar(
            x=patch.get_x() + patch.get_width() / 2,
            y=height,
            yerr=sem,
            color="black",
            capsize=3,
        )
    # change xtick font size to 14
    plt.xticks(fontsize=14)

    plt.title(
        f"Churn Probability is higher  {icd_interest} Claim Status and Outreach Intervention",
        fontsize=16,
    )
    plt.xlabel("Filed Claim Status", fontsize=14)
    plt.ylabel("Churn Probability", fontsize=14)
    plt.savefig(f"{fig_dir}/churn_icd_{icd_interest}_outreach_facet.png", dpi=300)


def plot_churn_vs_time(
    churn_df, fig_dir, time_unit="Month", separate_outreach=False, rolling_window=7
):
    """
    Plots churn probability over time with configurable granularity.

    Args:
        churn_df (pd.DataFrame): Source data containing 'signup_date', 'churn', and 'outreach'.
        fig_dir (str): Directory path to save generated figures.
        time_unit (str): Frequency of data points ('Day', 'Week', 'Month').
        separate_outreach (bool): If True, plots separate lines for outreach status.
        rolling_window (int): Window size for trend smoothing (only applied if time_unit='Day').
    """
    # Defensive copy to prevent mutation of the source dataframe
    df = churn_df.copy()

    # Map input string to pandas frequency aliases
    unit_map = {"day": "D", "week": "W", "month": "M"}
    unit = time_unit.lower()
    if unit not in unit_map:
        raise ValueError(
            f"Unsupported time_unit '{time_unit}'. Use 'Day', 'Week', or 'Month'."
        )

    # Standardize timestamps to the start of the specified period
    df["time_period"] = df["signup_date"].dt.to_period(unit_map[unit]).dt.to_timestamp()

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Define grouping hierarchy based on outreach toggle
    group_cols = ["time_period", "outreach"] if separate_outreach else ["time_period"]
    stats = df.groupby(group_cols)["churn"].mean()

    if separate_outreach:
        # Unstack to transform outreach levels into individual columns for plotting
        stats = stats.unstack()
        for col in stats.columns:
            if unit == "month":
                (line,) = ax.plot(
                    stats.index,
                    stats[col],
                    marker="o",
                    markersize=3,
                    alpha=1 if unit == "month" else 0.4,
                    label=f"Outreach {col}",
                )
            else:
                # Apply rolling average trendline for daily granularity
                rolling_series = (
                    stats[col].rolling(window=rolling_window, center=True).mean()
                )
                ax.plot(
                    stats.index,
                    rolling_series,
                    linewidth=2,
                    label=f"Outreach {col} {rolling_window}{unit} Mean",
                )
    else:
        (line,) = ax.plot(
            stats.index,
            stats.values,
            marker="o",
            color="black",
            alpha=1 if unit == "month" else 0.4,
            label="Raw Churn Rate",
        )

        if not unit == "month":
            rolling_series = stats.rolling(window=rolling_window, center=True).mean()
            ax.plot(
                stats.index,
                rolling_series,
                linewidth=3,
                color="black",
                label=f"{rolling_window}- Rolling Average",
            )

    # Configure x-axis to show monthly labels regardless of data granularity
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # Visual styling
    plt.title(f"Churn Risk Is Higher for More Recent Signups", fontsize=16, pad=15)
    plt.ylabel("Churn Probability", fontsize=14)
    plt.xlabel("Signup Month", fontsize=14)
    plt.grid(True, which="major", axis="y", linestyle="--", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.xticks(rotation=45)

    # Export visualization
    outreach_tag = "_split_outreach" if separate_outreach else ""
    filename = f"churn_prob_{unit}{outreach_tag}.png"
    plt.savefig(f"{fig_dir}/{filename}", dpi=300)
    plt.close()


if __name__ == "__main__":
    
    # Silence FutureWarnings
    print("Silencing FutureWarnings for cleaner output...")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    print("Starting data exploration...")
    main()
