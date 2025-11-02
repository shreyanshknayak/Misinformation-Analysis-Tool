import functions_framework
from google.cloud import bigquery
from google.cloud import firestore
import pandas as pd
import numpy as np
import hdbscan
from sklearn.decomposition import PCA
import os # <-- Import os

# --- CLIENT INITIALIZATION ---
bq_client = bigquery.Client()
firestore_client = firestore.Client(database="misinfo-reports")

# --- CONFIGURATION ---
PROJECT_ID = "agent-builder-472216"
DATASET = "misinformation_logs"
BQ_TABLE = f"{PROJECT_ID}.{DATASET}.submissions"

# --- MODIFIED: Firestore Collection Path ---
# This now points to the public path all users can read, taken from your previous version.
# Note: This line is Python, so I've adapted the JS-style check
app_id_var = os.environ.get('__app_id', 'default-app-id') 
TRENDING_COLLECTION_PATH = f"artifacts/{app_id_var}/public/data/trending_topics"
# --- End of Modification ---

TIME_WINDOW_HOURS = 48
MIN_CLUSTER_SIZE = 2          # minimum size to treat as a valid topic
TOP_CLUSTERS = 3              # how many clusters to store to Firestore


@functions_framework.http
def calculate_trends(request):
    """
    Cloud Function:
    - Loads recent embeddings from BigQuery
    - Performs automatic clustering using HDBSCAN
    - Aggregates and identifies top topic clusters
    - Writes representative summaries + topic counts to Firestore's public path
    """
    print(f"Starting auto trend calculation over last {TIME_WINDOW_HOURS} hours...")
    print(f"Saving results to: {TRENDING_COLLECTION_PATH}") # <-- Use correct variable

    try:
        # ===============================================================
        # STEP 1: Load recent embeddings from BigQuery
        # ===============================================================
        query = f"""
        SELECT
          submission_hash,
          report_summary,
          semantic_embedding
        FROM `{BQ_TABLE}`
        WHERE
          timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {TIME_WINDOW_HOURS} HOUR)
          AND ARRAY_LENGTH(semantic_embedding) > 0
        """

        df = bq_client.query(query).to_dataframe()
        if df.empty:
            print("No recent submissions found.")
            # Clear any old trends if no new data is found
            batch = firestore_client.batch()
            topics_ref = firestore_client.collection(TRENDING_COLLECTION_PATH) # <-- Use correct variable
            for doc in topics_ref.stream():
                batch.delete(doc.reference)
            batch.commit()
            print("Cleared old trending topics as no new data was found.")
            return ("No data available", 200)

        print(f"Loaded {len(df)} recent reports from BigQuery.")

        # ===============================================================
        # STEP 2: Prepare embeddings for clustering
        # ===============================================================
        embeddings = np.array(df["semantic_embedding"].to_list())

        # Dynamically determine PCA components
        # Ensure n_components is not larger than number of samples
        n_samples = embeddings.shape[0]
        n_features = embeddings.shape[1]
        
        # Set n_components: min of (50, features, and half of samples (but at least 2))
        n_components_pca = min(50, n_features, max(2, n_samples // 2))
        
        # PCA n_components cannot be larger than n_features or n_samples
        n_components = min(n_components_pca, n_samples, n_features)

        if n_samples <= 1:
             print("Not enough samples to perform PCA or clustering. Skipping.")
             return ("Not enough samples for clustering", 200)

        print(f"Applying PCA with {n_components} components (dynamic cap).")

        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
        print(f"Reduced embeddings to {reduced_embeddings.shape[1]} dimensions.")

        # ===============================================================
        # STEP 3: Run HDBSCAN for automatic clustering
        # ===============================================================
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=1,
            cluster_selection_epsilon=0.05,
            metric="euclidean"
        )
        df["cluster_id"] = clusterer.fit_predict(reduced_embeddings)

        num_clusters = len(set(df["cluster_id"])) - (1 if -1 in df["cluster_id"] else 0)
        print(f"Detected {num_clusters} clusters (excluding outliers).")

        # Filter valid clusters only
        df_valid = df[df.cluster_id != -1]
        if df_valid.empty:
            print("No valid clusters found after filtering.")
            # Clear any old trends if no new clusters are found
            batch = firestore_client.batch()
            topics_ref = firestore_client.collection(TRENDING_COLLECTION_PATH) # <-- Use correct variable
            for doc in topics_ref.stream():
                batch.delete(doc.reference)
            batch.commit()
            print("Cleared old trending topics as no valid clusters were found.")
            return ("No valid clusters found", 200)

        # ===============================================================
        # STEP 4: Aggregate topic clusters
        # ===============================================================
        topic_summary = (
            df_valid.groupby("cluster_id")
            .agg(
                topic_count=("submission_hash", "count"),
                example_report=("report_summary", "first"),
                example_hash=("submission_hash", "first"),
            )
            .sort_values("topic_count", ascending=False)
            .reset_index()
        )

        top_topics = topic_summary.head(TOP_CLUSTERS)
        print(f"Prepared top {len(top_topics)} topic summaries for Firestore upload.")

        # ===============================================================
        # STEP 5: Update Firestore with trending topics
        # ===============================================================
        batch = firestore_client.batch()
        topics_ref = firestore_client.collection(TRENDING_COLLECTION_PATH) # <-- Use correct variable

        # Delete old docs
        for doc in topics_ref.stream():
            batch.delete(doc.reference)
        print(f"Cleared old trending topics from {TRENDING_COLLECTION_PATH}.") # <-- Use correct variable

        # Add new trending topics
        for _, row in top_topics.iterrows():
            new_doc_ref = topics_ref.document()
            batch.set(new_doc_ref, {
                "report_summary": row.example_report,
                "example_hash": row.example_hash,
                "topic_count": int(row.topic_count),
                "cluster_id": int(row.cluster_id),
                "time_window_hours": TIME_WINDOW_HOURS,
                "num_clusters_detected": num_clusters,
                "last_updated": firestore.SERVER_TIMESTAMP,
            })

        batch.commit()
        print("Successfully wrote new trending topics to Firestore.")

        return ("OK", 200)

    except Exception as e:
        print(f"Error during trend calculation: {e}")
        return ("Internal Server Error", 500)

