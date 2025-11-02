import functions_framework
from google.cloud import bigquery
from google.cloud import firestore

# --- CLIENT INITIALIZATION ---
bq_client = bigquery.Client()
firestore_client = firestore.Client(database="misinfo-reports")

# --- CONFIGURATION ---
PROJECT_ID = "agent-builder-472216"
DATASET = "misinformation_logs"
MODEL_NAME = "semantic_kmeans_model"
BQ_TABLE = f"{PROJECT_ID}.{DATASET}.submissions"
TRENDING_COLLECTION = "trending_topics"
TIME_WINDOW_HOURS = 10      # Look back 10 hours
NUM_CLUSTERS = 2           # K-Means clusters (tunable)
TOP_CLUSTERS = 2            # Retrieve top 2 clusters only


@functions_framework.http
def calculate_trends(request):
    """
    Cloud Function:
    - Trains a K-Means model on embeddings from the last 10 hours
    - Predicts cluster assignments
    - Extracts reports from the top 2 clusters (most frequent topics)
    - Writes representative reports to Firestore
    """
    print(f"Starting trend calculation over last {TIME_WINDOW_HOURS} hours...")

    try:
        # ===============================================================
        # STEP 1: Create or replace K-Means model
        # ===============================================================
        create_model_query = f"""
        CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET}.{MODEL_NAME}`
        OPTIONS(
          model_type = 'kmeans',
          num_clusters = {NUM_CLUSTERS},
          standardize_features = TRUE,
          max_iterations = 50
        ) AS
        SELECT
          semantic_embedding
        FROM
          `{BQ_TABLE}`
        WHERE
          timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {TIME_WINDOW_HOURS} HOUR)
          AND ARRAY_LENGTH(semantic_embedding) > 0
        """

        print("Training/replacing K-Means model on recent data...")
        bq_client.query(create_model_query).result()
        print("Model training complete.")

        # ===============================================================
        # STEP 2: Predict clusters for reports in the last 10 hours
        # ===============================================================
        predict_query = f"""
        WITH predictions AS (
          SELECT
            p.centroid_id AS cluster_id,
            t.report_summary,
            t.submission_hash
          FROM
            ML.PREDICT(MODEL `{PROJECT_ID}.{DATASET}.{MODEL_NAME}`,
              (
                SELECT
                  submission_hash,
                  semantic_embedding
                FROM
                  `{BQ_TABLE}`
                WHERE
                  timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {TIME_WINDOW_HOURS} HOUR)
                  AND ARRAY_LENGTH(semantic_embedding) > 0
              )
            ) AS p
          JOIN `{BQ_TABLE}` AS t
          ON p.submission_hash = t.submission_hash
        ),
        cluster_counts AS (
          SELECT
            cluster_id,
            COUNT(*) AS topic_count
          FROM predictions
          GROUP BY cluster_id
        ),
        top_clusters AS (
          SELECT cluster_id
          FROM cluster_counts
          ORDER BY topic_count DESC
          LIMIT {TOP_CLUSTERS}
        )
        SELECT
          ANY_VALUE(t.report_summary) AS report_summary,
          ANY_VALUE(t.submission_hash) AS example_hash,
          t.cluster_id,
          COUNT(*) AS topic_count
        FROM predictions AS t
        WHERE t.cluster_id IN (SELECT cluster_id FROM top_clusters)
        GROUP BY t.cluster_id
        ORDER BY topic_count DESC
        """

        print("Running prediction and aggregation query...")
        results = list(bq_client.query(predict_query))
        print(f"Found {len(results)} trending clusters (top {TOP_CLUSTERS}).")

        # ===============================================================
        # STEP 3: Update Firestore
        # ===============================================================
        batch = firestore_client.batch()
        topics_ref = firestore_client.collection(TRENDING_COLLECTION)

        # Delete old docs
        for doc in topics_ref.stream():
            batch.delete(doc.reference)
        print("Cleared old trending topics from Firestore.")

        # Add new trending topics
        for row in results:
            new_doc_ref = topics_ref.document()
            batch.set(new_doc_ref, {
                "report_summary": row.report_summary,
                "example_hash": row.example_hash,
                "topic_count": row.topic_count,
                "cluster_id": row.cluster_id,
                "time_window_hours": TIME_WINDOW_HOURS,
                "last_updated": firestore.SERVER_TIMESTAMP,
            })

        batch.commit()
        print("Successfully wrote new trending topics to Firestore.")

        return ("OK", 200)

    except Exception as e:
        print(f"Error during trend calculation: {e}")
        return ("Internal Server Error", 500)



#############################################################
import functions_framework
from google.cloud import bigquery
from google.cloud import firestore
import pandas as pd
import numpy as np
import hdbscan
from sklearn.decomposition import PCA

# --- CLIENT INITIALIZATION ---
bq_client = bigquery.Client()
firestore_client = firestore.Client(database="misinfo-reports")

# --- CONFIGURATION ---
PROJECT_ID = "agent-builder-472216"
DATASET = "misinformation_logs"
BQ_TABLE = f"{PROJECT_ID}.{DATASET}.submissions"
TRENDING_COLLECTION = "trending_topics"
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
    - Writes representative summaries + topic counts to Firestore
    """
    print(f"Starting auto trend calculation over last {TIME_WINDOW_HOURS} hours...")

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
            return ("No data available", 200)

        print(f"Loaded {len(df)} recent reports from BigQuery.")

        # ===============================================================
        # STEP 2: Prepare embeddings for clustering
        # ===============================================================
        embeddings = np.array(df["semantic_embedding"].to_list())

        # Dynamically determine PCA components
        n_components = min(
            50,
            embeddings.shape[1],
            max(2, embeddings.shape[0] // 2)
        )

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
        topics_ref = firestore_client.collection(TRENDING_COLLECTION)

        # Delete old docs
        for doc in topics_ref.stream():
            batch.delete(doc.reference)
        print("Cleared old trending topics from Firestore.")

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

