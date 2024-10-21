# StatsBomb Data Analysis Hackathon

![Football Hackathon Logo](logo.jpeg)

This repository contains the necessary code and instructions to run a data analysis hackathon using StatsBomb open data and Google Cloud Platform (GCP) services.

## Prerequisites

- Python 3.7+
- Google Cloud SDK
- A Google Cloud Platform account with billing enabled

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/andrewankenobi/football_hackathon.git
   cd football_hackathon
   ```

2. Install the required Python dependencies:
   ```
   pip install -r requirements.txt
   ```

   Make sure the following packages are included in your requirements.txt file:
   ```
   pandas
   numpy
   scikit-learn
   plotly
   dash
   google-cloud-bigquery
   ```

3. Install the Google Cloud SDK by following the instructions [here](https://cloud.google.com/sdk/docs/install).

4. Authenticate with your Google Cloud account:
   ```
   gcloud auth login
   ```

5. Set your GCP project:
   ```
   gcloud config set project <your-project-id>
   ```

## Data Download

1. Run the data download script:
   ```
   python download_statsbomb_data.py
   ```
   This script will download JSON files from the StatsBomb open data repository. You can choose which competitions and seasons to download.

   Alternatively, you can use the pre-downloaded Premier League data in the `data` folder.

## Google Cloud Setup

1. Create a new project in the Google Cloud Console or use an existing one.

2. Enable the following APIs in your GCP project:
   - BigQuery API
   - Vertex AI API

3. Create a BigQuery dataset in your project:
   ```
   bq mk --dataset <your-project-id>:<dataset-name>
   ```

## Data Loading

1. Load the downloaded data into BigQuery:
   ```
   python load_to_bigquery.py <your-project-id> <dataset-name>
   ```

## Vertex AI Connection Setup

1. Create a connection to Vertex AI in BigQuery. Make sure it's in the same location as your dataset:
   ```
   bq mk --connection --display_name="Vertex AI Connection" --connection_type=CLOUD_RESOURCE --project_id=<your-project-id> --location=<your-location> vertex-ai-connection
   ```

2. Grant the Vertex AI User role to the service account associated with the connection:
   ```
   gcloud projects add-iam-policy-binding <your-project-id> --member="serviceAccount:<service-account-email>" --role="roles/aiplatform.user"
   ```

   Replace `<service-account-email>` with the email of the service account created for the connection.

## Running Queries and Experiments

The `sql_queries.sql` file contains a comprehensive set of SQL queries, views, and machine learning models that you can use to analyze the StatsBomb data. Here are some of the key use cases and experiments you can run:

1. **Schema Evaluation**:
   - Examine the schema of the StatsBomb data using the `vw_schema_info` view

2. **Data Exploration Views**:
   - Analyze goals by body part for each player and team (`vw_goals_by_body_part`)
   - Identify top scorers in the dataset (`vw_top_scorers`)
   - Calculate team possession percentages (`vw_team_possession`)

3. **Player Analysis**:
   - Create a detailed view of player statistics (`vw_player_stats`)
   - Analyze player shots and their characteristics (`vw_player_shots`)

4. **Match Analysis**:
   - Generate comprehensive match statistics (`vw_match_stats`)

5. **Advanced Analytics and Machine Learning Models**:
   - Cluster players based on their performance metrics (`player_clusters` model)
   - Predict expected goals (xG) for shots (`xg_prediction` model)
   - Predict whether a shot will result in a goal (`goal_prediction_model`)
   - Predict match outcomes based on in-game events (`match_outcome_prediction` model)

6. **Player Embeddings and Similarity Search**:
   - Generate player embeddings based on their statistics
   - Perform similarity searches to find players with similar characteristics

## Advanced Analytics and Machine Learning Models

The `sql_queries.sql` file includes several sophisticated machine learning models:

1. **Player Clustering Model**: Uses K-means clustering to group players based on their performance statistics. The model includes hyperparameter tuning to optimize the number of clusters (3 to 6).

2. **Expected Goals (xG) Prediction Model**: A linear regression model that predicts the probability of a shot resulting in a goal based on various factors such as shot type, body part, technique, and location.

3. **Goal Prediction Model**: A logistic regression model that predicts whether a shot will result in a goal based on features from the `vw_player_shots` view.

4. **Match Outcome Prediction Model**: A logistic regression model that predicts the outcome of a match (home win, away win, or draw) based on in-game statistics from the `vw_match_stats` view.

5. **Player Embedding Model**: A PCA model that generates embeddings for players based on their performance statistics, allowing for efficient similarity searches.

These models can be created and tested using the provided SQL queries. You can experiment with different features, model parameters, or even try different model types to improve predictions.

## Vertex AI Integration

The project leverages Google Cloud's Vertex AI capabilities, particularly the Gemini Pro model, for advanced natural language processing tasks. The SQL queries demonstrate how to create a connection to the Gemini Pro model, which can be used for various NLP tasks.

## Data Exploration Views

Several views are created to facilitate data exploration and analysis:

- `vw_schema_info`: Provides information about the schema of all tables in the dataset.
- `vw_goals_by_body_part`: Analyzes goals scored by different body parts for each player and team.
- `vw_top_scorers`: Ranks players by the number of goals scored.
- `vw_team_possession`: Calculates average possession percentages for each team.
- `vw_player_stats`: Provides comprehensive statistics for each player.
- `vw_player_shots`: Detailed analysis of shot characteristics.
- `vw_match_stats`: Comprehensive match statistics and derived metrics.

These views can be used as a starting point for further analysis or as input for machine learning models.

## Player Similarity Analysis

The project includes advanced player similarity analysis using embeddings:

1. **Player Statistics View**: We create a view `vw_player_stats_for_embeddings` that aggregates various performance metrics for each player.
2. **Player Embedding Model**: We create a PCA model to generate meaningful embeddings for each player based on their performance statistics.
3. **Player Embeddings**: Using the PCA model, we generate embeddings for each player and store them in the `player_embeddings` table.
4. **Similarity Search**: You can use vector search queries to find players similar to a given player based on their embeddings.

This feature can be used for player scouting, tactical analysis, or understanding player styles across different teams and leagues. The embeddings are based on a wide range of player actions including passes, shots, ball recoveries, duels, interceptions, goals, pressures, dribbles, fouls, carries, clearances, and blocks.

Example usage (commented out in the SQL file):

```sql
SELECT base.* FROM VECTOR_SEARCH(TABLE `statsbomb.player_embeddings`, 'ml_generate_embedding_result', (SELECT ml_generate_embedding_result FROM `statsbomb.player_embeddings` WHERE player_name = 'Petr Čech'), top_k => 10, distance_type => 'COSINE') WHERE base.player_name != 'Petr Čech';
```

This query would find the top 10 players most similar to Petr Čech based on their performance statistics.

## Running t-SNE Visualization Locally

After completing the setup and data loading steps, you can run the t-SNE visualization locally:

1. Ensure you have all the required dependencies installed (step 2 in the Setup section).

2. Run the t-SNE visualization script:
   ```
   python tsne.py
   ```

3. Open a web browser and navigate to the URL displayed in the console (typically http://127.0.0.1:8050/).

This will launch a local Dash server and open the interactive t-SNE visualization of player embeddings in your default web browser.

## Project Structure

- `data/`: Contains pre-downloaded Premier League data
- `doc/`: Documentation files, including StatsBomb event specifications
- `sql_queries.sql`: Sample SQL queries for data analysis
- `load_to_bigquery.py`: Script to load data into BigQuery
- `requirements.txt`: List of Python dependencies
- `statsbomb_schema.json`: JSON schema for StatsBomb data
- `tsne-players.py`: Script for t-SNE visualization of player embeddings

## Resources

- [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- [Google Cloud Documentation](https://cloud.google.com/docs)
- [BigQuery ML Documentation](https://cloud.google.com/bigquery-ml/docs)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)

## Support

If you encounter any issues or have questions, please open an issue in this repository.

Happy hacking!
