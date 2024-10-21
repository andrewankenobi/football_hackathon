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

1. **Data Exploration**:
   - Examine the schema of the StatsBomb data
   - Analyze goals by body part for each player and team
   - Identify top scorers in the dataset
   - Calculate team possession percentages

2. **Player Analysis**:
   - Create a detailed view of player statistics (`vw_player_stats`)
   - Analyze player shots and their characteristics (`vw_player_shots`)
   - Cluster players based on their performance metrics
   - Predict player performance scores

3. **Match Analysis**:
   - Generate comprehensive match statistics (`vw_match_stats`)
   - Predict match outcomes based on in-game events

4. **Expected Goals (xG) Analysis**:
   - Build and test an xG prediction model
   - Analyze the factors that contribute most to xG

5. **Goal Prediction**:
   - Create a model to predict whether a shot will result in a goal
   - Analyze the most important features for goal prediction

6. **Team Performance**:
   - Analyze team possession patterns
   - Evaluate team performance across various metrics

7. **Advanced Analytics**:
   - Use the Vertex AI connection to leverage Gemini Pro for natural language processing tasks, such as generating stadium locations

## Advanced Analytics and Machine Learning Models

The `sql_queries.sql` file includes several sophisticated machine learning models:

1. **Player Clustering Model**: Uses K-means clustering to group players based on their performance statistics. The model includes hyperparameter tuning to optimize the number of clusters.

2. **Expected Goals (xG) Prediction Model**: A linear regression model that predicts the probability of a shot resulting in a goal based on various factors.

3. **Goal Prediction Model**: A logistic regression model that predicts whether a shot will result in a goal.

4. **Match Outcome Prediction Model**: A logistic regression model that predicts the outcome of a match based on in-game statistics.

These models can be created and tested using the provided SQL queries. You can experiment with different features, model parameters, or even try different model types to improve predictions.

## Vertex AI Integration

The project leverages Google Cloud's Vertex AI capabilities, particularly the Gemini Pro model, for advanced natural language processing tasks. For example, the SQL queries demonstrate how to use Gemini Pro to generate latitude and longitude coordinates for stadiums based on their names.

## Data Exploration Views

Several views are created to facilitate data exploration and analysis:

- `vw_goals_by_body_part`: Analyzes goals scored by different body parts for each player and team.
- `vw_top_scorers`: Ranks players by the number of goals scored.
- `vw_team_possession`: Calculates average possession percentages for each team.
- `vw_player_stats`: Provides comprehensive statistics for each player.
- `vw_player_shots`: Detailed analysis of shot characteristics.
- `vw_match_stats`: Comprehensive match statistics and derived metrics.

These views can be used as a starting point for further analysis or as input for machine learning models.

## Project Structure

- `data/`: Contains pre-downloaded Premier League data
- `doc/`: Documentation files, including StatsBomb event specifications
- `sql_queries.sql`: Sample SQL queries for data analysis
- `load_to_bigquery.py`: Script to load data into BigQuery
- `requirements.txt`: List of Python dependencies
- `statsbomb_schema.json`: JSON schema for StatsBomb data

## Resources

- [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- [Google Cloud Documentation](https://cloud.google.com/docs)
- [BigQuery ML Documentation](https://cloud.google.com/bigquery-ml/docs)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)

## Support

If you encounter any issues or have questions, please open an issue in this repository.

Happy hacking!

## Player Similarity Analysis

The project includes advanced player similarity analysis using embeddings:

1. **Player Statistics View**: We create a view `vw_player_stats` that aggregates various performance metrics for each player.
2. **Player Statistics Table**: We create a table `player_stats_table` from the view, adding a temporary label for classification purposes.
3. **Player Embedding Model**: We create a DNN classifier model to generate meaningful embeddings for each player based on their performance statistics.
4. **Player Embeddings**: Using the DNN model, we generate embeddings for each player based on the feature importance weights.
5. **Vector Index**: A vector index is created on these embeddings to enable efficient similarity search.
6. **Similarity Search**: A custom function `find_similar_players` allows you to find the top 5 players most similar to any given player.

This feature can be used for player scouting, tactical analysis, or understanding player styles across different teams and leagues. The embeddings are based on a wide range of player actions including passes, shots, ball recoveries, duels, interceptions, goals, pressures, dribbles, fouls, carries, clearances, and blocks.

Example usage:

```
SELECT * FROM `statsbomb.find_similar_players`('Lionel Andrés Messi Cuccittini');

```