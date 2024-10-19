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
   - Create a detailed view of player statistics
   - Analyze player shots and their characteristics
   - Cluster players based on their performance metrics
   - Predict player performance scores

3. **Match Analysis**:
   - Generate comprehensive match statistics
   - Predict match outcomes based on in-game events

4. **Expected Goals (xG) Analysis**:
   - Build and test an xG prediction model
   - Analyze the factors that contribute most to xG

5. **Goal Prediction**:
   - Create a model to predict whether a shot will result in a goal
   - Analyze the most important features for goal prediction

6. **Team Performance**:
   - Predict team performance scores
   - Analyze factors contributing to team success

7. **Advanced Analytics**:
   - Use the Vertex AI connection to leverage Gemini Pro for natural language processing tasks, such as generating stadium locations

To run these experiments:

1. Open the BigQuery console in your GCP project.
2. Copy and paste queries from the `sql_queries.sql` file into the BigQuery query editor.
3. Run the queries to create views, train models, and analyze the data.
4. Experiment with different parameters, features, or model types to improve predictions.

Feel free to modify the queries or create new ones to explore additional aspects of the data that interest you.

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
