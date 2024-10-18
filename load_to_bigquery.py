import os
import json
from google.cloud import bigquery
import argparse

def convert_to_newline_delimited_json(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'a') as outfile:
        data = json.load(infile)
        for item in data:
            json.dump(item, outfile)
            outfile.write('\n')

def drop_all_tables(project_id, dataset_id):
    client = bigquery.Client(project=project_id)
    dataset_ref = client.dataset(dataset_id)
    tables = list(client.list_tables(dataset_ref))
    
    for table in tables:
        client.delete_table(table)
        print(f"Dropped table {table.table_id}")

def get_schema(table_id):
    if table_id == "matches":
        return [
            bigquery.SchemaField("match_id", "INTEGER"),
            bigquery.SchemaField("match_date", "DATE"),
            bigquery.SchemaField("kick_off", "TIME"),
            bigquery.SchemaField("competition", "RECORD", fields=[
                bigquery.SchemaField("competition_id", "INTEGER"),
                bigquery.SchemaField("country_name", "STRING"),
                bigquery.SchemaField("competition_name", "STRING")
            ]),
            bigquery.SchemaField("season", "RECORD", fields=[
                bigquery.SchemaField("season_id", "INTEGER"),
                bigquery.SchemaField("season_name", "STRING")
            ]),
            bigquery.SchemaField("home_team", "RECORD", fields=[
                bigquery.SchemaField("home_team_id", "INTEGER"),
                bigquery.SchemaField("home_team_name", "STRING"),
                bigquery.SchemaField("home_team_gender", "STRING"),
                bigquery.SchemaField("home_team_group", "STRING"),
                bigquery.SchemaField("country", "RECORD", fields=[
                    bigquery.SchemaField("id", "INTEGER"),
                    bigquery.SchemaField("name", "STRING")
                ])
            ]),
            bigquery.SchemaField("away_team", "RECORD", fields=[
                bigquery.SchemaField("away_team_id", "INTEGER"),
                bigquery.SchemaField("away_team_name", "STRING"),
                bigquery.SchemaField("away_team_gender", "STRING"),
                bigquery.SchemaField("away_team_group", "STRING"),
                bigquery.SchemaField("country", "RECORD", fields=[
                    bigquery.SchemaField("id", "INTEGER"),
                    bigquery.SchemaField("name", "STRING")
                ])
            ]),
            bigquery.SchemaField("home_score", "INTEGER"),
            bigquery.SchemaField("away_score", "INTEGER"),
            bigquery.SchemaField("match_status", "STRING"),
            bigquery.SchemaField("match_status_360", "STRING"),
            bigquery.SchemaField("last_updated", "TIMESTAMP"),
            bigquery.SchemaField("last_updated_360", "TIMESTAMP"),
            bigquery.SchemaField("match_week", "INTEGER"),
            bigquery.SchemaField("competition_stage", "RECORD", fields=[
                bigquery.SchemaField("id", "INTEGER"),
                bigquery.SchemaField("name", "STRING")
            ]),
            bigquery.SchemaField("stadium", "RECORD", fields=[
                bigquery.SchemaField("id", "INTEGER"),
                bigquery.SchemaField("name", "STRING"),
                bigquery.SchemaField("country", "RECORD", fields=[
                    bigquery.SchemaField("id", "INTEGER"),
                    bigquery.SchemaField("name", "STRING")
                ])
            ]),
            bigquery.SchemaField("referee", "RECORD", fields=[
                bigquery.SchemaField("id", "INTEGER"),
                bigquery.SchemaField("name", "STRING"),
                bigquery.SchemaField("country", "RECORD", fields=[
                    bigquery.SchemaField("id", "INTEGER"),
                    bigquery.SchemaField("name", "STRING")
                ])
            ]),
            bigquery.SchemaField("metadata", "RECORD", fields=[
                bigquery.SchemaField("data_version", "STRING"),
                bigquery.SchemaField("shot_fidelity_version", "STRING"),
                bigquery.SchemaField("xy_fidelity_version", "STRING")
            ])
        ]
    elif table_id == "lineups":
        return [
            bigquery.SchemaField("match_id", "INTEGER"),
            bigquery.SchemaField("team_id", "INTEGER"),
            bigquery.SchemaField("team_name", "STRING"),
            bigquery.SchemaField("lineup", "RECORD", mode="REPEATED", fields=[
                bigquery.SchemaField("player_id", "INTEGER"),
                bigquery.SchemaField("player_name", "STRING"),
                bigquery.SchemaField("player_nickname", "STRING"),
                bigquery.SchemaField("jersey_number", "INTEGER"),
                bigquery.SchemaField("country", "RECORD", fields=[
                    bigquery.SchemaField("id", "INTEGER"),
                    bigquery.SchemaField("name", "STRING")
                ])
            ])
        ]
    elif table_id == "events":
        return [
            bigquery.SchemaField("id", "STRING"),
            bigquery.SchemaField("index", "INTEGER"),
            bigquery.SchemaField("period", "INTEGER"),
            bigquery.SchemaField("timestamp", "STRING"),
            bigquery.SchemaField("minute", "INTEGER"),
            bigquery.SchemaField("second", "INTEGER"),
            bigquery.SchemaField("type", "RECORD", fields=[
                bigquery.SchemaField("id", "INTEGER"),
                bigquery.SchemaField("name", "STRING")
            ]),
            bigquery.SchemaField("possession", "INTEGER"),
            bigquery.SchemaField("possession_team", "RECORD", fields=[
                bigquery.SchemaField("id", "INTEGER"),
                bigquery.SchemaField("name", "STRING")
            ]),
            bigquery.SchemaField("play_pattern", "RECORD", fields=[
                bigquery.SchemaField("id", "INTEGER"),
                bigquery.SchemaField("name", "STRING")
            ]),
            bigquery.SchemaField("team", "RECORD", fields=[
                bigquery.SchemaField("id", "INTEGER"),
                bigquery.SchemaField("name", "STRING")
            ]),
            bigquery.SchemaField("player", "RECORD", fields=[
                bigquery.SchemaField("id", "INTEGER"),
                bigquery.SchemaField("name", "STRING")
            ]),
            bigquery.SchemaField("position", "RECORD", fields=[
                bigquery.SchemaField("id", "INTEGER"),
                bigquery.SchemaField("name", "STRING")
            ]),
            bigquery.SchemaField("location", "FLOAT", mode="REPEATED"),
            bigquery.SchemaField("duration", "FLOAT"),
            bigquery.SchemaField("related_events", "STRING", mode="REPEATED"),
            bigquery.SchemaField("under_pressure", "BOOLEAN"),
            bigquery.SchemaField("off_camera", "BOOLEAN"),
            bigquery.SchemaField("out", "BOOLEAN"),
            bigquery.SchemaField("shot", "RECORD", fields=[
                bigquery.SchemaField("outcome", "RECORD", fields=[
                    bigquery.SchemaField("id", "INTEGER"),
                    bigquery.SchemaField("name", "STRING")
                ]),
                bigquery.SchemaField("technique", "RECORD", fields=[
                    bigquery.SchemaField("id", "INTEGER"),
                    bigquery.SchemaField("name", "STRING")
                ]),
                bigquery.SchemaField("body_part", "RECORD", fields=[
                    bigquery.SchemaField("id", "INTEGER"),
                    bigquery.SchemaField("name", "STRING")
                ]),
                bigquery.SchemaField("type", "RECORD", fields=[
                    bigquery.SchemaField("id", "INTEGER"),
                    bigquery.SchemaField("name", "STRING")
                ]),
                bigquery.SchemaField("statsbomb_xg", "FLOAT"),
                bigquery.SchemaField("end_location", "FLOAT", mode="REPEATED")
            ])
        ]
    return []

def load_json_to_bigquery(project_id, dataset_id, table_id, file_paths):
    client = bigquery.Client(project=project_id)
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)

    schema = get_schema(table_id)
    
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
    job_config.schema = schema
    job_config.ignore_unknown_values = True

    temp_file = f"{table_id}.ndjson"
    for file_path in file_paths:
        convert_to_newline_delimited_json(file_path, temp_file)

    with open(temp_file, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_ref, job_config=job_config)
    
    job.result()

    os.remove(temp_file)

    print(f"Loaded data into {dataset_id}:{table_id}")

def main(project_id, dataset_id):
    drop_all_tables(project_id, dataset_id)

    data_types = {
        "matches": "data/matches",
        "events": "data/events",
        "lineups": "data/lineups"
    }

    for table_id, directory in data_types.items():
        file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json")]
        load_json_to_bigquery(project_id, dataset_id, table_id, file_paths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load StatsBomb data into BigQuery")
    parser.add_argument("project_id", help="Google Cloud project ID")
    parser.add_argument("dataset_id", help="BigQuery dataset ID")
    args = parser.parse_args()

    main(args.project_id, args.dataset_id)