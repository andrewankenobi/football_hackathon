import requests
import json
import os

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

os.makedirs("data/matches", exist_ok=True)
os.makedirs("data/events", exist_ok=True)
os.makedirs("data/lineups", exist_ok=True)

competitions_url = f"{BASE_URL}/competitions.json"
competitions_response = requests.get(competitions_url)
competitions_data = competitions_response.json()

competition_names = sorted(set(comp["competition_name"] for comp in competitions_data))

print("Available competitions:")
for i, name in enumerate(competition_names, 1):
    print(f"{i}. {name}")

print("\nEnter the numbers of the competitions you want to download (comma-separated):")
print("For example, to download Champions League data, enter: 3")
selected_indices = input("Your selection: ").split(',')
selected_names = [competition_names[int(i.strip()) - 1] for i in selected_indices]

selected_competitions = [comp for comp in competitions_data if comp["competition_name"] in selected_names]

for competition in selected_competitions:
    COMPETITION_ID = competition["competition_id"]
    COMPETITION_NAME = competition["competition_name"]
    
    print(f"\nAvailable seasons for {COMPETITION_NAME}:")
    seasons = [comp for comp in competitions_data if comp["competition_id"] == COMPETITION_ID]
    for i, season in enumerate(seasons, 1):
        print(f"{i}. {season['season_name']}")
    
    print("\nEnter the numbers of the seasons you want to download (comma-separated),")
    print("or press Enter to download all seasons:")
    season_input = input("Your selection: ")
    
    if season_input.strip():
        selected_season_indices = [int(i.strip()) - 1 for i in season_input.split(',')]
        selected_seasons = [seasons[i] for i in selected_season_indices]
    else:
        selected_seasons = seasons
    
    for season in selected_seasons:
        SEASON_ID = season["season_id"]
        SEASON_NAME = season["season_name"]
        
        print(f"\nDownloading data for {COMPETITION_NAME}, Season: {SEASON_NAME}")
        
        matches_url = f"{BASE_URL}/matches/{COMPETITION_ID}/{SEASON_ID}.json"
        matches_response = requests.get(matches_url)
        matches_data = matches_response.json()

        with open(f"data/matches/{COMPETITION_ID}_{SEASON_ID}.json", "w") as f:
            json.dump(matches_data, f)

        print(f"Downloaded {len(matches_data)} matches for {COMPETITION_NAME} season {SEASON_NAME}")

        for match in matches_data:
            match_id = match["match_id"]
            
            events_url = f"{BASE_URL}/events/{match_id}.json"
            events_response = requests.get(events_url)
            events_data = events_response.json()
            
            with open(f"data/events/{match_id}.json", "w") as f:
                json.dump(events_data, f)
            
            lineups_url = f"{BASE_URL}/lineups/{match_id}.json"
            lineups_response = requests.get(lineups_url)
            lineups_data = lineups_response.json()
            
            with open(f"data/lineups/{match_id}.json", "w") as f:
                json.dump(lineups_data, f)
            
            print(f"Downloaded events and lineups data for match {match_id}")

print("\nDownload complete!")