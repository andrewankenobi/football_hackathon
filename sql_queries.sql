
-- evaluate schema

SELECT
  *
FROM
  `awesome-advice-420021.statsbomb.INFORMATION_SCHEMA.COLUMNS`
ORDER BY
  table_name,
  ordinal_position;


-- Create an external connection in BigQuery to a vertex ai model. 
-- The connection is created in the same project and region as the BigQuery dataset. 
-- connection id projects/awesome-advice-420021/locations/us-central1/connections/biglake


CREATE OR REPLACE MODEL `awesome-advice-420021.statsbomb.gemini`
REMOTE WITH CONNECTION `projects/awesome-advice-420021/locations/us/connections/biglake`
OPTIONS (endpoint = 'gemini-pro');


-- Example query to use the model adding latitude and longitude to the stadium name. 
WITH model_output AS (
  SELECT
    stadium_name, 
    ml_generate_text_llm_result AS location
  FROM
    ML.GENERATE_TEXT( MODEL `awesome-advice-420021.statsbomb.gemini`,
      (
      SELECT
        DISTINCT matches.stadium.name AS stadium_name,
        CONCAT('give me latitude and longitude for this famous stadium. Respond with the only information, to be used straight away in Looker studio. Format the answer "latitude, longitude" - example "53.4336, -2.9611". Here is the stadium name: ', matches.stadium.name) AS prompt
      FROM
        `awesome-advice-420021.statsbomb.matches` AS matches ),
      STRUCT( 0.1 AS temperature,
        TRUE AS flatten_json_output ) )
)

SELECT
  stadium_name,
  location,
  CAST(SPLIT(location, ', ')[SAFE_OFFSET(0)] AS FLOAT64) AS latitude,
  CAST(SPLIT(location, ', ')[SAFE_OFFSET(1)] AS FLOAT64) AS longitude
FROM
  model_output


-- Goals by body part

SELECT 
  team.name AS team, 
  player.name AS player, 
  SUM(CASE WHEN shot.body_part.name = 'Left Foot' THEN 1 ELSE 0 END) AS left_foot_goals,
  SUM(CASE WHEN shot.body_part.name = 'Right Foot' THEN 1 ELSE 0 END) AS right_foot_goals,
  SUM(CASE WHEN shot.body_part.name = 'Head' THEN 1 ELSE 0 END) AS head_goals,
  SUM(CASE WHEN shot.body_part.name = 'Other' THEN 1 ELSE 0 END) AS other_goals,
  count(shot.body_part.name) as total_goals
FROM 
  `awesome-advice-420021.statsbomb.events`
WHERE 
  type.name = 'Shot' 
  AND shot.outcome.name = 'Goal'
GROUP BY 
  1,2
ORDER BY 
  7 desc 



-- top scorers
SELECT player.name AS player_name, COUNT(*) AS goals
FROM `awesome-advice-420021.statsbomb.events`
WHERE shot.outcome.name = "Goal"
GROUP BY player_name
ORDER BY goals DESC



--possession percentage by team
WITH possession_time AS (
  SELECT
    m.match_id,
    e.team.name as team_name,
    SUM(e.duration) as total_duration
  FROM
    `awesome-advice-420021.statsbomb.events` e
  JOIN
    `awesome-advice-420021.statsbomb.matches` m ON e.match_id = m.match_id
  WHERE
    e.type.name = 'Pass'
    AND e.team.name IS NOT NULL
  GROUP BY
    m.match_id, e.team.name
),
match_totals AS (   
  SELECT
    match_id,
    SUM(total_duration) as match_duration
  FROM
    possession_time
  GROUP BY
    match_id
)
SELECT
  pt.team_name,
  AVG(pt.total_duration / mt.match_duration) * 100 as avg_possession_percentage
FROM
  possession_time pt
JOIN 
  match_totals mt ON pt.match_id = mt.match_id
GROUP BY
  pt.team_name
            ORDER BY
  avg_possession_percentage DESC    

-- shots on target by team
SELECT team.name AS team_name, COUNT(*) AS shots_on_target
FROM `awesome-advice-420021.statsbomb.events`
WHERE type.name = 'Shot' AND shot.outcome.name = 'On Target'
GROUP BY team_name
ORDER BY shots_on_target DESC

-- shots on target percentage by team
WITH shots_on_target AS (
  SELECT
    m.match_id,
    e.team.name as team_name,
    COUNT(*) as total_shots,
    SUM(CASE WHEN e.shot.outcome.name IN ('Saved', 'Goal', 'Post', 'Saved to Post') THEN 1 ELSE 0 END) as shots_on_target
  FROM
    `awesome-advice-420021.statsbomb.events` e
  JOIN
    `awesome-advice-420021.statsbomb.matches` m ON e.match_id = m.match_id
  WHERE
    e.type.name = 'Shot'
  GROUP BY
    m.match_id, e.team.name
),
match_totals AS (
  SELECT
    match_id,
    SUM(total_shots) as match_shots
  FROM
    shots_on_target
  GROUP BY
    match_id
)
SELECT
  st.team_name,
  AVG(st.total_shots / mt.match_shots) * 100 as avg_shots_on_target_percentage
FROM
  shots_on_target st
JOIN
  match_totals mt ON st.match_id = mt.match_id
GROUP BY
  st.team_name
ORDER BY
  avg_shots_on_target_percentage DESC


  -- average xg by team
SELECT team.name AS team_name, AVG(shot.statsbomb_xg) AS average_xg
FROM `awesome-advice-420021.statsbomb.events`
WHERE type.name = 'Shot'
and  shot.statsbomb_xg IS NOT NULL
GROUP BY team_name
ORDER BY average_xg DESC    


--shots under pressure by player
SELECT player.name AS player_name, COUNT(*) AS shots_under_pressure
FROM `awesome-advice-420021.statsbomb.events`
WHERE shot.statsbomb_xg IS NOT NULL AND under_pressure = TRUE
GROUP BY player_name
ORDER BY shots_under_pressure DESC;


-- recoveries by player
SELECT player.name AS player_name, COUNT(*) AS recoveries
FROM `awesome-advice-420021.statsbomb.events`
WHERE type.name = "Ball Recovery"
GROUP BY player_name
ORDER BY recoveries DESC


-- matches with score
SELECT match_id, home_team.home_team_name, away_team.away_team_name, home_score, away_score
FROM `awesome-advice-420021.statsbomb.matches`
ORDER BY match_id;

-- goals by team and play pattern
SELECT team.name, play_pattern.name AS play_pattern, COUNT(*) AS goals
FROM `awesome-advice-420021.statsbomb.events`
WHERE shot.outcome.name = "Goal"
GROUP BY team.name, play_pattern
order by 1,2

-- substitutes by player 
SELECT player.name AS player_name, COUNT(*) AS appearances
FROM `awesome-advice-420021.statsbomb.events`
WHERE type.name = "Substitution"
GROUP BY player_name
ORDER BY appearances DESC

-- player with most appearances
SELECT player.name AS player_name, COUNT(*) AS appearances
FROM `awesome-advice-420021.statsbomb.events`
WHERE type.name = "Substitution"
GROUP BY player_name
ORDER BY appearances DESC


-- player stats view
CREATE OR REPLACE VIEW `awesome-advice-420021.statsbomb.player_stats` AS

SELECT
  player.name,
  COUNT(CASE WHEN type.name = 'Ball Recovery' THEN 1 END) AS Ball_Recovery_count,
  COUNT(CASE WHEN type.name = 'Dispossessed' THEN 1 END) AS Dispossessed_count,
  COUNT(CASE WHEN type.name = 'Duel' THEN 1 END) AS Duel_count,
  COUNT(CASE WHEN type.name = 'Block' THEN 1 END) AS Block_count,
  COUNT(CASE WHEN type.name = 'Offside' THEN 1 END) AS Offside_count,
  COUNT(CASE WHEN type.name = 'Clearance' THEN 1 END) AS Clearance_count,
  COUNT(CASE WHEN type.name = 'Interception' THEN 1 END) AS Interception_count,
  COUNT(CASE WHEN type.name = 'Dribble' THEN 1 END) AS Dribble_count,
  COUNT(CASE WHEN type.name = 'Shot' THEN 1 END) AS Shot_count,
  COUNT(CASE WHEN type.name = 'Pressure' THEN 1 END) AS Pressure_count,
  COUNT(CASE WHEN type.name = 'Half Start' THEN 1 END) AS Half_Start_count,
  COUNT(CASE WHEN type.name = 'Substitution' THEN 1 END) AS Substitution_count,
  COUNT(CASE WHEN type.name = 'Own Goal Against' THEN 1 END) AS Own_Goal_Against_count,
  COUNT(CASE WHEN type.name = 'Foul Won' THEN 1 END) AS Foul_Won_count,
  COUNT(CASE WHEN type.name = 'Foul Committed' THEN 1 END) AS Foul_Committed_count,
  COUNT(CASE WHEN type.name = 'Goal Keeper' THEN 1 END) AS Goal_Keeper_count,
  COUNT(CASE WHEN type.name = 'Bad Behaviour' THEN 1 END) AS Bad_Behaviour_count,
  COUNT(CASE WHEN type.name = 'Own Goal For' THEN 1 END) AS Own_Goal_For_count,
  COUNT(CASE WHEN type.name = 'Player On' THEN 1 END) AS Player_On_count,
  COUNT(CASE WHEN type.name = 'Player Off' THEN 1 END) AS Player_Off_count,
  COUNT(CASE WHEN type.name = 'Shield' THEN 1 END) AS Shield_count,
  COUNT(CASE WHEN type.name = 'Pass' THEN 1 END) AS Pass_count,
  COUNT(CASE WHEN type.name = '50/50' THEN 1 END) AS Fifty_Fifty_count,
  COUNT(CASE WHEN type.name = 'Half End' THEN 1 END) AS Half_End_count,
  COUNT(CASE WHEN type.name = 'Starting XI' THEN 1 END) AS Starting_XI_count,
  COUNT(CASE WHEN type.name = 'Tactical Shift' THEN 1 END) AS Tactical_Shift_count,
  COUNT(CASE WHEN type.name = 'Error' THEN 1 END) AS Error_count,
  COUNT(CASE WHEN type.name = 'Miscontrol' THEN 1 END) AS Miscontrol_count,
  COUNT(CASE WHEN type.name = 'Dribbled Past' THEN 1 END) AS Dribbled_Past_count,
  COUNT(CASE WHEN type.name = 'Injury Stoppage' THEN 1 END) AS Injury_Stoppage_count,
  COUNT(CASE WHEN type.name = 'Referee Ball-Drop' THEN 1 END) AS Referee_Ball_Drop_count,
  COUNT(CASE WHEN type.name = 'Ball Receipt*' THEN 1 END) AS Ball_Receipt_count,
  COUNT(CASE WHEN type.name = 'Carry' THEN 1 END) AS Carry_count,
  COUNT(*) AS total_events
FROM
  `awesome-advice-420021.statsbomb.events`
GROUP BY
  player.name

-- player clusters
CREATE OR REPLACE MODEL `awesome-advice-420021.statsbomb.player_clusters`
OPTIONS(model_type='kmeans', num_clusters=5) AS
SELECT
  * from `awesome-advice-420021.statsbomb.player_stats`


-- Model to predict xg from shot type, body part, technique, under pressure, play pattern, location

CREATE OR REPLACE MODEL
  `awesome-advice-420021.statsbomb.xg_prediction` OPTIONS(model_type='linear_reg',
    input_label_cols=['statsbomb_xg']) AS
SELECT
  shot.type.name AS shot_type,
  shot.body_part.name AS body_part,
  shot.technique.name AS technique,
  under_pressure,
  play_pattern.name,
  location,
  shot.statsbomb_xg
FROM
  `awesome-advice-420021.statsbomb.events`
WHERE
  shot.statsbomb_xg IS NOT NULL;



-- player shots
CREATE OR REPLACE VIEW `awesome-advice-420021.statsbomb.player_shots` AS 

SELECT
  CASE WHEN shot.outcome.name = 'Goal' THEN 1 ELSE 0 END as is_goal,
  shot.body_part.name AS body_part,
  shot.technique.name AS technique,
  shot.type.name AS shot_type,
  under_pressure,
  play_pattern.name AS play_pattern,
  player.name,
  period,
  SQRT(POW(120 - location[OFFSET(0)], 2) + POW(40 - location[OFFSET(1)], 2)) AS distance_to_goal,
  ABS(ATAN2(40 - location[OFFSET(1)], 120 - location[OFFSET(0)])) AS angle_to_goal,
  CASE 
    WHEN location[OFFSET(0)] >= 102 AND location[OFFSET(1)] BETWEEN 18 AND 62 THEN 1 
    ELSE 0 
  END AS is_in_box,
  CASE 
    WHEN ABS(location[OFFSET(1)] - 40) < 20 THEN 1 
    ELSE 0 
  END AS is_central,
  CASE 
    WHEN minute >= 75 THEN 1 
    ELSE 0 
  END AS is_late_game
FROM
  `awesome-advice-420021.statsbomb.events`
WHERE
  type.name = 'Shot'
  AND shot.type.name != 'Penalty'
  AND player.name IS NOT NULL


-- Model to predict if a shot is a goal or not
CREATE OR REPLACE MODEL `awesome-advice-420021.statsbomb.goal_prediction_model`
OPTIONS(
  model_type='AUTOML_CLASSIFIER',
  input_label_cols=['is_goal']
) AS
SELECT
  * from `awesome-advice-420021.statsbomb.player_shots`


-- Testing the model with explanation

SELECT
  *
FROM
  ML.EXPLAIN_PREDICT(MODEL `awesome-advice-420021.statsbomb.goal_prediction_model`,
    (
        SELECT
  * from `awesome-advice-420021.statsbomb.player_shots`

    )
  )


-- Create view with some stats per match

CREATE OR REPLACE VIEW `awesome-advice-420021.statsbomb.match_stats` as (

WITH match_stats AS (
  SELECT
    e.match_id,
    e.period,
    e.minute,
    e.second,
    m.home_team.home_team_name AS home_team,
    m.away_team.away_team_name AS away_team,
    m.home_score,
    m.away_score,
    SUM(CASE WHEN e.team.name = m.home_team.home_team_name THEN e.duration ELSE 0 END) AS home_possession,
    SUM(CASE WHEN e.team.name = m.away_team.away_team_name THEN e.duration ELSE 0 END) AS away_possession,
    SUM(CASE WHEN e.type.name = 'Shot' AND e.team.name = m.home_team.home_team_name THEN 1 ELSE 0 END) AS home_shots,
    SUM(CASE WHEN e.type.name = 'Shot' AND e.team.name = m.away_team.away_team_name THEN 1 ELSE 0 END) AS away_shots,
    SUM(CASE WHEN e.type.name = 'Shot' AND e.shot.outcome.name = 'Goal' AND e.team.name = m.home_team.home_team_name THEN 1 ELSE 0 END) AS home_goals,
    SUM(CASE WHEN e.type.name = 'Shot' AND e.shot.outcome.name = 'Goal' AND e.team.name = m.away_team.away_team_name THEN 1 ELSE 0 END) AS away_goals,
    SUM(CASE WHEN e.type.name = 'Pass' AND e.team.name = m.home_team.home_team_name THEN 1 ELSE 0 END) AS home_passes,
    SUM(CASE WHEN e.type.name = 'Pass' AND e.team.name = m.away_team.away_team_name THEN 1 ELSE 0 END) AS away_passes,
    SUM(CASE WHEN e.type.name = 'Foul Committed' AND e.team.name = m.home_team.home_team_name THEN 1 ELSE 0 END) AS home_fouls,
    SUM(CASE WHEN e.type.name = 'Foul Committed' AND e.team.name = m.away_team.away_team_name THEN 1 ELSE 0 END) AS away_fouls,
    SUM(CASE WHEN e.type.name = 'Duel' AND e.team.name = m.home_team.home_team_name THEN 1 ELSE 0 END) AS home_duels,
    SUM(CASE WHEN e.type.name = 'Duel' AND e.team.name = m.away_team.away_team_name THEN 1 ELSE 0 END) AS away_duels,
    COALESCE(AVG(CASE WHEN e.type.name = 'Shot' AND e.team.name = m.home_team.home_team_name THEN e.shot.statsbomb_xg END), 0) AS home_avg_xg,
    COALESCE(AVG(CASE WHEN e.type.name = 'Shot' AND e.team.name = m.away_team.away_team_name THEN e.shot.statsbomb_xg END), 0) AS away_avg_xg
  FROM
    `awesome-advice-420021.statsbomb.events` e
  JOIN
    `awesome-advice-420021.statsbomb.matches` m ON e.match_id = m.match_id
  GROUP BY
    e.match_id, e.period, e.minute, e.second, m.home_team.home_team_name, m.away_team.away_team_name, m.home_score, m.away_score
)

SELECT
  CASE
    WHEN home_score > away_score THEN 'home_win'
    WHEN home_score < away_score THEN 'away_win'
    ELSE 'draw'
  END AS match_outcome,
  period,
  minute,
  second,
  home_team,
  away_team,
  home_score - away_score AS score_difference,
  CASE WHEN (home_possession + away_possession) > 0 THEN home_possession / (home_possession + away_possession) ELSE 0 END AS home_possession_percentage,
  home_shots,
  away_shots,
  home_goals,
  away_goals,
  home_passes,
  away_passes,
  home_fouls,
  away_fouls,
  home_duels,
  away_duels,
  home_avg_xg,
  away_avg_xg,
  CASE WHEN minute > 0 THEN (home_shots - away_shots) / minute ELSE 0 END AS shot_difference_per_minute,
  CASE WHEN minute > 0 THEN (home_passes - away_passes) / minute ELSE 0 END AS pass_difference_per_minute,
  CASE WHEN minute > 0 THEN (home_fouls - away_fouls) / minute ELSE 0 END AS foul_difference_per_minute,
  CASE WHEN minute > 0 THEN (home_duels - away_duels) / minute ELSE 0 END AS duel_difference_per_minute,
  home_avg_xg - away_avg_xg AS xg_difference
FROM
  match_stats
WHERE
  minute > 0

  )

  --  Model to predict match outcome
CREATE OR REPLACE MODEL
  `awesome-advice-420021.statsbomb.match_outcome_prediction` OPTIONS( model_type='RANDOM_FOREST_CLASSIFIER',
    num_trials=20,
    input_label_cols=['match_outcome'] ) AS
SELECT
  *
FROM
  `awesome-advice-420021.statsbomb.match_stats`