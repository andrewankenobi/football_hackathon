-- Section 1: Schema Evaluation
SELECT
  *
FROM
  `statsbomb.INFORMATION_SCHEMA.COLUMNS`
ORDER BY
  table_name,
  ordinal_position;

-- Section 2: Vertex AI Connection and Gemini Model
CREATE OR REPLACE MODEL `statsbomb.gemini`
REMOTE WITH CONNECTION `projects/awesome-advice-420021/locations/us/connections/biglake`
OPTIONS (endpoint = 'gemini-pro');

-- Example query to use the Gemini model
WITH model_output AS (
  SELECT
    stadium_name, 
    ml_generate_text_llm_result AS location
  FROM
    ML.GENERATE_TEXT(MODEL `statsbomb.gemini`,
      (
      SELECT
        DISTINCT matches.stadium.name AS stadium_name,
        CONCAT('give me latitude and longitude for this famous stadium. Respond with the only information, to be used straight away in Looker studio. Format the answer "latitude, longitude" - example "53.4336, -2.9611". Here is the stadium name: ', matches.stadium.name) AS prompt
      FROM
        `statsbomb.matches` AS matches ),
      STRUCT( 0.1 AS temperature,
        TRUE AS flatten_json_output ) )
)

SELECT
  stadium_name,
  location,
  CAST(SPLIT(location, ', ')[SAFE_OFFSET(0)] AS FLOAT64) AS latitude,
  CAST(SPLIT(location, ', ')[SAFE_OFFSET(1)] AS FLOAT64) AS longitude
FROM
  model_output;

-- Section 3: Data Exploration Views
CREATE OR REPLACE VIEW `statsbomb.vw_goals_by_body_part` AS
SELECT 
  team.name AS team, 
  player.name AS player, 
  SUM(CASE WHEN shot.body_part.name = 'Left Foot' THEN 1 ELSE 0 END) AS left_foot_goals,
  SUM(CASE WHEN shot.body_part.name = 'Right Foot' THEN 1 ELSE 0 END) AS right_foot_goals,
  SUM(CASE WHEN shot.body_part.name = 'Head' THEN 1 ELSE 0 END) AS head_goals,
  SUM(CASE WHEN shot.body_part.name = 'Other' THEN 1 ELSE 0 END) AS other_goals,
  COUNT(shot.body_part.name) AS total_goals
FROM 
  `statsbomb.events`
WHERE 
  type.name = 'Shot' 
  AND shot.outcome.name = 'Goal'
GROUP BY 
  1, 2
ORDER BY 
  7 DESC;

CREATE OR REPLACE VIEW `statsbomb.vw_top_scorers` AS
SELECT player.name AS player_name, COUNT(*) AS goals
FROM `statsbomb.events`
WHERE shot.outcome.name = "Goal"
GROUP BY player_name
ORDER BY goals DESC;

CREATE OR REPLACE VIEW `statsbomb.vw_team_possession` AS
WITH possession_time AS (
  SELECT
    m.match_id,
    e.team.name as team_name,
    SUM(e.duration) as total_duration
  FROM
    `statsbomb.events` e
  JOIN
    `statsbomb.matches` m ON e.match_id = m.match_id
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
  avg_possession_percentage DESC;

-- Section 4: Player Statistics View
CREATE OR REPLACE VIEW `statsbomb.vw_player_stats` AS
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
  COUNT(CASE WHEN type.name = 'Pass' THEN 1 END) AS Pass_count,
  COUNT(*) AS total_events
FROM
  `statsbomb.events`
GROUP BY
  player.name;

-- Section 5: Player Shots View
CREATE OR REPLACE VIEW `statsbomb.vw_player_shots` AS 
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
  `statsbomb.events`
WHERE
  type.name = 'Shot'
  AND shot.type.name != 'Penalty'
  AND player.name IS NOT NULL;

-- Section 6: Match Statistics View
CREATE OR REPLACE VIEW `statsbomb.vw_match_stats` AS
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
    `statsbomb.events` e
  JOIN
    `statsbomb.matches` m ON e.match_id = m.match_id
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
  minute > 0;

-- Section 7: Models and Predictions
-- 7.1: Player Clustering Model
CREATE OR REPLACE MODEL `statsbomb.player_clusters`
OPTIONS(model_type='kmeans', num_clusters=5) AS
SELECT * FROM `statsbomb.vw_player_stats`;

-- Test the Player Clustering Model
SELECT
  *
FROM
  ML.PREDICT(MODEL `statsbomb.player_clusters`,
    (SELECT * FROM `statsbomb.vw_player_stats`))
LIMIT 10;

-- 7.2: Expected Goals (xG) Prediction Model
CREATE OR REPLACE MODEL
  `statsbomb.xg_prediction` 
OPTIONS(model_type='linear_reg', input_label_cols=['statsbomb_xg']) AS
SELECT
  shot.type.name AS shot_type,
  shot.body_part.name AS body_part,
  shot.technique.name AS technique,
  under_pressure,
  play_pattern.name,
  location,
  shot.statsbomb_xg
FROM
  `statsbomb.events`
WHERE
  shot.statsbomb_xg IS NOT NULL;

-- Test the xG Prediction Model
SELECT
  *
FROM
  ML.PREDICT(MODEL `statsbomb.xg_prediction`,
    (SELECT
      shot.type.name AS shot_type,
      shot.body_part.name AS body_part,
      shot.technique.name AS technique,
      under_pressure,
      play_pattern.name,
      location
    FROM
      `statsbomb.events`
    WHERE
      type.name = 'Shot'
    LIMIT 10))

-- 7.3: Goal Prediction Model
CREATE OR REPLACE MODEL `statsbomb.goal_prediction_model`
OPTIONS(
  model_type='AUTOML_CLASSIFIER',
  input_label_cols=['is_goal']
) AS
SELECT * FROM `statsbomb.vw_player_shots`;

-- Test the Goal Prediction Model
SELECT
  *
FROM
  ML.PREDICT(MODEL `statsbomb.goal_prediction_model`,
    (SELECT * FROM `statsbomb.vw_player_shots` LIMIT 10))

-- 7.4: Match Outcome Prediction Model
CREATE OR REPLACE MODEL
  `statsbomb.match_outcome_prediction` 
OPTIONS( 
  model_type='RANDOM_FOREST_CLASSIFIER',
  num_trials=20,
  input_label_cols=['match_outcome'] 
) AS
SELECT
  *
FROM
  `statsbomb.vw_match_stats`;

-- Test the Match Outcome Prediction Model
SELECT
  *
FROM
  ML.PREDICT(MODEL `statsbomb.match_outcome_prediction`,
    (SELECT * FROM `statsbomb.vw_match_stats` LIMIT 10))

-- 7.5: New Model: Player Performance Prediction
CREATE OR REPLACE MODEL `statsbomb.player_performance_prediction`
OPTIONS(
  model_type='AUTOML_REGRESSOR',
  input_label_cols=['performance_score']
) AS
SELECT
  player.name,
  COUNT(*) AS total_events,
  AVG(CASE WHEN type.name = 'Pass' THEN pass.length ELSE NULL END) AS avg_pass_length,
  SUM(CASE WHEN type.name = 'Shot' AND shot.outcome.name = 'Goal' THEN 1 ELSE 0 END) AS goals,
  SUM(CASE WHEN type.name = 'Pass' AND pass.outcome.name = 'Complete' THEN 1 ELSE 0 END) / NULLIF(COUNT(CASE WHEN type.name = 'Pass' THEN 1 END), 0) AS pass_completion_rate,
  SUM(CASE WHEN type.name = 'Duel' AND duel.outcome.name = 'Won' THEN 1 ELSE 0 END) / NULLIF(COUNT(CASE WHEN type.name = 'Duel' THEN 1 END), 0) AS duel_win_rate,
  (SUM(CASE WHEN type.name = 'Shot' AND shot.outcome.name = 'Goal' THEN 1 ELSE 0 END) * 3 +
   SUM(CASE WHEN type.name = 'Pass' AND pass.outcome.name = 'Complete' THEN 1 ELSE 0 END) * 0.1 +
   SUM(CASE WHEN type.name = 'Duel' AND duel.outcome.name = 'Won' THEN 1 ELSE 0 END) * 0.5) / NULLIF(COUNT(*), 0) AS performance_score
FROM
  `statsbomb.events`
GROUP BY
  player.name;

-- Test the Player Performance Prediction Model
SELECT
  *
FROM
  ML.PREDICT(MODEL `statsbomb.player_performance_prediction`,
    (SELECT
      player.name,
      COUNT(*) AS total_events,
      AVG(CASE WHEN type.name = 'Pass' THEN pass.length ELSE NULL END) AS avg_pass_length,
      SUM(CASE WHEN type.name = 'Shot' AND shot.outcome.name = 'Goal' THEN 1 ELSE 0 END) AS goals,
      SUM(CASE WHEN type.name = 'Pass' AND pass.outcome.name = 'Complete' THEN 1 ELSE 0 END) / NULLIF(COUNT(CASE WHEN type.name = 'Pass' THEN 1 END), 0) AS pass_completion_rate,
      SUM(CASE WHEN type.name = 'Duel' AND duel.outcome.name = 'Won' THEN 1 ELSE 0 END) / NULLIF(COUNT(CASE WHEN type.name = 'Duel' THEN 1 END), 0) AS duel_win_rate
    FROM
      `statsbomb.events`
    GROUP BY
      player.name
    LIMIT 10))

-- 7.6: New Model: Team Performance Prediction
CREATE OR REPLACE MODEL `statsbomb.team_performance_prediction`
OPTIONS(
  model_type='AUTOML_REGRESSOR',
  input_label_cols=['team_performance_score']
) AS
SELECT
  team.name AS team_name,
  COUNT(*) AS total_events,
  AVG(CASE WHEN type.name = 'Pass' THEN pass.length ELSE NULL END) AS avg_pass_length,
  SUM(CASE WHEN type.name = 'Shot' AND shot.outcome.name = 'Goal' THEN 1 ELSE 0 END) AS goals,
  SUM(CASE WHEN type.name = 'Pass' AND pass.outcome.name = 'Complete' THEN 1 ELSE 0 END) / NULLIF(COUNT(CASE WHEN type.name = 'Pass' THEN 1 END), 0) AS pass_completion_rate,
  SUM(CASE WHEN type.name = 'Duel' AND duel.outcome.name = 'Won' THEN 1 ELSE 0 END) / NULLIF(COUNT(CASE WHEN type.name = 'Duel' THEN 1 END), 0) AS duel_win_rate,
  (SUM(CASE WHEN type.name = 'Shot' AND shot.outcome.name = 'Goal' THEN 1 ELSE 0 END) * 3 +
   SUM(CASE WHEN type.name = 'Pass' AND pass.outcome.name = 'Complete' THEN 1 ELSE 0 END) * 0.1 +
   SUM(CASE WHEN type.name = 'Duel' AND duel.outcome.name = 'Won' THEN 1 ELSE 0 END) * 0.5) / NULLIF(COUNT(*), 0) AS team_performance_score
FROM
  `statsbomb.events`
GROUP BY
  team.name;

-- Test the Team Performance Prediction Model
SELECT
  *
FROM
  ML.PREDICT(MODEL `statsbomb.team_performance_prediction`,
    (SELECT
      team.name AS team_name,
      COUNT(*) AS total_events,
      AVG(CASE WHEN type.name = 'Pass' THEN pass.length ELSE NULL END) AS avg_pass_length,
      SUM(CASE WHEN type.name = 'Shot' AND shot.outcome.name = 'Goal' THEN 1 ELSE 0 END) AS goals,
      SUM(CASE WHEN type.name = 'Pass' AND pass.outcome.name = 'Complete' THEN 1 ELSE 0 END) / NULLIF(COUNT(CASE WHEN type.name = 'Pass' THEN 1 END), 0) AS pass_completion_rate,
      SUM(CASE WHEN type.name = 'Duel' AND duel.outcome.name = 'Won' THEN 1 ELSE 0 END) / NULLIF(COUNT(CASE WHEN type.name = 'Duel' THEN 1 END), 0) AS duel_win_rate
    FROM
      `statsbomb.events`
    GROUP BY
      team.name
    LIMIT 10))
