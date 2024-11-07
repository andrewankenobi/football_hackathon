
  -- SECTION 1: Schema Evaluation
  CREATE OR REPLACE VIEW `statsbomb.vw_schema_info` AS
  SELECT
    *
  FROM
    `statsbomb.INFORMATION_SCHEMA.COLUMNS`
  ORDER BY
    table_name,
    ordinal_position;

  -- SECTION 2: Vertex AI Connection and Gemini Model
  CREATE OR REPLACE MODEL `statsbomb.gemini`
  REMOTE WITH CONNECTION `us-central1.gemini-connection`
  OPTIONS (endpoint = 'gemini-pro');

  -- SECTION 3: Data Exploration Views
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

  -- SECTION 4: Player Statistics View
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

  -- SECTION 5: Player Shots View
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

  -- SECTION 6: Match Statistics View
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

  -- SECTION 7: Models and Predictions
  CREATE OR REPLACE MODEL `statsbomb.player_clusters`
  OPTIONS(
    model_type = 'kmeans',
    num_clusters = HPARAM_RANGE(3, 6), -- This defines the range of clusters to try
    num_trials = 10,  -- This defines how many different models to try with different hyperparameters
    hparam_tuning_objectives = ['DAVIES_BOULDIN_INDEX'], -- This defines the objective function to optimize
    hparam_tuning_algorithm = 'VIZIER_DEFAULT' -- This defines the algorithm to use for hyperparameter tuning
  ) AS
  SELECT * FROM `statsbomb.vw_player_stats`;

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

  CREATE OR REPLACE MODEL `statsbomb.goal_prediction_model`
  OPTIONS(
    model_type='LOGISTIC_REG',
    input_label_cols=['is_goal']
  ) AS
  SELECT * FROM `statsbomb.vw_player_shots`;

  CREATE OR REPLACE MODEL
    `statsbomb.match_outcome_prediction` 
  OPTIONS( 
    model_type='LOGISTIC_REG',
    input_label_cols=['match_outcome'] 
  ) AS
  SELECT
    *
  FROM
    `statsbomb.vw_match_stats`;

  -- SECTION 8: Player Embeddings and Similarity Search
  CREATE OR REPLACE VIEW `statsbomb.vw_player_stats_for_embeddings` AS
  SELECT
    player.name AS player_name,
    COUNT(*) AS total_events,
    AVG(CASE WHEN type.name = 'Pass' THEN 1 ELSE 0 END) AS pass_ratio,
    AVG(CASE WHEN type.name = 'Shot' THEN 1 ELSE 0 END) AS shot_ratio,
    AVG(CASE WHEN type.name = 'Ball Recovery' THEN 1 ELSE 0 END) AS ball_recovery_ratio,
    AVG(CASE WHEN type.name = 'Duel' THEN 1 ELSE 0 END) AS duel_ratio,
    AVG(CASE WHEN type.name = 'Interception' THEN 1 ELSE 0 END) AS interception_ratio,
    AVG(CASE WHEN shot.outcome.name = 'Goal' THEN 1 ELSE 0 END) AS goal_ratio,
    AVG(CASE WHEN type.name = 'Pressure' THEN 1 ELSE 0 END) AS pressure_ratio,
    AVG(CASE WHEN type.name = 'Dribble' THEN 1 ELSE 0 END) AS dribble_ratio,
    AVG(CASE WHEN type.name = 'Foul Committed' THEN 1 ELSE 0 END) AS foul_committed_ratio,
    AVG(CASE WHEN type.name = 'Foul Won' THEN 1 ELSE 0 END) AS foul_won_ratio,
    AVG(CASE WHEN type.name = 'Carry' THEN 1 ELSE 0 END) AS carry_ratio,
    AVG(CASE WHEN type.name = 'Dispossessed' THEN 1 ELSE 0 END) AS dispossessed_ratio,
    AVG(CASE WHEN type.name = 'Clearance' THEN 1 ELSE 0 END) AS clearance_ratio,
    AVG(CASE WHEN type.name = 'Block' THEN 1 ELSE 0 END) AS block_ratio
  FROM
    `statsbomb.events`
  WHERE
    player.name IS NOT NULL
  GROUP BY
    player.name
  HAVING
    COUNT(*) > 100;  -- Filter out players with too few events

  CREATE OR REPLACE MODEL `statsbomb.player_embedding_model`
  OPTIONS(
    MODEL_TYPE='PCA', 
    pca_explained_variance_ratio= 0.9
  ) AS
  SELECT
    *
  FROM
    `statsbomb.vw_player_stats_for_embeddings`;

  CREATE OR REPLACE TABLE `statsbomb.player_embeddings` AS
  SELECT
    *
  FROM
    ML.GENERATE_EMBEDDING(
      MODEL `statsbomb.player_embedding_model`,
      TABLE `statsbomb.vw_player_stats_for_embeddings`);

  -- Register player_clusters model
  ALTER MODEL IF EXISTS `statsbomb.player_clusters` 
  SET OPTIONS (vertex_ai_model_id="statsbomb_player_clusters");

  -- Register xg_prediction model
  ALTER MODEL IF EXISTS `statsbomb.xg_prediction` 
  SET OPTIONS (vertex_ai_model_id="statsbomb_xg_prediction");

  -- Register goal_prediction_model
  ALTER MODEL IF EXISTS `statsbomb.goal_prediction_model` 
  SET OPTIONS (vertex_ai_model_id="statsbomb_goal_prediction");

  -- Register match_outcome_prediction model
  ALTER MODEL IF EXISTS `statsbomb.match_outcome_prediction` 
  SET OPTIONS (vertex_ai_model_id="statsbomb_match_outcome_prediction");

  -- SECTION 9: Queries to Showcase Results
  -- These queries demonstrate the insights we can gain from our views, tables, and models
  -- Note: These should be run separately after all the above DDL statements have been executed

  -- 9.1: Top 10 Scorers
  -- SELECT * FROM `statsbomb.vw_top_scorers` LIMIT 10;

  -- 9.2: Team Possession Analysis
  -- SELECT * FROM `statsbomb.vw_team_possession` LIMIT 10;

  -- 9.3: Player Clustering Results
  -- SELECT * FROM ML.PREDICT(MODEL `statsbomb.player_clusters`, (SELECT * FROM `statsbomb.vw_player_stats`)) LIMIT 10;

  -- 9.4: Expected Goals (xG) Prediction
  -- SELECT * FROM ML.EXPLAIN_PREDICT(MODEL `statsbomb.xg_prediction`, (SELECT shot.type.name AS shot_type, shot.body_part.name AS body_part, shot.technique.name AS technique, under_pressure, play_pattern.name, location FROM `statsbomb.events` WHERE type.name = 'Shot' LIMIT 10))

  -- 9.5: Goal Prediction
  -- SELECT * FROM ML.PREDICT(MODEL `statsbomb.goal_prediction_model`, (SELECT * FROM `statsbomb.vw_player_shots` LIMIT 10))

  -- 9.6: Match Outcome Prediction
  -- SELECT * FROM ML.PREDICT(MODEL `statsbomb.match_outcome_prediction`, (SELECT * FROM `statsbomb.vw_match_stats` LIMIT 10))

  -- 9.7: Similar Players to a Goalkeeper (e.g., Petr Čech)
  -- SELECT base.* FROM VECTOR_SEARCH(TABLE `statsbomb.player_embeddings`, 'ml_generate_embedding_result', (SELECT ml_generate_embedding_result FROM `statsbomb.player_embeddings` WHERE player_name = 'Petr Čech'), top_k => 10, distance_type => 'COSINE') WHERE base.player_name != 'Petr Čech';

  -- 9.8: Similar Players to a Defender (e.g., Angelo Obinze Ogbonna)
  -- SELECT base.* FROM VECTOR_SEARCH(TABLE `statsbomb.player_embeddings`, 'ml_generate_embedding_result', (SELECT ml_generate_embedding_result FROM `statsbomb.player_embeddings` WHERE player_name = 'Angelo Obinze Ogbonna'), top_k => 10, distance_type => 'COSINE') WHERE base.player_name != 'Angelo Obinze Ogbonna';

