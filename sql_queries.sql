
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
    
