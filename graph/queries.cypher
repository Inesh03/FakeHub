-- View all users and their comment counts
MATCH (u:User)-[:COMMENTED_ON]->(v:Video)
RETURN u.name, count(*) AS comments
ORDER BY comments DESC;

-- Find coordinated posting rings (within 30 seconds)
MATCH (u1:User)-[c1:COMMENTED_ON]->(v:Video)<-[c2:COMMENTED_ON]-(u2:User)
WHERE u1 <> u2
  AND abs(duration.between(datetime(c1.timestamp), datetime(c2.timestamp)).seconds) < 30
RETURN u1.name, u2.name, c1.timestamp, c2.timestamp;

-- Identify isolated bot-like nodes (high output, no engagement received)
MATCH (u:User)-[c:COMMENTED_ON]->(v:Video)
WITH u, count(c) AS total_comments
WHERE total_comments > 5
RETURN u.name, total_comments
ORDER BY total_comments DESC;
