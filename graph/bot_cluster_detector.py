from graph.neo4j_connector import Neo4jConnector

def detect_bot_clusters() -> list:
    """
    Detect groups of users who all commented within a tight time window.
    Returns list of suspected bot cluster groups.
    """
    connector = Neo4jConnector()

    # Find users who commented within 60 seconds of each other
    cluster_query = """
    MATCH (u1:User)-[c1:COMMENTED_ON]->(v:Video)
    MATCH (u2:User)-[c2:COMMENTED_ON]->(v)
    WHERE u1 <> u2
      AND abs(duration.between(
            datetime(c1.timestamp),
            datetime(c2.timestamp)
          ).seconds) < 60
    RETURN u1.name AS user1, u2.name AS user2,
           c1.timestamp AS time1, c2.timestamp AS time2
    ORDER BY c1.timestamp
    LIMIT 200
    """
    results = connector.run_query(cluster_query)

    # Find users with suspiciously high comment counts relative to the pool
    high_volume_query = """
    MATCH (u:User)-[:COMMENTED_ON]->(v:Video)
    WITH u, count(*) AS comment_count
    WHERE comment_count > 3
    RETURN u.name AS author, comment_count
    ORDER BY comment_count DESC
    """
    high_volume = connector.run_query(high_volume_query)

    connector.close()
    return {"coordinated_pairs": results, "high_volume_users": high_volume}

def get_graph_cluster_scores(authors: list) -> dict:
    """
    Assign a graph-based suspicion score (0-1) to each author.
    Authors in bot clusters get higher scores.
    """
    clusters = detect_bot_clusters()
    suspicious_authors = set()

    for pair in clusters["coordinated_pairs"]:
        suspicious_authors.add(pair["user1"])
        suspicious_authors.add(pair["user2"])
    for user in clusters["high_volume_users"]:
        suspicious_authors.add(user["author"])

    return {author: (1.0 if author in suspicious_authors else 0.0) for author in authors}
