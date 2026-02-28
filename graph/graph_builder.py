import pandas as pd
from graph.neo4j_connector import Neo4jConnector

def build_interaction_graph(df: pd.DataFrame):
    """
    Build a graph where:
    - Nodes: User accounts
    - Edges: REPLIED_TO (with timestamp and comment_id properties)
    """
    connector = Neo4jConnector()
    connector.clear_graph()

    # Create User nodes
    create_user_query = """
    MERGE (u:User {channel_id: $channel_id})
    SET u.name = $name,
        u.comment_count = $comment_count
    """
    for _, row in df.drop_duplicates("author_channel_id").iterrows():
        connector.run_query(create_user_query, {
            "channel_id":    row["author_channel_id"] or row["author"],
            "name":          row["author"],
            "comment_count": int(df[df["author"] == row["author"]].shape[0])
        })

    # Create COMMENTED_ON edges (User → Video)
    video_node_query = "MERGE (v:Video {id: $video_id})"
    connector.run_query(video_node_query, {"video_id": "current_video"})

    edge_query = """
    MATCH (u:User {channel_id: $channel_id})
    MATCH (v:Video {id: $video_id})
    CREATE (u)-[:COMMENTED_ON {
        timestamp: $timestamp,
        text:      $text,
        comment_id: $comment_id
    }]->(v)
    """
    for _, row in df.iterrows():
        connector.run_query(edge_query, {
            "channel_id": row["author_channel_id"] or row["author"],
            "video_id":   "current_video",
            "timestamp":  row["published_at"].isoformat(),
            "text":       row["text"][:200],
            "comment_id": row["comment_id"]
        })

    connector.close()
    print(f"Graph built: {len(df)} comment edges created.")
