import pandas as pd
from graph.neo4j_connector import Neo4jConnector
from pyvis.network import Network

def generate_network_html(suspicious_authors: set) -> str:
    """
    Connects to Neo4j, fetches the interaction graph, and generates
    an interactive PyVis HTML string for Streamlit embedding.
    """
    connector = Neo4jConnector()
    query = """
    MATCH (u:User)-[r:COMMENTED_ON]->(v:Video)
    RETURN u.name AS user, v.id AS video, r.text AS comment
    LIMIT 300
    """
    records = connector.run_query(query)
    connector.close()
    
    # Initialize PyVis network with a dark theme
    net = Network(
        height="600px", 
        width="100%", 
        bgcolor="#0E1117", 
        font_color="white", 
        directed=True
    )
    
    # Use physics engine for organic layout
    net.barnes_hut(spring_length=150, spring_strength=0.05, damping=0.09)
    
    # Add central video node
    net.add_node("current_video", label="YouTube Video", color="#EF4444", size=35, shape="star")
    
    # Add User nodes and edges
    added_users = set()
    for row in records:
        user = row["user"]
        comment = row["comment"]
        
        if user not in added_users:
            if user in suspicious_authors:
                color = "#F59E0B"  # Orange/Suspicious
                size = 20
            else:
                color = "#10B981"  # Green/Human
                size = 15
                
            net.add_node(user, label=user, color=color, size=size)
            added_users.add(user)
            
        # Add edge from User to Video
        net.add_edge(user, "current_video", title=comment, color="#334155")
        
    return net.generate_html()
