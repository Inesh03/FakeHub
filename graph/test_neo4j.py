from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env")

uri      = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

print(f"Connecting to: {uri}")

driver = GraphDatabase.driver(uri, auth=(username, password))
with driver.session() as session:
    result = session.run("RETURN 'Neo4j connected! ✅' AS message")
    print(result.single()["message"])
driver.close()
