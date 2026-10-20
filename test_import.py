import pandas as pd
from neo4j import GraphDatabase

uri = "neo4j+ssc://81a4f7a1.databases.neo4j.io"
user = "81a4f7a1"
pw = "M26SjhxYPT8o-5w7-zRzyJNhG80Bok-l1VqeABIsb_U"

df_sample = pd.read_csv("Tugas/events_p1_train.csv", nrows=100)
df_sample["transactionid"] = df_sample["transactionid"].fillna("").astype(str)
records = df_sample.to_dict(orient="records")

cypher_query = """
UNWIND $batch AS row
MERGE (v:Visitor {visitorId: row.visitorid})
MERGE (i:Item {itemId: row.itemid})
WITH v, i, row
CALL {
    WITH v, i, row
    WITH v, i, row WHERE row.event = 'view'
    CREATE (v)-[:VIEWED {timestamp: row.timestamp}]->(i)
    UNION ALL
    WITH v, i, row
    WITH v, i, row WHERE row.event = 'addtocart'
    CREATE (v)-[:ADDED_TO_CART {timestamp: row.timestamp}]->(i)
    UNION ALL
    WITH v, i, row
    WITH v, i, row WHERE row.event = 'transaction'
    CREATE (v)-[:BOUGHT {timestamp: row.timestamp, transactionId: row.transactionid}]->(i)
}
"""

with GraphDatabase.driver(uri, auth=(user, pw)) as driver:
    with driver.session() as session:
        session.run(cypher_query, batch=records)
        v_count = session.run("MATCH (v:Visitor) RETURN count(v) as c").single()["c"]
        i_count = session.run("MATCH (i:Item) RETURN count(i) as c").single()["c"]
        r_count = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
        print(f"Import Test Success! Visitor nodes: {v_count}, Item nodes: {i_count}, Relationships: {r_count}")
