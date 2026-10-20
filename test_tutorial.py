from neo4j import GraphDatabase, RoutingControl

URI = "neo4j+ssc://81a4f7a1.databases.neo4j.io"
AUTH = ("81a4f7a1", "M26SjhxYPT8o-5w7-zRzyJNhG80Bok-l1VqeABIsb_U")
DATABASE_NAME = "81a4f7a1"

def add_friend(driver, name, friend_name):
    driver.execute_query(
        "MERGE (a:Person {name: $name}) "
        "MERGE (friend:Person {name: $friend_name}) "
        "MERGE (a)-[:KNOWS]->(friend)",
        name=name, friend_name=friend_name, database_=DATABASE_NAME
    )

def print_friends(driver, name):
    records, _, _ = driver.execute_query(
        "MATCH (a:Person)-[:KNOWS]->(friend) WHERE a.name = $name "
        "RETURN friend.name ORDER BY friend.name",
        name=name, database_=DATABASE_NAME, routing_=RoutingControl.READ
    )
    for record in records:
        print(f"Teman dari {name}:", record["friend.name"])

if __name__ == "__main__":
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        print("[INFO] Menjalankan tutorial 1.2: Pembuatan Node Person & Relasi KNOWS...")
        add_friend(driver, "Arthur", "Guinevere")
        add_friend(driver, "Arthur", "Lancelot")
        add_friend(driver, "Arthur", "Merlin")
        print_friends(driver, "Arthur")
        
        # Bersihkan node tutorial agar tidak mencemari dataset RetailRocket
        driver.execute_query("MATCH (p:Person) DETACH DELETE p", database_=DATABASE_NAME)
        print("[INFO] Node Person tutorial berhasil dibersihkan (CYPHER Total Reset).")
