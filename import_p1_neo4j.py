import time
import os
import pandas as pd
from neo4j import GraphDatabase

URI = "neo4j+ssc://81a4f7a1.databases.neo4j.io"
USER = "81a4f7a1"
PASSWORD = "M26SjhxYPT8o-5w7-zRzyJNhG80Bok-l1VqeABIsb_U"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "events_p1_train.csv")
BATCH_SIZE = 5000

def main():
    print(f"[INFO] Membaca data {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    total_rows = len(df)
    print(f"[INFO] Total data P1 yang akan diimpor: {total_rows:,} baris")
    
    # Bersihkan NaN pada transactionid
    df["transactionid"] = df["transactionid"].fillna("").astype(str)
    
    print("[INFO] Menghubungkan ke Neo4j AuraDB...")
    start_time = time.time()
    
    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver:
        with driver.session() as session:
            # 1. Pastikan Constraints ada
            print("[INFO] Memastikan constraints unique ID...")
            session.run("CREATE CONSTRAINT visitor_id IF NOT EXISTS FOR (v:Visitor) REQUIRE v.visitorId IS UNIQUE")
            session.run("CREATE CONSTRAINT item_id IF NOT EXISTS FOR (i:Item) REQUIRE i.itemId IS UNIQUE")
            
            # 2. Impor dalam batch
            cypher_query = """
            UNWIND $batch AS row
            MERGE (v:Visitor {visitorId: row.visitorid})
            MERGE (i:Item {itemId: row.itemid})
            WITH v, i, row
            CALL (v, i, row) {
                WITH v, i, row WHERE row.event = 'view'
                CREATE (v)-[:VIEWED {timestamp: row.timestamp}]->(i)
                UNION ALL
                WITH v, i, row WHERE row.event = 'addtocart'
                CREATE (v)-[:ADDED_TO_CART {timestamp: row.timestamp}]->(i)
                UNION ALL
                WITH v, i, row WHERE row.event = 'transaction'
                CREATE (v)-[:BOUGHT {timestamp: row.timestamp, transactionId: row.transactionid}]->(i)
            }
            """
            
            total_batches = (total_rows // BATCH_SIZE) + (1 if total_rows % BATCH_SIZE != 0 else 0)
            print(f"[INFO] Memulai impor {total_batches} batch (batch size = {BATCH_SIZE})...")
            
            for b_idx in range(total_batches):
                b_start = b_idx * BATCH_SIZE
                b_end = min((b_idx + 1) * BATCH_SIZE, total_rows)
                chunk = df.iloc[b_start:b_end]
                records = chunk.to_dict(orient="records")
                session.run(cypher_query, batch=records)
                print(f"  * Batch {b_idx + 1}/{total_batches} selesai ({b_end:,}/{total_rows:,} rows)")
            
            elapsed = time.time() - start_time
            print(f"\n[SUKSES] Seluruh data berhasil diimpor dalam {elapsed:.2f} detik!")
            
            # Cek statistik graf di Neo4j
            n_visitor = session.run("MATCH (v:Visitor) RETURN count(v) as c").single()["c"]
            n_item = session.run("MATCH (i:Item) RETURN count(i) as c").single()["c"]
            n_view = session.run("MATCH ()-[r:VIEWED]->() RETURN count(r) as c").single()["c"]
            n_cart = session.run("MATCH ()-[r:ADDED_TO_CART]->() RETURN count(r) as c").single()["c"]
            n_bought = session.run("MATCH ()-[r:BOUGHT]->() RETURN count(r) as c").single()["c"]
            
            print("=" * 60)
            print("STATISTIK SUBGRAPH NEO4J AURADB:")
            print(f"  - Total Node Visitor : {n_visitor:,}")
            print(f"  - Total Node Item    : {n_item:,}")
            print(f"  - Relasi VIEWED      : {n_view:,}")
            print(f"  - Relasi ADDED_CART  : {n_cart:,}")
            print(f"  - Relasi BOUGHT      : {n_bought:,}")
            print("=" * 60)

if __name__ == "__main__":
    main()
