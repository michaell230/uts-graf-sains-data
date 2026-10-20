import json
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase

URI = "neo4j+ssc://81a4f7a1.databases.neo4j.io"
USER = "81a4f7a1"
PASSWORD = "M26SjhxYPT8o-5w7-zRzyJNhG80Bok-l1VqeABIsb_U"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_analysis():
    print("=== [1] MEMUAT DATA P1 (TRAIN) & P2 (TEST) ===")
    p1 = pd.read_csv(os.path.join(BASE_DIR, "events_p1_train.csv"))
    p2 = pd.read_csv(os.path.join(BASE_DIR, "events_p2_test.csv"))
    print(f"P1 (10-12 Sept 2015): {len(p1):,} baris")
    print(f"P2 (13-14 Sept 2015): {len(p2):,} baris")

    # Bobot event
    weight_map = {"view": 1.0, "addtocart": 2.0, "transaction": 3.0}
    p1["weight"] = p1["event"].map(weight_map)
    p2["weight"] = p2["event"].map(weight_map)

    # 1. Matriks VSM User-Item
    user_cats = pd.CategoricalDtype(categories=sorted(p1["visitorid"].unique()), ordered=True)
    item_cats = pd.CategoricalDtype(categories=sorted(p1["itemid"].unique()), ordered=True)
    
    n_users = len(user_cats.categories)
    n_items = len(item_cats.categories)
    
    u_codes = p1["visitorid"].astype(user_cats).cat.codes
    i_codes = p1["itemid"].astype(item_cats).cat.codes
    
    # Matriks VSM: User x Item
    R_sparse = csr_matrix((p1["weight"], (u_codes, i_codes)), shape=(n_users, n_items))
    print(f"\n=== [2] VSM CHARACTERISTICS ===")
    print(f"User VSM Shape: {R_sparse.shape} (Users: {n_users:,}, Items: {n_items:,})")
    nnz = R_sparse.nnz
    total_cells = n_users * n_items
    sparsity = (1.0 - (nnz / total_cells)) * 100
    print(f"Non-zero interactions: {nnz:,}")
    print(f"Sparsity: {sparsity:.5f}%")
    
    # 2. Item-Item Similarity (Anchor Item)
    # Temukan item terpopuler sebagai acuan (anchor item)
    top_items = p1["itemid"].value_counts()
    anchor_item_id = int(top_items.index[0])
    anchor_item_count = top_items.iloc[0]
    print(f"\n=== [3] ANCHOR ITEM IDENTIFICATION ===")
    print(f"Anchor Item ID (Paling Populer): {anchor_item_id} (Dilihat/Diinteraksikan {anchor_item_count} kali)")
    
    # Hitung Item-Item cosine similarity untuk anchor item
    # Item VSM adalah R_sparse.T (shape: n_items x n_users)
    R_item = R_sparse.T.tocsr()
    anchor_idx = item_cats.categories.get_loc(anchor_item_id)
    anchor_vec = R_item[anchor_idx]
    
    item_sims = cosine_similarity(anchor_vec, R_item)[0]
    item_sims[anchor_idx] = 0.0 # abaikan self-similarity
    
    top5_item_indices = np.argsort(item_sims)[::-1][:5]
    top5_item_results = []
    print("\nTop 5 Items Paling Mirip dengan Item", anchor_item_id, ":")
    for rank, idx in enumerate(top5_item_indices, 1):
        target_item_id = int(item_cats.categories[idx])
        sim_score = float(item_sims[idx])
        top5_item_results.append({"rank": rank, "item_id": target_item_id, "similarity": sim_score})
        print(f"  {rank}. Item {target_item_id}: Similarity = {sim_score:.4f}")

    # 3. Tuliskan Kembali Similarity ke Neo4j AuraDB
    print("\n=== [4] MENYIMPAN SIMILARITY RELATIONSHIP KE NEO4J ===")
    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver:
        with driver.session() as session:
            # Hapus relasi SIMILAR lama jika ada
            session.run("MATCH ()-[s:SIMILAR_TO]->() DELETE s")
            
            # Tulis 5 relasi SIMILAR_TO
            write_query = """
            UNWIND $sims AS row
            MATCH (source:Item {itemId: row.source_id})
            MATCH (target:Item {itemId: row.target_id})
            MERGE (source)-[s:SIMILAR_TO {rank: row.rank}]->(target)
            SET s.score = row.score
            RETURN source.itemId, type(s), s.score, target.itemId
            """
            sim_records = [
                {"source_id": anchor_item_id, "target_id": r["item_id"], "rank": r["rank"], "score": r["similarity"]}
                for r in top5_item_results
            ]
            res = session.run(write_query, sims=sim_records)
            print("Berhasil menyimpan relasi SIMILAR_TO ke AuraDB:")
            for r in res:
                print(f"  ({r['source.itemId']})-[:SIMILAR_TO {{score: {r['s.score']:.4f}}}]->({r['target.itemId']})")

    # 4. Evaluasi KNN Recommendation (Soal 2.2)
    print("\n=== [5] EVALUASI KNN RECOMMENDATION (SOAL 2.2) ===")
    # Pemilihan user acak yang reproducible
    # Kita gunakan target user: 1003939 (memiliki interaksi di P1 dan P2)
    u_target = 1003939
    print(f"Target User Terpilih (Reproducible): {u_target}")
    
    p1_target_items = set(p1[p1["visitorid"] == u_target]["itemid"])
    p2_target_groundtruth = set(p2[p2["visitorid"] == u_target]["itemid"])
    print(f"Histori User di P1 (Train): {list(p1_target_items)}")
    print(f"Ground Truth User di P2 (Test): {list(p2_target_groundtruth)}")
    
    # Hitung User-User Similarity
    user_idx = user_cats.categories.get_loc(u_target)
    u_sims = cosine_similarity(R_sparse[user_idx], R_sparse)[0]
    u_sims[user_idx] = 0.0 # self-sim = 0
    
    top5_neighbor_indices = np.argsort(u_sims)[::-1][:5]
    top5_neighbors = []
    print("\nTop 5 Tetangga Terdekat (K=5 KNN):")
    for rank, n_idx in enumerate(top5_neighbor_indices, 1):
        n_user_id = int(user_cats.categories[n_idx])
        sim_val = float(u_sims[n_idx])
        top5_neighbors.append((n_user_id, sim_val, n_idx))
        print(f"  {rank}. Neighbor User {n_user_id}: Sim = {sim_val:.4f}")
        
    # Rekomendasi Item dari tetangga terdekat
    candidate_item_scores = {}
    for n_user_id, sim_val, n_idx in top5_neighbors:
        if sim_val > 0:
            neighbor_items = R_sparse[n_idx].nonzero()[1]
            for it_code in neighbor_items:
                it_id = int(item_cats.categories[it_code])
                # Jangan rekomendasikan item yang sudah pernah dilihat di P1
                if it_id not in p1_target_items:
                    candidate_item_scores[it_id] = candidate_item_scores.get(it_id, 0.0) + (sim_val * R_sparse[n_idx, it_code])
                    
    # Urutkan rekomendasi
    recommended_items = sorted(candidate_item_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 Rekomendasi Item:")
    rec_ids = []
    for rank, (it_id, score) in enumerate(recommended_items, 1):
        rec_ids.append(it_id)
        is_hit = " [HIT di P2!]" if it_id in p2_target_groundtruth else ""
        print(f"  {rank}. Item {it_id}: Score = {score:.4f}{is_hit}")
        
    top3_advertised = rec_ids[:3]
    print(f"\n3 Item yang Diiklankan untuk K=5 (No. 1b): {top3_advertised}")
    
    # Hitung Hit@5
    hit_at_5 = 1 if any(it in p2_target_groundtruth for it in rec_ids[:5]) else 0
    print(f"\n(a) Hit@5 = {hit_at_5}")
    
    # Hitung NDCG@5
    dcg = 0.0
    for rank, it_id in enumerate(rec_ids[:5], 1):
        rel = 1.0 if it_id in p2_target_groundtruth else 0.0
        dcg += rel / np.log2(rank + 1)
        
    # IDCG (Ideal DCG untuk 1 relevant item di rank 1)
    idcg = 1.0 / np.log2(1 + 1) # 1.0
    ndcg_at_5 = dcg / idcg
    print(f"(b) NDCG@5 = {ndcg_at_5:.4f}")
    
    # Simpan summary hasil dalam JSON
    summary_data = {
        "p1_rows": len(p1),
        "p2_rows": len(p2),
        "vsm_shape": [n_users, n_items],
        "sparsity": sparsity,
        "anchor_item": anchor_item_id,
        "top5_item_sims": top5_item_results,
        "target_user": u_target,
        "k5_neighbors": [(n[0], n[1]) for n in top5_neighbors],
        "top3_advertised": top3_advertised,
        "top5_recommended": rec_ids,
        "hit_at_5": hit_at_5,
        "ndcg_at_5": ndcg_at_5
    }
    output_json_path = os.path.join(BASE_DIR, "hasil_analisis_uts.json")
    with open(output_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[SELESAI] Hasil analisis disimpan ke {output_json_path}")

if __name__ == "__main__":
    run_analysis()
