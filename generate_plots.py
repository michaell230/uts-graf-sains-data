import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

p1 = pd.read_csv(os.path.join(BASE_DIR, "events_p1_train.csv"))

# 1. Visualisasi VSM User (Ego network target user 1003939 dan top neighbors)
print("[1] Membuat plot VSM User...")
u_target = 1003939
top_neighbors = [1235988, 406151, 45462]

sub_users = [u_target] + top_neighbors
df_u = p1[p1["visitorid"].isin(sub_users)]

G_u = nx.DiGraph()
for _, row in df_u.iterrows():
    u_node = f"User_{row['visitorid']}"
    i_node = f"Item_{row['itemid']}"
    G_u.add_node(u_node, node_type="User")
    G_u.add_node(i_node, node_type="Item")
    G_u.add_edge(u_node, i_node, event=row['event'])

plt.figure(figsize=(12, 7), dpi=300)
pos = nx.spring_layout(G_u, seed=42, k=0.6)

user_nodes = [n for n, d in G_u.nodes(data=True) if d.get('node_type') == 'User']
item_nodes = [n for n, d in G_u.nodes(data=True) if d.get('node_type') == 'Item']

# Draw nodes
nx.draw_networkx_nodes(G_u, pos, nodelist=[f"User_{u_target}"], node_color="#e63946", node_size=900, label="Target User (1003939)")
nx.draw_networkx_nodes(G_u, pos, nodelist=[f"User_{u}" for u in top_neighbors if f"User_{u}" in G_u], node_color="#457b9d", node_size=700, label="KNN Neighbors")
nx.draw_networkx_nodes(G_u, pos, nodelist=item_nodes, node_color="#2a9d8f", node_size=400, label="Interacted Items")

# Draw edges
nx.draw_networkx_edges(G_u, pos, edge_color="#b0bec5", arrows=True, arrowsize=14, alpha=0.7, width=1.2)
nx.draw_networkx_labels(G_u, pos, font_size=8, font_weight="bold", font_color="#1d3557")

plt.title("Visualisasi Vector Space Model (VSM) - Subgraf Interaksi User & Tetangga KNN", fontsize=14, fontweight="bold", pad=15)
plt.legend(scatterpoints=1, loc="upper right", frameon=True, facecolor="white", edgecolor="#ddd")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "visualisasi_vsm_user.png"), dpi=300)
plt.close()
print("Saved visualisasi_vsm_user.png")

# 2. Visualisasi VSM Item (Anchor Item 245085 dan Visitor yang melihatnya)
print("[2] Membuat plot VSM Item...")
anchor_item = 245085
df_item = p1[p1["itemid"] == anchor_item].head(25) # sample 25 pengunjung

G_i = nx.DiGraph()
G_i.add_node(f"Item_{anchor_item}", node_type="Anchor")
for _, row in df_item.iterrows():
    u_node = f"User_{row['visitorid']}"
    G_i.add_node(u_node, node_type="User")
    G_i.add_edge(u_node, f"Item_{anchor_item}")

plt.figure(figsize=(11, 7), dpi=300)
pos_i = nx.spring_layout(G_i, seed=2026, k=0.5)

nx.draw_networkx_nodes(G_i, pos_i, nodelist=[f"Item_{anchor_item}"], node_color="#f4a261", node_size=1200, label=f"Anchor Item ({anchor_item})")
nx.draw_networkx_nodes(G_i, pos_i, nodelist=[n for n in G_i.nodes if n != f"Item_{anchor_item}"], node_color="#457b9d", node_size=450, label="Visitors (Sample 25)")
nx.draw_networkx_edges(G_i, pos_i, edge_color="#90a4ae", arrows=True, arrowsize=12, alpha=0.75, width=1.2)
nx.draw_networkx_labels(G_i, pos_i, font_size=7.5, font_weight="bold")

plt.title(f"Visualisasi VSM Item - Sebaran Pengunjung pada Anchor Item {anchor_item}", fontsize=14, fontweight="bold", pad=15)
plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#ddd")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "visualisasi_vsm_item.png"), dpi=300)
plt.close()
print("Saved visualisasi_vsm_item.png")

# 3. Visualisasi Sebelum vs Sesudah Similarity
print("[3] Membuat plot Kondisi Sebelum vs Sesudah Similarity...")
anchor = 245085
top5_sim_items = [58677, 403848, 70148, 76060, 315309]
sim_scores = [0.1747, 0.1361, 0.1326, 0.1295, 0.1271]

# Sebelum: Item-item hanya terhubung melalui interaksi user bersama
G_before = nx.Graph()
G_before.add_node(f"Item_{anchor}", type="Anchor")
for it in top5_sim_items:
    G_before.add_node(f"Item_{it}", type="Target")

# Sample shared user
common_users = ["User_Common_A", "User_Common_B", "User_Common_C"]
for u in common_users:
    G_before.add_node(u, type="User")
    G_before.add_edge(u, f"Item_{anchor}")
    G_before.add_edge(u, f"Item_{top5_sim_items[0]}")
    G_before.add_edge(u, f"Item_{top5_sim_items[1]}")

fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# Posisi konsisten untuk item
pos_sim = {
    f"Item_{anchor}": np.array([0.0, 0.0]),
    f"Item_{top5_sim_items[0]}": np.array([0.8, 0.6]),
    f"Item_{top5_sim_items[1]}": np.array([0.9, -0.4]),
    f"Item_{top5_sim_items[2]}": np.array([0.2, 0.9]),
    f"Item_{top5_sim_items[3]}": np.array([0.3, -0.9]),
    f"Item_{top5_sim_items[4]}": np.array([-0.7, 0.7]),
    "User_Common_A": np.array([0.35, 0.3]),
    "User_Common_B": np.array([0.45, -0.2]),
    "User_Common_C": np.array([0.1, 0.45]),
}

# Subplot 1: Sebelum Perhitungan Similarity
ax1 = axes[0]
nx.draw_networkx_nodes(G_before, pos_sim, nodelist=[f"Item_{anchor}"], node_color="#e76f51", node_size=1000, ax=ax1, label="Anchor Item (245085)")
nx.draw_networkx_nodes(G_before, pos_sim, nodelist=[f"Item_{it}" for it in top5_sim_items], node_color="#2a9d8f", node_size=750, ax=ax1, label="Target Items")
nx.draw_networkx_nodes(G_before, pos_sim, nodelist=common_users, node_color="#e9c46a", node_size=550, ax=ax1, label="Bipartite Visitor Connectors")
nx.draw_networkx_edges(G_before, pos_sim, edge_color="#78909c", style="dashed", width=1.5, ax=ax1)
nx.draw_networkx_labels(G_before, pos_sim, font_size=8, font_weight="bold", ax=ax1)
ax1.set_title("Kondisi SEBELUM Perhitungan Similarity\n(Item hanya terhubung tak langsung lewat interaksi pengunjung)", fontsize=11, fontweight="bold")
ax1.legend(loc="lower left", frameon=True, facecolor="white")
ax1.axis('off')

# Subplot 2: Sesudah Perhitungan Similarity
ax2 = axes[1]
G_after = nx.DiGraph()
G_after.add_node(f"Item_{anchor}")
for it, score in zip(top5_sim_items, sim_scores):
    G_after.add_node(f"Item_{it}")
    G_after.add_edge(f"Item_{anchor}", f"Item_{it}", weight=score)

nx.draw_networkx_nodes(G_after, pos_sim, nodelist=[f"Item_{anchor}"], node_color="#e76f51", node_size=1000, ax=ax2, label="Anchor Item (245085)")
nx.draw_networkx_nodes(G_after, pos_sim, nodelist=[f"Item_{it}" for it in top5_sim_items], node_color="#2a9d8f", node_size=750, ax=ax2, label="Top 5 Similar Items")

edge_labels = {(f"Item_{anchor}", f"Item_{it}"): f"sim={score:.3f}" for it, score in zip(top5_sim_items, sim_scores)}
nx.draw_networkx_edges(G_after, pos_sim, edge_color="#e63946", width=2.5, arrows=True, arrowsize=18, ax=ax2)
nx.draw_networkx_edge_labels(G_after, pos_sim, edge_labels=edge_labels, font_size=8, font_color="#d62828", ax=ax2)
nx.draw_networkx_labels(G_after, pos_sim, font_size=8, font_weight="bold", ax=ax2)
ax2.set_title("Kondisi SESUDAH Perhitungan Similarity\n(Terbentuk relasi eksplisit [:SIMILAR_TO {score}] pada graf terproyeksi)", fontsize=11, fontweight="bold")
ax2.legend(loc="lower left", frameon=True, facecolor="white")
ax2.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "kondisi_sebelum_sesudah_similarity.png"), dpi=300)
plt.close()
print("Saved kondisi_sebelum_sesudah_similarity.png")
