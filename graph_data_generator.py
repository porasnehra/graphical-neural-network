import pandas as pd
import numpy as np
import json
import random
from pathlib import Path

def generate_synthetic_graph(num_nodes=500, num_edges=1000):
    print("Generating synthetic graph network...")
    np.random.seed(42)
    random.seed(42)

    # 1. Generate Nodes (Accounts)
    nodes = []
    labels = []
    # Features: [Total_In, Total_Out, Tx_Count, Age, Credit_Score]
    features = []

    # Define topology indices
    mule_indices = set([0, 100, 200, 201, 202])

    for i in range(num_nodes):
        nodes.append(f"ACC_{i:04d}")
        
        if i in mule_indices:
            labels.append(1)
            feat = [
                np.random.uniform(10000, 50000),
                np.random.uniform(9000, 49000),
                np.random.randint(50, 200),
                np.random.randint(18, 25),
                np.random.randint(300, 500)
            ]
        else:
            labels.append(0) # Normal
            feat = [
                np.random.uniform(100, 5000),
                np.random.uniform(50, 4000),
                np.random.randint(1, 20),
                np.random.randint(25, 70),
                np.random.randint(600, 850)
            ]
        features.append(feat)

    # Generate Edges
    edges = []
    
    # Topology: Star 1
    for i in range(1, 15):
        edges.append({"source": f"ACC_{i:04d}", "target": f"ACC_0000", "weight": np.random.uniform(500, 2000)})
        
    # Topology: Star 2
    for i in range(101, 115):
        edges.append({"source": f"ACC_{i:04d}", "target": f"ACC_0100", "weight": np.random.uniform(500, 2000)})
        
    # Topology: Bipartite
    bipartite_sources = [210, 211, 212]
    bipartite_mules = [200, 201, 202]
    bipartite_dest = 220
    
    for src in bipartite_sources:
        for mule in bipartite_mules:
            edges.append({"source": f"ACC_{src:04d}", "target": f"ACC_{mule:04d}", "weight": np.random.uniform(1000, 3000)})
    for mule in bipartite_mules:
        edges.append({"source": f"ACC_{mule:04d}", "target": f"ACC_{bipartite_dest:04d}", "weight": np.random.uniform(3000, 9000)})

    # Random noise edges
    for _ in range(num_edges):
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        if src != dst:
            edges.append({"source": f"ACC_{src:04d}", "target": f"ACC_{dst:04d}", "weight": np.random.uniform(10, 500)})

    # Format nodes for JSON
    nodes_list = []
    for i, n in enumerate(nodes):
        nodes_list.append({
            "id": n,
            "label": labels[i],
            "features": features[i]
        })

    graph_data = {
        "nodes": nodes_list,
        "links": edges
    }

    # Ensure data directory exists
    Path("gnn_backend/data").mkdir(parents=True, exist_ok=True)
    
    with open("gnn_backend/data/synthetic_graph.json", "w") as f:
        json.dump(graph_data, f, indent=2)
        
    print(f"Graph generated: {num_nodes} Nodes, {len(edges)} Edges.")
    print("Saved to gnn_backend/data/synthetic_graph.json")

if __name__ == "__main__":
    generate_synthetic_graph()
