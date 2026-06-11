import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from gnn_model import create_gnn_model

def load_graph_data():
    with open("gnn_backend/data/synthetic_graph.json", "r") as f:
        data = json.load(f)
        
    nodes = data["nodes"]
    links = data["links"]
    
    num_nodes = len(nodes)
    num_features = len(nodes[0]["features"])
    
    # Extract features and labels
    X = np.zeros((num_nodes, num_features), dtype=np.float32)
    y = np.zeros((num_nodes, 1), dtype=np.float32)
    
    # Create mapping from Node ID to Index
    node_id_to_idx = {}
    for i, node in enumerate(nodes):
        node_id_to_idx[node["id"]] = i
        X[i] = node["features"]
        y[i] = node["label"]
        
    # Build Adjacency Matrix
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    for link in links:
        src = node_id_to_idx[link["source"]]
        dst = node_id_to_idx[link["target"]]
        A[src, dst] = 1.0
        A[dst, src] = 1.0
        
    np.fill_diagonal(A, 1.0)
    
    # Normalize adjacency matrix
    rowsum = A.sum(axis=1)
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D = np.diag(d_inv_sqrt)
    A_norm = D.dot(A).dot(D)
    
    return X, A_norm, y

def train():
    print("Loading graph data...")
    X, A, y = load_graph_data()
    
    X_batch = np.expand_dims(X, axis=0)
    A_batch = np.expand_dims(A, axis=0)
    y_batch = np.expand_dims(y, axis=0)
    
    num_features = X.shape[1]
    
    print("Creating GNN Model...")
    model = create_gnn_model(num_features)
    
    print("Training GNN...")
    model.fit([X_batch, A_batch], y_batch, epochs=150, verbose=1)
    
    # Save the model
    Path("gnn_backend/results").mkdir(parents=True, exist_ok=True)
    model.save("gnn_backend/results/gnn_model.keras")
    print("GNN Model saved to gnn_backend/results/gnn_model.keras")

if __name__ == "__main__":
    train()
