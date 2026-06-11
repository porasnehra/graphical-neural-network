import os
import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys

# Ensure Python can find our custom layer
sys.path.insert(0, str(Path(__file__).parent))
from gnn_model import SimpleGCNLayer

app = FastAPI(title="Graph Neural Network (GNN) Anti-Money Laundering API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path("gnn_backend/results/gnn_model.keras")
DATA_PATH = Path("gnn_backend/data/synthetic_graph.json")
model = None

try:
    if MODEL_PATH.exists():
        # Must pass custom_objects so Keras knows how to load our custom GCN layer
        model = tf.keras.models.load_model(str(MODEL_PATH), custom_objects={'SimpleGCNLayer': SimpleGCNLayer})
        print(f"Loaded TensorFlow GNN model from {MODEL_PATH}")
    else:
        print("Warning: GNN model not found. Please run train_gnn.py to train the network.")
except Exception as e:
    print(f"Error loading model: {e}")

def prepare_graph_data():
    """Load graph JSON and convert to tensors for GNN model."""
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        
    nodes = data["nodes"]
    links = data["links"]
    num_nodes = len(nodes)
    num_features = len(nodes[0]["features"])
    
    X = np.zeros((num_nodes, num_features), dtype=np.float32)
    node_id_to_idx = {}
    for i, node in enumerate(nodes):
        node_id_to_idx[node["id"]] = i
        X[i] = node["features"]
        
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for link in links:
        src = node_id_to_idx[link["source"]]
        dst = node_id_to_idx[link["target"]]
        A[src, dst] = 1.0
        A[dst, src] = 1.0
        
    np.fill_diagonal(A, 1.0)
    rowsum = A.sum(axis=1)
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D = np.diag(d_inv_sqrt)
    A_norm = D.dot(A).dot(D)
    
    return np.expand_dims(X, axis=0), np.expand_dims(A_norm, axis=0), data

@app.get("/")
def read_root():
    return {"status": "Active", "message": "GNN Backend is running. Frontend dev: hit /api/graph via GET to get graph JSON."}

@app.get("/api/graph")
def get_graph_predictions():
    """Returns graph topology with predictions for frontend visualization."""
    if model is None:
        raise HTTPException(status_code=503, detail="GNN Model not trained. Run train_gnn.py first.")
        
    if not DATA_PATH.exists():
        raise HTTPException(status_code=500, detail="Graph data missing. Run graph_data_generator.py")
        
    X_batch, A_batch, original_data = prepare_graph_data()
    
    preds = model.predict([X_batch, A_batch], verbose=0)[0]
    
    # Attach predictions back to the JSON payload
    for i, node in enumerate(original_data["nodes"]):
        node["mule_probability"] = round(float(preds[i][0]), 4)
        node["is_flagged"] = bool(preds[i][0] > 0.5)
        
    return original_data

if __name__ == "__main__":
    import uvicorn
    # Port 8001 so it doesn't conflict with your tf_backend API
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
