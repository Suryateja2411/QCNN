from flask import Flask, request, jsonify
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import pennylane as qml
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define Quantum Circuit
def quantum_circuit(inputs):
    for i in range(len(inputs)):
        qml.RY(inputs[i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(len(inputs))]

dev = qml.device("default.qubit", wires=6)
qnode = qml.QNode(quantum_circuit, dev, interface='torch')

# Define QCNN Model
class QCNN(nn.Module):
    def __init__(self):
        super(QCNN, self).__init__()
        self.fc1 = nn.Linear(6, 4)
        self.fc2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        q_out = torch.stack([torch.tensor(qnode(x_i), dtype=torch.float32) for x_i in x])
        x = torch.relu(self.fc1(q_out))
        x = self.sigmoid(self.fc2(x))
        return x

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Load and preprocess data
    df = pd.read_csv(filepath)
    features = ['LYVE1', 'REG1B', 'TFF1', 'REG1A', 'plasma_CA19_9', 'creatinine']
    df = df.dropna(subset=features + ['diagnosis'])
    df['diagnosis'] = df['diagnosis'].astype(int)
    X = df[features].values
    y = df['diagnosis'].values
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min())
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min())

    model = QCNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train model
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_predictions = (test_outputs >= 0.5).float()
    
    test_accuracy = (test_predictions == y_test).float().mean().item() * 100
    return jsonify({"accuracy": round(test_accuracy, 2)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)