import os
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch 
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from sklearn.utils.class_weight import compute_class_weight

file_path = '/Users/kangaroo/Documents/Sydney_Uni_Study/2024/Honours/Semester_2/Data/DILIST_DATA.xlsx' 
graph_data_path = 'DILI_molecule_graphs.pt'
df_01 = pd.read_excel(file_path)
graph_data_list = torch.load(graph_data_path)

for graph, (_, row) in zip(graph_data_list, df_01.iterrows()):
    # Convert descriptors to a tensor
    descriptors = torch.tensor([row['LogP'], row['Molecular_Weight'], row['TPSA'], row['HBD'], row['HBA'], row['num_rot_bonds']], dtype=torch.float)
    
    # Convert DILI annotation to a tensor
    DILI_label = torch.tensor([row['DILIst_Classification']], dtype=torch.long)  # Use torch.long as labels are categorical

    # Assign the descriptors as a graph-level attribute
    graph.descriptor = descriptors

    # Add DILI label as the target for the graph
    graph.y = DILI_label

# Ensuring the data is ready to go into a GCNN model
# Seed for reproducibility
random.seed(0)

# Shuffle the data
random.shuffle(graph_data_list)

# Determine split sizes
train_size = int(0.7 * len(graph_data_list))
val_size = int(0.15 * len(graph_data_list))
test_size = len(graph_data_list) - train_size - val_size

# Split the data
train_data = graph_data_list[:train_size]
val_data = graph_data_list[train_size:train_size + val_size]
test_data = graph_data_list[train_size + val_size:]

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Core functions
def train(model, loader):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad() # given that by default, gradients accmulate when weights are being updated
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward() # section where the model learns by adjusting weights based on the error
        optimizer.step() # update model parameters
        total_loss += loss.item()
    return total_loss / len(loader) # Computes average loss of all batches 

def train_imbalance(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.view(-1)) 
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval() # in eval mode, the model doesn't perform dropout but uses the statistics for batch normalization
    correct = 0
    for batch in loader:
        out = model(batch)
        pred = out.argmax(dim=1)
        correct += pred.eq(batch.y).sum().item()
    return correct / len(loader.dataset) # Used to compute the accuracy

def evaluate_imbalance(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            pred = out.argmax(dim=1)
            loss = criterion(out, batch.y.view(-1))
            total_loss += loss.item()
            correct += pred.eq(batch.y.view(-1)).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

# Simple GCNN
class GCNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch # Extract node feature, edge information, and batch information from input data

        # Graph Convolutional Layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training) # Ensures dropout only occurs during training, not testing
        x = F.relu(self.conv2(x, edge_index))

        # Global Pooling Layer
        x = global_mean_pool(x, batch)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1) # Apply softmax function to the output

# Set input, hidden, and output dimensions
input_dim = graph_data_list[0].x.size(1)  # Number of node features; assumpting that all nodes have the same feature number
hidden_dim = 64  # Number of hidden neurons
output_dim = 2  # Two classes: DILI positive and negative

model = GCNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) # weight_decay: regularization parameter
criterion = torch.nn.CrossEntropyLoss() # Calculates the loss between softmax outputs of the model and true distribution

# Training loop
epochs = 50
for epoch in range(epochs):
    train_loss = train(model, train_loader)
    val_acc = evaluate(model, val_loader)
    print(f'Epoch {epoch+1:03d}, Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}')
test_acc = evaluate(model, test_loader)
print(f'Test Accuracy: {test_acc:.4f}')


# Attempt with GCNN model with batch normalization
class EnhancedGCNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EnhancedGCNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First Graph Convolutional Layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Graph Convolutional Layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Third Graph Convolutional Layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Global Pooling Layer
        x = global_mean_pool(x, batch)

        # Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

# Set input, hidden, and output dimensions
input_dim = graph_data_list[0].x.size(1)  # Number of node features
hidden_dim = 128  # Increased number of hidden units
output_dim = 2    # Two classes: DILI positive and negative

model_02 = EnhancedGCNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# Extract labels from the graph data list
labels = [data.y.item() for data in graph_data_list]

# Convert to numpy array
labels = np.array(labels)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Move to the same device as the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = class_weights.to(device)

# Adjust loss function to include these weights
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)  # Fine-tuning learning rate and weight decay

# Training loop remains the same
epochs = 100  # Increase the number of epochs to allow the model to converge
for epoch in range(epochs):
    train_loss = train_imbalance(model_02, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate_imbalance(model_02, val_loader, criterion)
    print(f'Epoch {epoch+1:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

test_acc = evaluate_imbalance(model_02, test_loader, criterion)
print(test_acc)
