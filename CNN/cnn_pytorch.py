import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

batch_size = 64
n_epochs = 10
learning_rate = 0.0001

X_train = np.load("./data/cnn_data/X_train_windows.npy")
y_train = np.load("./data/cnn_data/y_train_windows.npy")
X_test = np.load("./data/cnn_data/X_test_windows.npy")
y_test = np.load("./data/cnn_data/y_test_windows.npy")

X_train = X_train.reshape(-1, 1, 20, 11)
X_test = X_test.reshape(-1, 1, 20, 11)

print(X_train.shape, X_test.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=128*2*1, out_features=128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # x shape expected: (batch_size, 1, 20, 11)

        # Block 1
        x = self.conv1(x)              # -> (batch_size, 32, 20, 11)
        x = F.relu(x)
        x = self.pool1(x)             # -> (batch_size, 32, 10, 5)

        # Block 2
        x = self.conv2(x)             # -> (batch_size, 64, 10, 5)
        x = F.relu(x)
        x = self.pool2(x)             # -> (batch_size, 64, 5, 2)

        # Block 3
        x = self.conv3(x)             # -> (batch_size, 128, 5, 2)
        x = F.relu(x)
        x = self.pool3(x)             # -> (batch_size, 128, 2, 1)

        # Flatten
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = F.relu(x) 
        x = self.fc2(x) # -> (batch_size, 2)
        
        return x

model = CNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    total_loss = 0.0
    for i in range(0, len(X_train), batch_size):
        X_batch = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32, device=device)
        y_batch = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32, device=device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    total_loss = total_loss + loss.item()
    
    print(f"Epoch: {epoch}, Loss: {total_loss}")
    
torch.save(model.state_dict(), "cnn_model.pth")

test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
y_pred_list = []

with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch)
        # Move back to CPU, convert to NumPy
        y_pred_list.append(preds.cpu().numpy())

# Concatenate predictions for the entire test set
y_pred_all = np.concatenate(y_pred_list, axis=0)
np.save("./data/cnn_data/y_pred_pytorch_cnn.npy", y_pred_all)
print("Saved predictions with shape:", y_pred_all.shape)
    