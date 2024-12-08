import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# Data preparation
X = np.array([
    [0.8, 0.9, 1], 
    [0.4, 0.5, 2], 
    [0.6, 0.7, 1]
])
y = np.array([1, 3, 2])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
class DifficultyRegressor(nn.Module):
    def __init__(self):
        super(DifficultyRegressor, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = DifficultyRegressor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(500):
    model.train()
    inputs = torch.tensor(X_train, dtype=torch.float32)
    targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Save model
torch.save(model.state_dict(), "models/difficulty_model.pth")
print("Model trained and saved!")
