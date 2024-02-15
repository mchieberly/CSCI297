# Malachi Eberly
# Project 1 - Apple Quality Problem

# Data pre-processing
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Pre-process data
df = pd.read_csv('apple_quality.csv')
del df['A_id']
df = df.iloc[0:3999, :]

columns = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']

df['Acidity'] = df['Acidity'].astype("float")
df['Quality'] = df['Quality'].replace({'good':1,'bad':0})

# EDA
print(df.describe())

# Standardization
for col in columns:
    df[col] = zscore(df[col])

# Split Training and Testing
X = df[columns].values
Y = df['Quality'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

# Feed-forward neural network architecture
class FFNet(nn.Module):
    def __init__(self):
        super(FFNet, self).__init__()
        self.fc1 = nn.Linear(len(columns), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create neural network model
model = FFNet()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

# Training and testing tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

# Training loop
n_epochs = 200
for epoch in range(n_epochs):
    inputs = Variable(X_train)
    labels = Variable(Y_train)

    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print('Epoch {}/{} - Loss: {:.4f}'.format(epoch + 1, n_epochs, loss.item()))

# Test
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    total += Y_test.size(0)
    correct += (predicted == Y_test).sum().item()

print('Test Accuracy: {:.2f}%'.format(100 * correct / total))