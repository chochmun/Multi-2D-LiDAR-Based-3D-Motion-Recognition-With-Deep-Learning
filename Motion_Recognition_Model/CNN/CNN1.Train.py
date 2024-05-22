import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define file names and epochs
file_names = ['CNN/ex3_5_32000/ex3_5_sun_32000aug.csv', 'CNN/ex3_5_32000/ex3_5_chung_32000aug.csv',
               'CNN/ex3_5_32000/ex3_5_min_32000aug.csv','CNN/ex3_5_32000/ex3_5_jun_32000aug.csv']
optim_what=1 #Adam :1, SGD:2
learning_rate=0.005


epochs_list = [15]#[10, 30, 50, 100, 200, 300]

# CNN model class
class LidarCNN(nn.Module):
    def __init__(self, num_classes):
        super(LidarCNN, self).__init__()
        self.conv1 = nn.Conv1d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 22, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 32 * 22)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Train and evaluate function
def train_and_evaluate(file_name, num_epochs):
    # Load the data
    data_path = f'{file_name}'
    data = pd.read_csv(data_path)
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    print(f"Processing {data_path}, training for {num_epochs} epochs.")

    # Split and scale the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = X_train_scaled.reshape(-1, 3, 90)
    X_test_scaled = X_test_scaled.reshape(-1, 3, 90)

    # Prepare model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LidarCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    if optim_what==1:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optim_what==2:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Training loop
    loss_values = []
    accuracy_values = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss_values.append(running_loss / len(train_loader))

        # Evaluate accuracy on test data
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        accuracy_values.append(accuracy)
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

    # Save the model
    model_save_dir = 'saved_models'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = os.path.join(model_save_dir, f'cnn_{os.path.basename(file_name).split(".")[0]}_model_{num_epochs}.pth')
    torch.save(model.state_dict(), model_save_path)

    # Plotting the results
    plot_loss_accuracy(range(1, num_epochs+1), loss_values, accuracy_values, file_name, num_epochs)

def plot_loss_accuracy(epochs, loss_values, accuracy_values, file_name, num_epochs):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, loss_values, color='tab:red', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(epochs, accuracy_values, color='tab:blue', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()
    plt.title(f'Training Loss and Accuracy over Epochs for {num_epochs}')

    plot_save_dir = 'plots'
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)
    plot_save_path = os.path.join(plot_save_dir, f'loss_accuracy_plot_{os.path.basename(file_name).split(".")[0]}_{num_epochs}.jpg')
    plt.savefig(plot_save_path)
    plt.close(fig)

# Execute training for all files and epochs
for file_name in file_names:
    for num_epochs in epochs_list:
        train_and_evaluate(file_name, num_epochs)