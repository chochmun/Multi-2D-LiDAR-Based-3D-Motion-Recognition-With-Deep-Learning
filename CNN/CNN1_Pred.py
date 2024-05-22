import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
# Define the model class once
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
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x.view(-1, 32 * 22))
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

# List of data files and epochs

data_files = ['sun.csv', 'chung.csv', 'min.csv','jun.csv']


epochs = [15]#, 30, 50, 100, 200, 300]
dataset_accuracies = {file: [] for file in data_files}

for data_path in data_files:
    # Load the CSV data
    data = pd.read_csv(data_path)
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values

    # Preprocessing
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).reshape(-1, 3, 90)
     
    # Convert to tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    accuracies = []
    for epoch in epochs:
        #model_path = f'cnn_ex{data_path[:-4]}_aug__model_{epoch}.pth'
        model = LidarCNN(len(label_encoder.classes_))

        model_path = f'saved_models\cnn_ex3_5_{data_path[:-4]}_32000aug_model_{epoch}.pth'

        model.load_state_dict(torch.load(model_path))
        model.eval()

        correct = total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracies.append(accuracy)
    
        # Plot and save the accuracy for each model
        """plt.figure(figsize=(10, 5))
        plt.plot(epochs[:len(accuracies)], accuracies)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Accuracy vs Epoch for {data_path[:-4].capitalize()} Data')
        plt.savefig(f'cnn_{data_path[:-4]}__model_accuracy_aug.jpg')
        plt.close()"""
    
    dataset_accuracies[data_path].extend(accuracies)

# Plotting average accuracies per dataset
plt.figure(figsize=(10, 5))
labels = []
print(dataset_accuracies.items())
all_values = []
for values in dataset_accuracies.values():
    all_values.extend(values)
# 평균을 계산
average = sum(all_values) / len(all_values) if all_values else 0
print(f"Average: {average:.2f}")

for data_path, acc in dataset_accuracies.items():
    avg_accuracy = sum(acc) / len(acc)
    plt.bar(data_path[:-4], avg_accuracy)
    labels.append(data_path[:-4].capitalize())
plt.xlabel('Dataset')
plt.ylabel('Average Accuracy (%)')
plt.title('Average Accuracy per Dataset')
plt.xticks(ticks=range(len(labels)), labels=labels)  # Ensure labels are correctly shown
plt.savefig('average_accuracy_per_dataset_aug.jpg')
plt.show()