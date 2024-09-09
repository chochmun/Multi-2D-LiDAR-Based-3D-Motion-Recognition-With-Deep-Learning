import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import yaml
import time

with open('Motion_Recognition_Model/parameters.yaml', 'r', encoding='utf-8') as file:
    yaml_data = yaml.safe_load(file)
    file_paths = yaml_data['file_paths']
    file_paths=file_paths['predict_csv_files']
    predict_csv_files=file_paths['Filter_files']

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

# Define file paths and mappings
index_to_name = {
    0: "junk",
    1: "sunk",
    2: "mink",
    3: "munc"
}

# List of epochs
epochs = [15]  # Add more epochs if needed

dataset_accuracies = {index_to_name[key]: [] for key in [0,1,2,3]} # 인덱스마다 [] 어레이 생성하는 딕셔너리 생성

for index, data_path in enumerate(predict_csv_files):
    dataset_name = index_to_name[index]
    print(dataset_name)
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
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    accuracies = []
    for epoch in epochs:
        model = LidarCNN(len(label_encoder.classes_))
        model_path = f'Motion_Recognition_Model/saved_models/cnn_ex3_5_{dataset_name}_32000aug_model_{epoch}.pth'
        model.load_state_dict(torch.load(model_path))
        model.eval()

        correct = total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print(predicted,inputs)
                time.sleep(0.05)

        accuracy = 100 * correct / total
        accuracies.append(accuracy)
    
    dataset_accuracies[dataset_name].extend(accuracies)

# Plotting average accuracies per dataset
plt.figure(figsize=(10, 5))
labels = []
all_values = []
for values in dataset_accuracies.values():
    all_values.extend(values)
average = sum(all_values) / len(all_values) if all_values else 0
print(f"Average: {average:.2f}")

for dataset_name, acc in dataset_accuracies.items():
    avg_accuracy = sum(acc) / len(acc)
    plt.bar(dataset_name, avg_accuracy)
    labels.append(dataset_name.capitalize())
plt.xlabel('Dataset')
plt.ylabel('Average Accuracy (%)')
plt.title('Average Accuracy per Dataset')
plt.xticks(ticks=range(len(labels)), labels=labels)  # Ensure labels are correctly shown
plt.savefig('average_accuracy_per_dataset_aug.jpg')
plt.show()
