import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
import winsound
import time
import random
import yaml
import matplotlib.pyplot as plt

#함수
def predict_labels(model, predict_data_loader):
    model.eval()
    predicted_labels = []
    with torch.no_grad():
        for inputs in predict_data_loader:
            outputs = model(inputs[0])
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.extend(predicted.tolist())
    return predicted_labels

#Yaml데이터 추출 및 초기화================================================================
with open('MLP/parameters.yaml', 'r', encoding='utf-8') as file:
    yaml_data = yaml.safe_load(file)
    # Extract parameters from the configuration
    debugging_parameters = yaml_data['debugging_parameters']
    hyperparameters = yaml_data['hyperparameters']
    file_paths = yaml_data['file_paths']
    train_csv_files=file_paths['train_csv_files']
    predict_csv_files=file_paths['predict_csv_files']
    cases = yaml_data['cases']
    index_to_name=yaml_data['index_to_name']
    optimizers_dict=yaml_data['optimizers_dict']
    loss_functions_dict=yaml_data['loss_functions_dict']
    activation_functions_dict=yaml_data['activation_functions_dict']
    # Extract parameters from debugging_parameters
    wrong_window = debugging_parameters['wrong_window']
    see_all_epochs = debugging_parameters['see_all_epochs']
    beep_level = debugging_parameters['beep_level']
    num_random_times = debugging_parameters['num_random_times']
    # Extract parameters from hyperparameters
    loss_is_values=hyperparameters['loss_is_values']
    hidden_dims_array = hyperparameters['hidden_dims_array']
    momentum_num = hyperparameters['momentum_num']
    dropout_rates = hyperparameters['dropout_rates']
    num_epoch_values = hyperparameters['num_epoch_values']
    learning_rates = hyperparameters['learning_rates']
    weight_decay_values = hyperparameters['weight_decay_values']
    optim_is_values = hyperparameters['optim_is_values']
    early_stop_accuracies = hyperparameters['early_stop_accuracies']
    batch_size_values = hyperparameters['batch_size_values']
    BN_use_choice = hyperparameters['BN_use_choice']
    activation_list = hyperparameters['activation_list']
    loss_threshold = hyperparameters['loss_threshold']
    max_Loss_small_change_count = hyperparameters['max_Loss_small_change_count']

#파일 경로 직접 설정==============================================================
train_csv_files = train_csv_files['Ex3_5_Filter_Aug_files']
predict_csv_files = predict_csv_files['Ex0_5_Filter_files']
#학습 케이스 직접 설정=============================================================
cases = cases['case3to1']
#cases= [(3, [1, 2, 4], 3)]
high_accuracy_count = 0
prediction_accuracies= []
best_accuracy=0
batch_size_predict=64 #실제 예측에서는 batch사이즈가 1이어야 한다. #그래서 batch스케일링도 하면안된다.
hidden_dims_indexs=list(range(len(hidden_dims_array))) #

#======================Parameters_Search_loop==================
for j in range(num_random_times):
    loss_is=np.random.choice(loss_is_values)
    optim_is=np.random.choice(optim_is_values)
    dropout_rate = np.random.choice(dropout_rates)                  
    num_epochs = np.random.choice(num_epoch_values)
    learning_rate = np.random.choice(learning_rates)
    weight_decay = np.random.choice(weight_decay_values)
    early_stop_accuracy = np.random.choice(early_stop_accuracies)
    batch_size = int(np.random.choice(batch_size_values))
    activation_is = np.random.choice(activation_list)
    Loss_small_change_count=0
    BN_use=np.random.choice(BN_use_choice)
    hidden_dim = hidden_dims_array[np.random.choice(hidden_dims_indexs)]     

    print(f"<Rand_times:{j+1}> -- ")
    print("loss_is:", loss_functions_dict[loss_is], ", optim:", optimizers_dict[optim_is],", Activation:",activation_functions_dict[activation_is])
    print("epochs:", num_epochs, ", lr:", learning_rate, ", weight_decay:", weight_decay, ", dropout_rate:", dropout_rate)
    print("early_stop:", early_stop_accuracy, ", batch_size:", batch_size, ", loss_threshold:", loss_threshold)
    print("hidden_dim:", hidden_dim, ", BatchNorm:", BN_use, ", scaler: Standardscaler")
    print('-' * 25)
    #========================== MLP ==============================
    class Advanced_MLP_Model(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, activation_func, BN_use=True):
            super(Advanced_MLP_Model, self).__init__()
            if activation_func == 1:
                self.activation = nn.ReLU()
            elif activation_func == 2:
                self.activation = nn.Sigmoid()
            elif activation_func == 3:
                self.activation = nn.Tanh()
            self.BN_use = BN_use

            self.layers = nn.ModuleList()
            prev_dim = input_dim
            for dim in hidden_dim: #ex [ 90, 60,30]
                self.layers.append(nn.Linear(prev_dim, dim))
                if BN_use:
                    self.layers.append(nn.BatchNorm1d(dim))
                self.layers.append(self.activation)
                self.layers.append(nn.Dropout(p=dropout_rate))
                prev_dim = dim
            self.layers.append(nn.Linear(prev_dim, output_dim))

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
        
    #=========================Case_loop==============================
    #       num_train_files= 몇명골라서 학습할건지, train_file에서의 인덱스를 담고있는 어레이
                                                        #Predict할 인덱스 하나
    for case_index, (num_train_files, train_file_index_array, pred_file_index) in enumerate(cases, start=1):
        previous_loss=None

        train_names = [index_to_name[index] for index in train_file_index_array]
        print(f"Case {case_index}: {train_names} {num_train_files}명 학습 -> {index_to_name[pred_file_index]} 예측")
        
        #train_file_index_array에따라 학습 파일 선택하기
        selected_train_files = [train_csv_files[i - 1] for i in train_file_index_array]
        #pred_file_index에 따라 예측 파일 선택하기
        predict_data_path = predict_csv_files[pred_file_index - 1]
        #Debug : 파일을 잘 선택했는가
        print("selected_train_files :",selected_train_files,"\npredict_data_path:",predict_data_path)
        dfs=[]
        for i,file in enumerate(selected_train_files):
            df=pd.read_csv(file)
            dfs.append(df)
        data=pd.concat(dfs, axis=0, ignore_index=True)

        # Encode labels as consecutive integers starting from 0
        label_encoder = LabelEncoder()
        y_data = label_encoder.fit_transform(data.iloc[:, 0].values)  # Label column
        x_data = data.iloc[:, 1:].values  # Features are the remaining columnss are in the first
        output_dim = len(label_encoder.classes_)  # Number of unique classes

        # Normalize features
        scaler = StandardScaler()
        x_data = scaler.fit_transform(x_data)

        # Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

        # Convert to PyTorch tensors
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # DataLoader
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # Model instantiation
        input_dim = x_train.shape[1]
        
        #모델 선언
        model = Advanced_MLP_Model(input_dim, hidden_dim, output_dim,dropout_rate,activation_is,BN_use)
        # 손실함수 선택
        if loss_is == 1:
            criterion = nn.CrossEntropyLoss()
        elif loss_is == 2:
            criterion = nn.MSELoss()
        else:
            print("Invalid value for loss_is")
        # 옵티마이저 선택
        if optim_is == 1:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optim_is == 2:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_num, weight_decay=weight_decay)
        elif optim_is == 3:
            optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optim_is == 4:
            optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            print("Invalid value for optim_is")
        
#------------------Train & Test-------------------------Train & Test -------------------Train & Test---------
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            #============================== Train ==============================
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            #======= Loss에따른 조기 종료 판단 =======
            if previous_loss is not None:
                loss_change = abs(train_loss - previous_loss)/len(train_loader)
                if loss_change <= loss_threshold:
                    Loss_small_change_count += 1
                else:
                    Loss_small_change_count = 0

                if Loss_small_change_count >= max_Loss_small_change_count:
                    print(f"Early stopping at epoch {epoch+1} due to loss change {loss_threshold} low- {max_Loss_small_change_count} times.")
                    break
            previous_loss = train_loss
            #========================== Test============================
            model.eval()
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total_samples += labels.size(0)
                    total_correct += (predicted == labels).sum().item()
            test_accuracy = 100 * total_correct / total_samples
            #Dubug용 : 일반적으로 epoch마다 트레이닝 상태 출력
            if see_all_epochs==1:
                print(f'Epoch {epoch+1}, Training Loss: {train_loss/len(train_loader):.4f}, Test(Trianing) Accuracy: {test_accuracy:.2f}%')

            #=========== Test Accuracy 에따른 조기 종료 판단 =========
            if test_accuracy >= early_stop_accuracy:
                high_accuracy_count += 1
                if high_accuracy_count >= 3:
                    print(f"Early stopping at epoch {epoch+1} due to test accuracy reaching {early_stop_accuracy}% three times.")
                    high_accuracy_count = 0
                    break
            else:
                high_accuracy_count = 0
        #Dubug용 : 각각 epoch를 무시하고, 마지막epoch만 보기
        if see_all_epochs==0:
            print(f'Total Epoch {epoch+1}, Training Loss: {train_loss/len(train_loader):.4f}, Test(Trianing) Accuracy: {test_accuracy:.2f}%')

#-------------------PREDICT-------------------------PREDICT  ---------------PREDICT------------------------------------------------------------------        # Load the test data
        predict_data = pd.read_csv(predict_data_path)

        predict_data_label_encoder = LabelEncoder()
        y_predict = predict_data_label_encoder.fit_transform(predict_data.iloc[:, 0].values)


        # Preprocess the test data
        x_predict = predict_data.iloc[:, 1:].values  # Features are the remaining columns
        x_predict_data = scaler.fit_transform(x_predict)#스탠다드 스케일 적용
        x_predict_data_tensor = torch.tensor(x_predict_data, dtype=torch.float32)

        # DataLoader for the test data
        predict_data_set = TensorDataset(x_predict_data_tensor)
        predict_data_loader = DataLoader(predict_data_set, batch_size=batch_size_predict, shuffle=False)

        # Predict labels for the test data
        predicted_labels = predict_labels(model, predict_data_loader)

        # Decode the predicted labels
        predicted_labels_decoded = label_encoder.inverse_transform(predicted_labels)

        # Add the predicted labels to the test data
        predict_data['predicted_label'] = predicted_labels_decoded

        # Calculate accuracy
        from sklearn.metrics import accuracy_score
        predict_data_accuracy = accuracy_score(y_predict, predicted_labels)
        predict_data_accuracy = round(predict_data_accuracy * 100, 2) 
        prediction_accuracies.append(predict_data_accuracy)
        
        print("Prediction Accuracy: {:.2f}%".format(predict_data_accuracy))

        #=======================Debug: 틀린 Label 보여주기=======================
        #===========wrong_window 값만큼 보여준다==============
        incorrect_indices = [i for i in range(len(predicted_labels)) if predicted_labels[i] != y_predict[i]]
        random_incorrect_indices = random.sample(incorrect_indices, min(wrong_window, len(incorrect_indices)))
        #랜덤선택된 틀린라벨 출력
        for i in random_incorrect_indices:
            print(f"Predicted label: {predicted_labels[i]}, Actual label: {y_predict[i]}")
        print('-' * 50)
    #------------------------------Case 종료---------------------------------------
    #======Case에따른 예측 평균=======
    average_accuracy = np.mean(prediction_accuracies)
    print("Average Prediction Accuracy: {:.2f}%\n".format(average_accuracy))
    #======Case에따른 Best Accuracy========
    if average_accuracy > best_accuracy:  #Best Accuracy인지 확인
        best_accuracy = average_accuracy #Best_parameters 갱신
        best_hyperparameters = {
            'hidden_dims': hidden_dim,
            'dropout_rate': dropout_rate,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'optimizer': optimizers_dict[optim_is],
            'loss_func': loss_functions_dict[loss_is],
            'activation': activation_functions_dict[activation_is],
            'early_stop': f"{early_stop_accuracy}%",
            'batch_size': batch_size,
            'Batch_use': BN_use,
            'Scaler': "Standard sclaer"
        }
    if best_accuracy>beep_level: #Debug: beep_level을 설정했을때 그거보다 높으면 울리기
        winsound.Beep(1000, 1000)
    print("Best Hyperparameters:",best_hyperparameters)
    print("Best Accuracy:", best_accuracy)
    prediction_accuracies = []#예측어큐리시 초기화

print("optimizer:", optimizers_dict[optim_is])
print("optimizer:",loss_functions_dict[loss_is])

winsound.Beep(1000, 1000)
time.sleep(1)
winsound.Beep(1000, 1000)
time.sleep(1)
winsound.Beep(1000, 1000)