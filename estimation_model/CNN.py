import numpy as np
import torch
import os
import glob
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class MotionDataset(Dataset):
    def __init__(self, root_dir, folder_names=None, max_length=50, min_length=30):
        self.root_dir = root_dir
        self.folder_names = folder_names if isinstance(folder_names, list) else [folder_names]
        self.max_length = max_length
        self.min_length = min_length
        self.labels = []
        self.indices = []
        self.features = []
        self.sequences = []
        self.lengths = []
        self.folder_names_list = []

        self.load_all_csv_files()
        self.combine_features()
        self.filter_sequences()
        self.pad_sequences()

    def collect_csv_files(self):
        csv_files = []
        for folder_name in self.folder_names:
            if folder_name is not None:
                search_path = os.path.join(self.root_dir, folder_name, '*.csv')
            else:
                search_path = os.path.join(self.root_dir, '**', '*.csv')
            csv_files.extend(glob.glob(search_path, recursive=True))
        return csv_files

    def load_csv_file(self, file, global_idx_offset):
        data = np.loadtxt(file, delimiter=',', skiprows=1)
        labels = data[:, 0].astype(int)
        
        # 라벨 5("run")을 무시하고 나머지 라벨을 재매핑합니다.
        #labels = labels[labels != 5]  # 라벨 5 제거
        labels = np.where(labels > 5, labels - 1, labels)  # 6 이상 라벨을 1씩 줄임

        file_indices = data[:, 1].astype(int)
        indices = file_indices + global_idx_offset
        features = data[:, 3:]

        # 재매핑된 라벨과 관련된 시퀀스만 남깁니다.
        valid_indices = (data[:, 0] != 5)
        labels = labels[valid_indices]
        indices = indices[valid_indices]
        features = features[valid_indices]

        folder_name = os.path.basename(os.path.dirname(file))

        self.labels.extend(labels)
        self.indices.extend(indices)
        self.features.append(features)
        num_sequences = file_indices.max() + 1
        self.folder_names_list.extend([folder_name] * num_sequences)

        return indices.max() + 1

    def combine_features(self):
        self.labels = np.array(self.labels)
        self.indices = np.array(self.indices)
        self.features = np.vstack(self.features)

        unique_indices = np.unique(self.indices)
        for idx in unique_indices:
            seq_features = self.features[self.indices == idx]
            self.sequences.append(torch.tensor(seq_features, dtype=torch.float32))
            self.lengths.append(seq_features.shape[0])

    def pad_sequences(self):
        padded_sequences = []
        for seq in self.sequences:
            if len(seq) < self.max_length:
                pad_size = self.max_length - len(seq)
                padding = torch.zeros((pad_size, seq.size(1)))
                padded_seq = torch.cat((seq, padding), dim=0)
            else:
                padded_seq = seq[:self.max_length]
            padded_sequences.append(padded_seq)
        self.sequences = padded_sequences
        self.lengths = [min(length, self.max_length) for length in self.lengths]

    def filter_sequences(self):
        filtered_sequences = []
        filtered_lengths = []
        filtered_labels = []
        filtered_indices = []
        filtered_folder_names = []
        for i in range(len(self.sequences)):
            if self.min_length <= self.lengths[i] <= self.max_length:
                filtered_sequences.append(self.sequences[i])
                filtered_lengths.append(self.lengths[i])
                filtered_labels.append(self.labels[self.indices == i][0])
                filtered_indices.append(self.indices[i])
                filtered_folder_names.append(self.folder_names_list[i])
        
        self.sequences = filtered_sequences
        self.lengths = filtered_lengths
        self.labels = np.array(filtered_labels)
        self.indices = np.array(filtered_indices)
        self.folder_names_list = filtered_folder_names

    def aggregate_labels(self):
        unique_indices = np.unique(self.indices)
        aggregated_labels = []
        for idx in unique_indices:
            aggregated_labels.append(self.labels[self.indices == idx][0])
        self.labels = np.array(aggregated_labels)

        assert len(self.labels) == len(self.sequences), "Label length does not match sequence"

    def load_all_csv_files(self):
        csv_files = self.collect_csv_files()
        global_idx_offset = 0
        for file in csv_files:
            global_idx_offset = self.load_csv_file(file, global_idx_offset)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        label = self.labels[idx]
        length = self.lengths[idx]
        feature = self.sequences[idx]
        folder_name = self.folder_names_list[idx]
        return torch.tensor(label), torch.tensor(length), feature, folder_name

    def print_sequence_lengths(self):
        for idx, length in enumerate(self.lengths):
            print(f"Sequence {idx} length: {length}, Folder: {self.folder_names_list[idx]}")

    def get_final_dataframe(self):
        import pandas as pd

        sequences = [seq.numpy() for seq in self.sequences]
        
        data = {
            'Label': self.labels,
            'Sequence': sequences
        }
        
        df = pd.DataFrame(data)
        return df
    def save_random_sequence_by_label(self, label_value, save_dir='.', file_name=None):
        # 특정 라벨에 해당하는 인덱스 필터링
        matching_indices = [i for i, label in enumerate(self.labels) if label == label_value]
        
        if len(matching_indices) == 0:
            print(f"No sequences found for label: {label_value}")
            return

        # 랜덤으로 시퀀스 선택
        random_idx = np.random.choice(matching_indices)
        selected_sequence = self.sequences[random_idx].numpy()
        
        # 파일 이름 생성
        if file_name is None:
            file_name = f'label_{label_value}_random.csv'
        
        # CSV로 저장
        save_path = os.path.join(save_dir, file_name)
        df = pd.DataFrame(selected_sequence)
        df.to_csv(save_path, index=False)

        print(f"Sequence with label {label_value} saved to {save_path}")
    def save_random_sequences_all_labels(self, save_dir='.', file_name='random_sequences_all_labels.csv'):
        all_sequences = []
        all_labels = []

        for label_value in range(10):  # 라벨 값이 0부터 9까지라고 가정
            matching_indices = [i for i, label in enumerate(self.labels) if label == label_value]
            
            if len(matching_indices) == 0:
                print(f"No sequences found for label: {label_value}")
                continue

            # 랜덤으로 시퀀스 선택
            random_idx = np.random.choice(matching_indices)
            selected_sequence = self.sequences[random_idx].numpy()

            all_sequences.append(selected_sequence)
            all_labels.append(np.full((selected_sequence.shape[0], 1), label_value))

        # 라벨과 시퀀스를 하나의 데이터프레임으로 결합
        combined_sequences = np.vstack(all_sequences)
        combined_labels = np.vstack(all_labels)

        # CSV로 저장
        df = pd.DataFrame(np.hstack([combined_labels, combined_sequences]))
        df.to_csv(os.path.join(save_dir, file_name), index=False, header=False)

        print(f"Random sequences from all labels saved to {file_name}")
from collections import OrderedDict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.callbacks import LambdaCallback
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Flatten, Reshape
from tensorflow.keras.models import load_model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
class CNNModel:
    def __init__(self, input_shape, device, model_name=None):
        self.device = device
        # Load an existing model if a model_name is provided, otherwise create a new one
        if model_name and os.path.exists(model_name):
            print(f"Loading existing model from {model_name}")
            self.model = load_model(model_name)
        else:
            self.model = self.build_model(input_shape)

    def build_model(self, input_shape):
        model = Sequential([
            Conv1D(filters=16, kernel_size=2, activation='relu', input_shape=input_shape),  # 작은 커널 크기
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            Conv1D(filters=32, kernel_size=2, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            Conv1D(filters=64, kernel_size=2, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Flatten(),
            
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.5),
            
            Dense(10, activation='softmax')  # 10개의 클래스
        ])
        optimizer = Adam(learning_rate=0.0005)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model


    def train_and_evaluate_model(self, X_train, y_train, epochs=10, batch_size=32, model_name='default.h5'):
        # PyTorch 텐서를 NumPy 배열로 변환
        X_train = X_train.detach().cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train
        y_train = y_train.detach().cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train

        # 학습 데이터 70%, 검증 데이터 30%로 분할
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

        print(self.model.summary())

        # Model checkpoint to save the model when val_accuracy > 0.6 and when val_accuracy improves
        checkpoint = ModelCheckpoint(
            filepath=model_name, 
            monitor='val_accuracy', 
            mode='max', 
            save_best_only=True, 
            save_weights_only=False, 
            verbose=1,
            save_freq='epoch'
        )

        # 모델 학습: validation_data로 나눠진 30% 데이터 사용
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1,
                    callbacks=[LambdaCallback(on_epoch_end=lambda epoch, logs: 
                                                print(f"Epoch {epoch + 1}: Training Accuracy = {logs['accuracy']:.2f}, Validation Accuracy = {logs['val_accuracy']:.2f}")),
                                checkpoint])
        
        #y_pred = self.model.predict(X_test)
        #self.print_accuracy_per_label(y_test, y_pred)

    def predict_and_evaluate(self, X_test, y_test, model_path, label_mapping):
        class_labels = [label_mapping[i] for i in range(len(label_mapping))]

        model = load_model(model_path)
        print(f"Loaded model from {model_path}")
        
        # PyTorch 텐서를 NumPy 배열로 변환
        X_test = X_test.detach().cpu().numpy() if isinstance(X_test, torch.Tensor) else X_test
        y_test = y_test.detach().cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test

        y_pred = model.predict(X_test)
        
        # Confusion matrix 계산
        matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        
        # 퍼센티지로 변환
        matrix_percentage = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        
        # 전체 정확도 계산
        overall_accuracy = np.sum(matrix.diagonal()) / np.sum(matrix)

        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix_percentage, annot=True, fmt=".2%", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
        
        # 모델 이름과 전체 정확도를 매트릭스 상단에 표시
        plt.title(f"Confusion Matrix (Percentage)\nModel: {model_path}\nOverall Accuracy: {overall_accuracy:.2%}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

        report = classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=class_labels)
        print("Classification Report:\n", report)

        accuracy_per_label = matrix.diagonal() / matrix.sum(axis=1)
        for i, acc in enumerate(accuracy_per_label):
            print(f"Label {class_labels[i]}: Accuracy = {acc:.2f}")
        print(f"Overall Accuracy: {overall_accuracy:.2f}")


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    label_mapping = {
        0: "attention",
        1: "walk",
        2: "walk_left_handsup",
        3: "walk_right_handsup",
        4: "wave",
        5: "jump",  # 기존 6번 라벨이 5번으로 변경됨
        6: "squat",  # 기존 7번 라벨이 6번으로 변경됨
        7: "left_hands_updown",  # 기존 8번 라벨이 7번으로 변경됨
        8: "right_hands_updown",  # 기존 9번 라벨이 8번으로 변경됨
        9: "none"  # 기존 10번 라벨이 9번으로 변경됨
    }

    # 클래스 없는 함수들
    def preprocess_data(data):
        X = np.array(data['Sequence'].tolist())
        y = np.array(data['Label'])
        y = np.eye(10)[y]
        return X, y
    def get_unique_labels(y_train):
        class_indices = np.argmax(y_train)
        unique_classes = np.unique(class_indices)
        return unique_classes


    # 데이터셋 생성 및 전처리
    usr_name = ['min', 'jun', 'mun', 'sun']
    root_directory = 'csv_files\\240808_lidar3_sequence_label10'
    
    # 3명의 데이터를 학습 데이터로 사용
    train_set = MotionDataset(root_directory, folder_names=usr_name[0:3])
    
    # 4번째 사람의 데이터를 가져온 후, 절반을 학습에, 절반을 테스트에 사용
    fourth_set = MotionDataset(root_directory, folder_names=usr_name[3])

    # 데이터를 DataFrame으로 변환
    train_set = train_set.get_final_dataframe()
    fourth_set = fourth_set.get_final_dataframe()

    # 4번째 사람의 데이터를 50%로 분할 (절반 학습, 절반 테스트)
    from sklearn.model_selection import train_test_split

    fourth_train, fourth_test = train_test_split(fourth_set, test_size=0.5, random_state=42)

    # 기존 학습 데이터와 4번째 사람의 절반 학습 데이터를 합침
    combined_train = pd.concat([train_set, fourth_train])

    # 최종 학습 및 테스트 데이터를 numpy로 변환
    X_train, y_train = preprocess_data(combined_train)
    X_test, y_test = preprocess_data(fourth_test)

    # 데이터 형태 맞추기
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    # RNNModel 생성
    rnn_model = CNNModel(input_shape=(X_train.shape[1], X_train.shape[2]), device=device)

    # 모델 학습
    rnn_model.train_and_evaluate_model(X_train, y_train, epochs=100, model_name='motion_cnn_3_5to0_5.h5')

    # 모델을 사용해 최종 예측 및 평가
    rnn_model.predict_and_evaluate(X_train, y_train, model_path='motion_cnn_3_5to0_5.h5', label_mapping=label_mapping)