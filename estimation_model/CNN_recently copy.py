import numpy as np
import torch
import os
import glob
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from collections import OrderedDict
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, MaxPooling1D, LSTM, Dense, Dropout, TimeDistributed,
    Flatten, Reshape, LayerNormalization, MultiHeadAttention, Add, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import time
import tensorflow as tf

class MotionDataset(Dataset):
    def __init__(self, root_dir, folder_names=None):
        self.root_dir = root_dir
        self.folder_names = folder_names if isinstance(folder_names, list) else [folder_names]
        self.labels = []
        self.indices = []
        self.features = []
        self.sequences = []
        self.lengths = []

        self.load_all_csv_files()
        self.combine_features()
        self.aggregate_labels()

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
        data = np.loadtxt(file, delimiter=',')
        labels = data[:, 0].astype(int)

        # 파일 내에서 인덱스를 계산합니다.
        file_indices = data[:, 2].astype(int)
        indices = file_indices + global_idx_offset
        features = data[:, 3:]

        folder_name = os.path.basename(os.path.dirname(file))

        self.labels.extend(labels)
        self.indices.extend(indices)
        self.features.append(features)
        num_sequences = file_indices.max() + 1

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
        return torch.tensor(label), torch.tensor(length), feature

    def print_sequence_lengths(self):
        for idx, length in enumerate(self.lengths):
            print(f"Sequence {idx} length: {length}")

    def get_final_dataframe(self):
        sequences = [seq.numpy() for seq in self.sequences]
        
        data = {
            'Label': self.labels,
            'Sequence': sequences
        }
        
        df = pd.DataFrame(data)
        return df

class CNNModel:
    def __init__(self, input_shape, output_shape, device, model_name=None):
        self.device = device
        # Load an existing model if a model_name is provided, otherwise create a new one
        if model_name and os.path.exists(model_name):
            print(f"Loading existing model from {model_name}")
            self.model = load_model(model_name)
        else:
            self.model = self.build_model(input_shape, output_shape)

    def build_model(self, input_shape, output_shape):
        model = Sequential([
            # Reshape layer to convert input data to 3D
            Reshape((input_shape[0], 1), input_shape=input_shape),
            
            # First Conv1D layer
            Conv1D(64, 3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.4),
            
            # Second Conv1D layer
            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.4),
            
            # Flatten and Dense layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            
            # Output layer
            Dense(output_shape, activation='softmax')  # output_shape is the number of classes
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

        # Model checkpoint to save the model when val_accuracy improves
        checkpoint = ModelCheckpoint(
            filepath=model_name, 
            monitor='val_accuracy', 
            mode='max', 
            save_best_only=True, 
            save_weights_only=False, 
            verbose=1
        )

        # 모델 학습: validation_data로 나눠진 30% 데이터 사용
        self.model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(X_val, y_val), 
            verbose=1,
            callbacks=[
                LambdaCallback(on_epoch_end=lambda epoch, logs: 
                              print(f"Epoch {epoch + 1}: Training Accuracy = {logs['accuracy']:.2f}, Validation Accuracy = {logs['val_accuracy']:.2f}")),
                checkpoint
            ]
        )

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

    def real_time_predict(self, data_stream, label_mapping, window_size, delay=0.05):
        """
        Simulate real-time prediction by feeding one data point at a time every 'delay' seconds.

        Parameters:
        - data_stream: pandas DataFrame with 'Sequence' and 'Label' columns.
        - label_mapping: dict mapping label indices to label names.
        - window_size: number of data points to form a sequence for prediction.
        - delay: time in seconds between data points.
        """
        # Ensure the model is loaded
        if not hasattr(self, 'model'):
            raise ValueError("Model is not loaded.")

        # Initialize the sliding window
        window = []

        for idx, row in data_stream.iterrows():
            start_time = time.time()

            # Extract the new data point
            new_data = row['Sequence']  # Assuming 'Sequence' is a numpy array or list of features

            # Handle multiple features if necessary
            # Here, we assume the model expects one feature per time step
            if isinstance(new_data, np.ndarray) and new_data.ndim > 1:
                # For multiple features, you might want to select specific features or flatten them
                new_feature = new_data[0]  # Example: taking the first feature
            else:
                new_feature = new_data

            # Append the new feature to the window
            window.append(new_feature)

            # Maintain the window size
            if len(window) > window_size:
                window.pop(0)

            # Make a prediction if the window is full
            if len(window) == window_size:
                # Prepare the input data for the model
                input_data = np.array(window).reshape(1, window_size, 1)  # (batch_size, sequence_length, 1)

                # Measure prediction time
                pred_start = time.time()
                prediction = self.model.predict(input_data)
                pred_end = time.time()
                prediction_delay = pred_end - pred_start

                # Determine the predicted label
                predicted_label_idx = np.argmax(prediction)
                predicted_label = label_mapping.get(predicted_label_idx, "Unknown")

                # Print the predicted label and delay
                print(f"Predicted Label: {predicted_label}, Prediction Delay: {prediction_delay:.4f} seconds")
            else:
                # Not enough data to make a prediction yet
                print(f"Insufficient data for prediction. Collected {len(window)}/{window_size} data points.")

            # Wait for the specified delay before processing the next data point
            time.sleep(delay)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    label_mapping = {
        0: "stand",
        1: "none",
        2: "left_hand",
        3: "right_hand",
        4: "wave",
        5: "squat",
        6: "plank",
        7: "too_close",
        8: "jump"
    }

    # Helper functions
    def preprocess_data(data):
        X = np.array(data['Sequence'].tolist())
        y = np.array(data['Label']).astype(int)
        y = np.eye(9)[y]
        return X, y

    def get_unique_labels(y_train):
        class_indices = np.argmax(y_train, axis=1)
        unique_classes = np.unique(class_indices)
        return unique_classes

    # 데이터셋 생성 및 전처리
    usr_names = ['mun', 'mun_test']
    root_directory = 'csv_files\\'

    # 3명의 데이터를 학습 데이터로 사용
    train_set = MotionDataset(root_directory, folder_names=usr_names[0])

    # 4번째 사람의 데이터를 가져온 후, 절반을 학습에, 절반을 테스트에 사용
    test_set = MotionDataset(root_directory, folder_names=usr_names[1])

    # 데이터를 DataFrame으로 변환
    train_set_df = train_set.get_final_dataframe()
    test_set_df = test_set.get_final_dataframe()

    # 최종 학습 및 테스트 데이터를 numpy로 변환
    X_train, y_train = preprocess_data(train_set_df)
    X_test, y_test = preprocess_data(test_set_df)

    # 데이터 형태 맞추기
    # 모델이 (sequence_length, 1) 형태를 기대하므로, 각 시퀀스의 특성 수에 맞게 reshape
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2])

    # CNNModel 생성
    cnn_model = CNNModel(input_shape=(X_train.shape[1],), output_shape=9, device=device, model_name='CNN_4mb3.h5')

    # 모델을 사용해 최종 예측 및 평가
    # 여기서는 기존 모델을 사용하므로 학습 코드는 주석 처리
    # cnn_model.train_and_evaluate_model(X_train, y_train, epochs=30, batch_size=128, model_name='CNN_4mb3.h5')

    # 실시간 예측을 위한 시퀀스 길이 설정
    window_size = X_train.shape[1]  # 모델이 학습된 시퀀스 길이와 일치해야 합니다.

    print("Starting real-time prediction...")
    cnn_model.real_time_predict(
        data_stream=test_set_df,
        label_mapping=label_mapping,
        window_size=window_size,
        delay=0.05  # 0.05초 간격으로 데이터 포인트 입력
    )
