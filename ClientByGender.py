from datetime import datetime
import os
import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tensorflow.python.keras.layers import Dense, Dropout

import argparse

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Constants
SAMPLING_FRACTION = 0.8

fds = pd.read_csv('datasets\data_acs.csv')
X = fds.drop(columns="ESR")
y = fds['ESR']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(input_size):
    model = Sequential()
    model.add(Dense(64, input_dim=input_size, activation='relu'))
    model.add(Dropout(0.5))  # Dropout layer for regularization

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def eval_learning(y_true, y_pred):
    print(y_pred)
    print(y_true)
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, average="micro")
    prec = precision_score(y_true, y_pred, average="micro")
    f1 = f1_score(y_true, y_pred, average="micro")
    return acc, rec, prec, f1

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, gender_distribution, x_train, x_test, y_train, y_test, model):
        self.client_id = client_id
        self.gender_distribution = gender_distribution
        self.model = model

        client_data = fds[fds['SEX'] == gender_distribution[client_id]]
        client_data = client_data.sample(frac=SAMPLING_FRACTION)
        biased_data = fds[fds['SEX'] != gender_distribution[client_id]]

        client_data = pd.concat([biased_data.sample(frac=(1 - SAMPLING_FRACTION)), client_data])
        print(client_data.head(5))
        self.male_samples, self.female_samples = (client_data['SEX'] == 1.0).values,  (client_data['SEX'] == 2.0).values
        
        X, y = client_data.drop(columns="ESR"), client_data['ESR']
        print('X',X)
        print('Label',y)
        #X = np.asarray(X).astype('float32')
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.male_samples, self.female_samples = (x_test['SEX'] == 1.0).values,  (x_test['SEX'] == 2.0).values

        self.y_train, self.y_test = np.asarray(y_train).astype('int8'),  np.asarray(y_test).astype('int8')

        self.x_train, self.x_test, = np.asarray(x_train).astype('float32'),  np.asarray(x_test).astype('float32')
        
    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
         # Create TensorBoard callback
        # log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Train the model on the selected data
        self.model.fit(self.x_train, self.y_train, epochs=5, batch_size=32)
        return self.model.get_weights(), len(self.x_train), {}

       

    def evaluate(self, parameters, config):
        
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        y_pred = self.model.predict(self.x_test).astype('int8')
        print("y_pred",y_pred)
        # male_indices = (self.x_test['SEX'] == 1.0).values
        # female_indices = (self.x_test['SEX'] == 2.0).values
        
        print(self.male_samples)
        male_predictions = y_pred[self.male_samples]
        female_predictions = y_pred[self.female_samples]
        print("femal predictions:",female_predictions)
        print("male predictions : ",male_predictions)
        print("Mean male prediction:", np.mean(male_predictions), "Mean female prediction:", np.mean(female_predictions))
        spd = np.abs(np.mean(male_predictions) - np.mean(female_predictions))

        print(f"SPD: {spd}")

        acc, rec, prec, f1 = eval_learning(self.y_test, y_pred)
        output_dict = {
            "accuracy": accuracy,
            "acc": acc,
            "rec": rec,
            "prec": prec,
            "f1": f1,
            "spd": spd,
        }

        return loss, len(self.x_test), output_dict

def main():
    input_size = x_train.shape[1]
    model = build_model(input_size=input_size)

    client_id_1 = 0
    client_id_2 = 1
    gender_distribution = {client_id_1: 1.0, client_id_2: 2.0}

    parser = argparse.ArgumentParser(description='A simple Python script with command-line arguments.')
    parser.add_argument('--gender', '-g', default="1", help='Biased Gender filter')
    args = parser.parse_args()

    client_id = int(args.gender)

    fl.client.start_client(
        server_address="127.0.0.1:8081",
        client=FlowerClient(client_id, gender_distribution, x_train, x_test, y_train, y_test, model).to_client()
    )

if __name__ == "__main__":
    main()