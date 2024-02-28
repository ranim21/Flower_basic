import os
import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load data
fds = pd.read_csv('datasets\data_acs.csv')
X = fds.drop(columns="ESR")
y = fds['ESR']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using Numpy format
train_np = x_train.values
test_np = x_test.values

# Load model
model = Sequential()
model.add(Dense(10, input_dim=train_np.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Adjust for your problem, e.g., softmax for multiclass
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Adjust loss for your problem

# Method for extra learning metrics calculation
def eval_learning(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, average="micro")
    prec = precision_score(y_true, y_pred, average="micro")
    f1 = f1_score(y_true, y_pred, average="micro")
    return acc, rec, prec, f1

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, gender_distribution):
        self.client_id = client_id
        self.gender_distribution = gender_distribution

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)

        # Filter data based on gender distribution
        client_data = fds[fds['SEX'] == self.gender_distribution[self.client_id]]
        client_labels = client_data.pop('ESR')
        print(client_data.head(5))
        print('Gender',self.gender_distribution[self.client_id])
        print('Filter data based on SEX distribution ')
        # Train the model on the selected data
        model.fit(client_data, client_labels, epochs=1, batch_size=32)
        print('Training the model based on the distributed data')
        return model.get_weights(), len(client_data), {}


    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1).reshape(
            -1, 1
        )  # MobileNetV2 outputs 10 possible classes, argmax returns just the most probable

        acc, rec, prec, f1 = eval_learning(y_test, y_pred)
        output_dict = {
            "accuracy": accuracy,  # accuracy from tensorflow model.evaluate
            "acc": acc,
            "rec": rec,
            "prec": prec,
            "f1": f1,
        }
        return loss, len(x_test), output_dict

# def calculate_eod(self, client_test_data, client_test_labels, y_pred):
#     # Modify the average parameter based on your classification task
#     # For binary classification, use "binary"
#     # For multiclass classification, use "weighted" or "micro" or others
#     rec_group_1 = recall_score(client_test_labels, y_pred, average="binary")
#     rec_group_2 = recall_score(client_test_labels, y_pred, average="binary")

#     eod = rec_group_1 - rec_group_2

#     return eod


# Set up FlowerClient instances with client_id and gender_distribution
client_id_1 = 0
client_id_2 = 1
gender_distribution = {client_id_1: 1.0, client_id_2: 2.0}


import argparse

def main():

    parser = argparse.ArgumentParser(description='A simple Python script with command-line arguments.')

    # Add arguments
    parser.add_argument('--gender', '-g', default = "1",help='Biased Gender filter')
        # Parse the command-line arguments
    args = parser.parse_args()

    client_id = int(args.gender)

    # Start Flower clients
    fl.client.start_client(
        server_address="127.0.0.1:8081",
        client=FlowerClient(client_id, gender_distribution).to_client()

    )


if __name__ == "__main__":

    main()

