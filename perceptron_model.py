import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from data import load_dataset

def initialize_parameters(input_size):
    np.random.seed(42)
    W = np.random.randn(input_size, 1)
    b = np.random.randn()
    return W, b

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagation(X, W, b):
    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    return A

def compute_cost(A, y):
    m = y.shape[0]
    cost = -1/m * np.sum(y * np.log(A) + (1-y) * np.log(1-A))
    return cost

def compute_accuracy(A, y):
    y_pred = (A > 0.5).astype(int)
    accuracy = np.mean(y_pred == y)
    return accuracy

def backward_propagation(X, y, A):
    m = y.shape[0]
    dZ = A - y
    dW = 1/m * np.dot(X.T, dZ)
    db = 1/m * np.sum(dZ)
    return dW, db

def update_parameters(W, b, dW, db, learning_rate):
    W -= learning_rate * dW
    b -= learning_rate * db
    return W, b

def train_model(X_train, y_train, epochs=50, learning_rate=0.01):
    input_size = X_train.shape[1]
    W, b = initialize_parameters(input_size)

    history = {'cost': [], 'accuracy': []}

    for epoch in range(epochs):
        A = forward_propagation(X_train, W, b)
        cost = compute_cost(A, y_train)
        accuracy = compute_accuracy(A, y_train)
        dW, db = backward_propagation(X_train, y_train, A)
        W, b = update_parameters(W, b, dW, db, learning_rate)

        history['cost'].append(cost)
        history['accuracy'].append(accuracy)

    return W, b, history

def predict(X, W, b):
    A = forward_propagation(X, W, b)
    return (A > 0.5).astype(int)

def evaluate_model(X_test, y_test, W, b):
    y_pred = predict(X_test, W, b)
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusión - Modelo Perceptrón:")
    print(cm)

def plot_training(history):
    epochs = range(1, len(history['cost']) + 1)

    plt.plot(epochs, history['cost'], label='Costo')
    plt.xlabel('Épocas')
    plt.ylabel('Métricas')
    plt.title('Reducción del costo en el entrenamiento - Modelo Perceptrón')
    plt.legend()
    plt.show()

def run_perceptron(X_train, y_train, X_test, y_test, epochs=50):
    print(f'Starting Perceptrón model execution...')
    start_time = time.time()

    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    W, b, history = train_model(X_train, y_train, epochs)
    plot_training(history)

    evaluate_model(X_test, y_test, W, b)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Process has finished in {total_time} seconds.")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, _, _ = load_dataset()
    run_perceptron(X_train, y_train, X_test, y_test)
