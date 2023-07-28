import time
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix

from data import load_dataset

def get_model(X_train):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def train_model(model, X_train, y_train, epochs=50):
    return model.fit(
        X_train, 
        y_train, 
        epochs=epochs, 
        batch_size=32, 
        validation_split=0.2
    )

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusión - Modelo TensorFlow:")
    print(cm)

def plot_training(history):
    plt.plot(history['accuracy'])
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.title('Mejora de la precisión en el entrenamiento - Modelo TensorFlow')
    plt.show()


def run_tensorflow(X_train, y_train, X_test, y_test, epochs=50):
    print(f'Starting Tensorflow model execution...')
    start_time = time.time()

    model = get_model(X_train)

    training = train_model(model, X_train, y_train, epochs)
    plot_training(training.history)

    evaluate_model(model, X_test, y_test)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Process has finished in {total_time} seconds.")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, _, _ = load_dataset()
    run_tensorflow(X_train, y_train, X_test, y_test)
