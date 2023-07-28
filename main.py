from data import load_dataset
from perceptron_model import run_perceptron
from tensorflow_model import run_tensorflow
from scikit_model import run_decision_tree

def main():
    X_train, y_train, X_test, y_test, features, _ = load_dataset()
    run_perceptron(X_train, y_train, X_test, y_test)
    run_tensorflow(X_train, y_train, X_test, y_test, epochs=20)
    run_decision_tree(X_train, y_train, X_test, y_test, features)

if __name__ == '__main__':
    main()