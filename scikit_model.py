import time
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

from data import load_dataset

def get_model():
    return DecisionTreeClassifier(random_state=42)

def train_model(model, X_train, y_train):
    return model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Precisión del modelo DecisionTreeClassifier: {accuracy}')

    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusión - Modelo DecisionTreeClassifier:")
    print(cm)

def plot_training(model, training, features):
    tree_rules = export_text(model, feature_names=features)
    print("Árbol de Decisión:")
    print(tree_rules)

    plt.figure(figsize=(10,10))
    plot_tree(training)
    plt.show()

def run_decision_tree(X_train, y_train, X_test, y_test, features):
    print(f'Starting Decision Tree model execution...')
    start_time = time.time()

    model = get_model()

    training = train_model(model, X_train, y_train)
    plot_training(model, training, features)

    evaluate_model(model, X_test, y_test)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Process has finished in {total_time} seconds.")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, features, _ = load_dataset()
    run_decision_tree(X_train, y_train, X_test, y_test, features)