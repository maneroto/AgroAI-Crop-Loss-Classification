# Crop Loss Classification Project

This is a classification project aimed at predicting whether a crop has experienced a significant loss or not, using different machine learning models. It uses a dataset containing information about various crops and the amount of loss they experienced in a specific year.

## Implemented Models

In this project, three different models have been implemented to tackle the crop loss classification problem:

1. **Perceptron**: A perceptron has been implemented using only numpy and pandas libraries to carry out the classification.

2. **TensorFlow Model**: A neural network with TensorFlow has been created using the Dense layers architecture and Sequential sequence to solve the problem.

3. **Scikit-learn's DecisionTreeClassifier Model**: The DecisionTreeClassifier algorithm from Scikit-learn has been utilized to build a decision tree-based model for classification.

## How to Use the Project

To use this project, follow these steps:

1. Clone this repository on your local machine.

2. Create a virtual environment using the following command:

```bash
pip venv env
```

3. Activate the virtual environment with the following command:

```bash
source env/Scripts/activate
```

4. Install the necessary dependencies using the "requirements.txt" file with the following command:

```bash
pip install -r requirements.txt
```

5. Run the project using the command:

```bash
python main.py
```

This will execute the models and display the results, including confusion matrices and metrics such as accuracy, recall, and F1-score for each model.

## Conclusions
After analyzing and comparing the results obtained by the three models, we can conclude that the DecisionTreeClassifier model has shown the best performance for this particular problem. While the Perceptron is a simple and efficient model for linearly separable problems, it was not able to handle the non-linear complexity of this problem. On the other hand, the TensorFlow model, although capable of learning non-linear functions, requires more training time and hyperparameter tuning to achieve optimal results.

The DecisionTreeClassifier model showed high accuracy and a good balance between precision and recall, making it more suitable for this crop loss classification task.

In summary, the choice of the model will depend on the problem's complexity and the amount of available data. For more complex and non-linear problems, the use of more advanced machine learning algorithms, such as neural networks, may be more appropriate, while for simpler and interpretable features, decision tree-based models can be a good choice.