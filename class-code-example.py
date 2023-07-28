from data import load_dataset
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_text
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report

def main():
    train_dataset, test_dataset = load_dataset()
    print(f"Train dataset:\n{train_dataset}")
    print(f"Test dataset:\n{test_dataset}")

    Clases=train_dataset[train_dataset.columns.values[-1]]
    Caracteristicas=train_dataset.columns.values[:-1]
    Valores=train_dataset[Caracteristicas]

    le = preprocessing.LabelEncoder()
    for column_name in Valores.columns:
        if Valores[column_name].dtype == object:
            Valores[column_name] = le.fit_transform(Valores[column_name])
        else:
            pass

    #definir modelo (elegir clasificador)
    Modelo=DecisionTreeClassifier(max_depth=2)

    #Entrenamiento
    grafica=Modelo.fit(Valores,Clases)

    arbol=export_text(Modelo,feature_names=Caracteristicas)
    print(arbol)


    plt.figure(figsize=(10,10))
    tree.plot_tree(grafica)
    plt.show()

    #Clasificar el test
    ClasesRecuperadas=Modelo.predict(Valores)

    print(ClasesRecuperadas)

    #Evaluar
    NombreClases = ['False', 'True'] #Cuidar el orden, al final la clase que nos interesa

    Clase_Test = test_dataset[test_dataset.columns.values[-1]]
    

    Matriz=confusion_matrix(Clase_Test, ClasesRecuperadas,labels=NombreClases)
    tn,fp,fn,tp=confusion_matrix(Clase_Test, ClasesRecuperadas,labels=NombreClases).ravel()
    print(Matriz)
    """ Se imprime en formato:
    [ TN  FP
    FN  TP ]
    """
    print(tp)

    ReporteFormato=classification_report(Clase_Test, ClasesRecuperadas, target_names=NombreClases)
    print(ReporteFormato)



    Reporte=classification_report(Clase_Test, ClasesRecuperadas, target_names=NombreClases, output_dict=True)
    print(Reporte)
    print(Reporte['tested_positive']['precision'])

if __name__ == "__main__":
    main()