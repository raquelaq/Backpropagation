import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class DatasetLoader:

    # Esta clase permite cargar y preprocesar conjuntos de datos comunes de sklearn, como "iris" y "wine".
    # Facilita el acceso a estos datos en un formato listo para usar en redes neuronales y modelos de aprendizaje automático.
    # La clase realiza las siguientes tareas:
    # 1. Carga del dataset especificado.
    # 2. Normalización de las características (escalando los valores para que estén entre 0 y 1).
    # 3. One-Hot Encoding de las etiquetas, transformando las clases en vectores binarios.
    # 4. División del conjunto de datos en subconjuntos de entrenamiento y prueba.

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_data(self):

        if self.dataset_name not in ["iris", "wine"]:
            raise ValueError(f"Dataset '{self.dataset_name}' no reconocido")

        if self.dataset_name == "iris":
            data = datasets.load_iris()
        elif self.dataset_name == "wine":
            data = datasets.load_wine()

        X, y = data.data, data.target.reshape(-1, 1)

        # Normalización de características y One-Hot Encoding de etiquetas
        X = X / X.max(axis=0)
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(y)

        # Dividir en datos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test
