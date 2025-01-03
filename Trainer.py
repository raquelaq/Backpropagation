import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


class Trainer:

    # Esta clase gestiona el entrenamiento, evaluación y visualización de la pérdida de la red neuronal.
    # Simplifica el flujo de trabajo, permitiendo entrenar el modelo, evaluar su precisión y monitorear su rendimiento.

    # Principales métodos:
    # 1. train: entrena el modelo por un número específico de épocas, registrando la pérdida en cada una para
    #    observar la convergencia.
    # 2. evaluate: calcula la precisión y la matriz de confusión en el conjunto de prueba, facilitando el análisis
    #    de rendimiento del modelo.
    # 3. plot_losses: grafica la pérdida durante el entrenamiento, permitiendo una evaluación visual de la estabilidad
    #    del modelo.

    def __init__(self, model, X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size

    def train(self, epochs):
        losses = []
        val_losses = []
        n_samples = self.X_train.shape[0]

        for epoch in range(epochs):
            # Mezclar los datos al inicio de cada época
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train_shuffled = self.X_train[indices]
            y_train_shuffled = self.y_train[indices]

            epoch_loss = 0

            # Iterar por batches
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_train_shuffled[i:i + self.batch_size]
                y_batch = y_train_shuffled[i:i + self.batch_size]

                # Forward y backward para el batch
                output = self.model.forward(X_batch)
                batch_loss = np.mean((y_batch - output) ** 2)
                self.model.backward(X_batch, y_batch)

                epoch_loss += batch_loss

            # Guardar la pérdida promedio de la época
            epoch_loss /= (n_samples // self.batch_size)
            losses.append(epoch_loss)

            val_output = self.model.forward(self.X_val)
            val_loss = np.mean((self.y_val - val_output) ** 2)
            val_losses.append(val_loss)

            # Opción de impresión para monitorear
            #if epoch % 100 == 0:
            #    print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

        return losses, val_losses

    def evaluate(self):
        predictions = self.model.forward(self.X_test)
        predictions = np.argmax(predictions, axis=1)
        y_true = np.argmax(self.y_test, axis=1)

        conf_matrix = confusion_matrix(y_true, predictions)
        accuracy = accuracy_score(y_true, predictions)
        return conf_matrix, accuracy

    def plot_losses(self, losses):
        plt.figure(figsize=(5, 5))
        plt.ylim(0, 1)
        plt.plot(losses)
        plt.axhline(y=0, color='r', linewidth=1)
        plt.title("Evolución de la pérdida durante el entrenamiento")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()
