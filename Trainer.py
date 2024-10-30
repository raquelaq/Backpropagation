import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


class Trainer:

    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train(self, epochs):
        losses = []
        for epoch in range(epochs):
            output = self.model.forward(self.X_train)
            loss = np.mean((self.y_train - output) ** 2)
            losses.append(loss)
            self.model.backward(self.X_train, self.y_train, output)

            # if epoch % 100 == 0:
            #     print(f"Epoch {epoch}, Loss: {loss}")

        return losses

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
