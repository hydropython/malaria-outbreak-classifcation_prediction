import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import spearmanr

class ModelEvaluation:
    def __init__(self, model_1, model_2, surrogate_model, X_train, X_test, y_train, y_test):
        """
        Initialize the ModelEvaluation object with the models and data.
        """
        self.model_1 = model_1
        self.model_2 = model_2
        self.surrogate_model = surrogate_model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_models(self):
        """
        Train both models (model_1 and model_2) and the surrogate model.
        """
        # Train the complex models
        self.model_1.fit(self.X_train, self.y_train)
        self.model_2.fit(self.X_train, self.y_train)

        # Train the surrogate model (Decision Tree) on predictions from model_1
        surrogate_model_1 = self.surrogate_model
        surrogate_model_1.fit(self.X_train, self.model_1.predict(self.X_train))

        # Train the surrogate model (Decision Tree) on predictions from model_2
        surrogate_model_2 = self.surrogate_model
        surrogate_model_2.fit(self.X_train, self.model_2.predict(self.X_train))

        self.surrogate_model_1 = surrogate_model_1
        self.surrogate_model_2 = surrogate_model_2

    def calculate_accuracy(self, model):
        """
        Calculate the accuracy of a model on the test set.
        """
        y_pred = model.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)

    def calculate_fidelity(self, surrogate_model, model):
        """
        Calculate the fidelity of a surrogate model by comparing its predictions
        with the original model's predictions.
        """
        y_pred_surrogate = surrogate_model.predict(self.X_test)
        y_pred_model = model.predict(self.X_test)
        return accuracy_score(y_pred_model, y_pred_surrogate)

    def calculate_spearman_fidelity(self, accuracy_1, fidelity_1, accuracy_2, fidelity_2):
        """
        Calculate the Spearman's rank correlation coefficient between the
        accuracy of the models and the fidelity of their surrogate models.
        """
        return spearmanr([accuracy_1, accuracy_2], [fidelity_1, fidelity_2])[0]

    def plot_accuracy_vs_fidelity(self, accuracy_1, accuracy_2, fidelity_1, fidelity_2):
        """
        Plot accuracy vs. fidelity for both models.
        """
        fig, ax = plt.subplots()
        ax.scatter(accuracy_1, fidelity_1, color='blue', label='Model 1')
        ax.scatter(accuracy_2, fidelity_2, color='green', label='Model 2')

        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Fidelity')
        ax.set_title('Accuracy vs Fidelity')
        ax.legend()
        plt.show()

    def evaluate_models(self):
        """
        Perform model evaluation and plotting.
        """
        # Calculate accuracy for both models
        accuracy_1 = self.calculate_accuracy(self.model_1)
        accuracy_2 = self.calculate_accuracy(self.model_2)

        # Calculate fidelity for both surrogate models
        fidelity_1 = self.calculate_fidelity(self.surrogate_model_1, self.model_1)
        fidelity_2 = self.calculate_fidelity(self.surrogate_model_2, self.model_2)

        # Calculate Spearman Fidelity
        spearman_corr = self.calculate_spearman_fidelity(accuracy_1, fidelity_1, accuracy_2, fidelity_2)
        print(f"Spearman's Rank Correlation between Accuracy and Fidelity: {spearman_corr}")

        # Plot accuracy vs. fidelity
        self.plot_accuracy_vs_fidelity(accuracy_1, accuracy_2, fidelity_1, fidelity_2)

