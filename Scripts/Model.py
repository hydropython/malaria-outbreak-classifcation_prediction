from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
class MLAlgorithms:
    def __init__(self, data):
        self.data = data
        self.X = None  # Features
        self.y = None  # Target (Median_classification)
        self.scaler = StandardScaler()

    def preprocess_features(self):
        """
        Preprocesses features by selecting necessary columns and scaling.
        """
        # Select features (you can modify these based on your needs)
        feature_columns = ['T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'WS2M', 'PRECTOTCORR', 'GWETTOP', 'T2MDEW']
        self.X = self.data[feature_columns]
        self.y = self.data['Median_classification']

        # Standardize features (optional, depending on the model)
        self.X = self.scaler.fit_transform(self.X)

    def hyperparameter_tuning(self, model, param_grid):
        """
        Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
        """
        search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        search.fit(self.X, self.y)
        print(f"Best Parameters: {search.best_params_}")
        return search.best_estimator_

    def plot_roc_curve(self, model):
        """
        Plots ROC Curve and computes AUC.
        """
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        # Get probabilities for ROC curve
        y_probs = model.predict_proba(X_test)[:, 1]

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

        print(f"AUC Score: {roc_auc:.4f}")

    def train_random_forest(self):
        """
        Trains a Random Forest Classifier with hyperparameter tuning.
        """
        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Initialize Random Forest model
        rf_model = RandomForestClassifier(random_state=42)

        # Hyperparameter tuning
        best_rf_model = self.hyperparameter_tuning(rf_model, param_grid)

        # Train model and plot ROC curve
        self.plot_roc_curve(best_rf_model)

    def train_gradient_boosting(self):
        """
        Trains a Gradient Boosting Classifier with hyperparameter tuning.
        """
        # Define parameter grid for Gradient Boosting
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5]
        }

        # Initialize Gradient Boosting model
        gb_model = GradientBoostingClassifier(random_state=42)

        # Hyperparameter tuning
        best_gb_model = self.hyperparameter_tuning(gb_model, param_grid)

        # Train model and plot ROC curve
        self.plot_roc_curve(best_gb_model)

    def train_xgboost(self):
        """
        Trains an XGBoost Classifier with hyperparameter tuning.
        """
        # Define parameter grid for XGBoost
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 6],
            'subsample': [0.8, 1.0]
        }

        # Initialize XGBoost model
        xgb_model = XGBClassifier(random_state=42)

        # Hyperparameter tuning
        best_xgb_model = self.hyperparameter_tuning(xgb_model, param_grid)

        # Train model and plot ROC curve
        self.plot_roc_curve(best_xgb_model)

    def train_svr(self):
        """
        Trains a Support Vector Regression model with hyperparameter tuning.
        """
        # Define parameter grid for SVR
        param_grid = {
            'C': [1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }

        # Initialize SVR model
        svr_model = SVR()

        # Hyperparameter tuning
        best_svr_model = self.hyperparameter_tuning(svr_model, param_grid)

        # Train model and plot ROC curve
        self.plot_roc_curve(best_svr_model)

    def train_mlp(self):
        """
        Trains a Multi-Layer Perceptron Classifier with hyperparameter tuning.
        """
        # Define parameter grid for MLP
        param_grid = {
            'hidden_layer_sizes': [(100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'max_iter': [200, 500]
        }

        # Initialize MLP model
        mlp_model = MLPClassifier(random_state=42)

        # Hyperparameter tuning
        best_mlp_model = self.hyperparameter_tuning(mlp_model, param_grid)

        # Train model and plot ROC curve
        self.plot_roc_curve(best_mlp_model)