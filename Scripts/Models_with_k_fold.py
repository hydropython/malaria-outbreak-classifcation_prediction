import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
class ModelEvaluator:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.models = {
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "SVR": SVC(probability=True),  # SVR for classification (using probability=True for ROC)
            "MLP": MLPClassifier(max_iter=1000)  # MLP Classifier
        }
        self.cv = StratifiedKFold(n_splits=5)  # 5-fold cross-validation
        self.param_grids = {
            "Random Forest": {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            },
            "Gradient Boosting": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'subsample': [0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "XGBoost": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            "SVR": {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            "MLP": {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'sgd'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }

    def plot_roc(self, filename='roc_curve.png'):
        plt.figure(figsize=(10, 8))  # Set the figure size
        for model_name, model in self.models.items():
            best_model = self._tune_hyperparameters(model_name, model)
            self._plot_model_roc(model_name, best_model)
            self._display_metrics(model_name, best_model)
        
        # Plot the diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)

        # Customizing plot for publication quality
        plt.title(f'Mean ROC Curves for Selected Models', fontsize=16)
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # Save the plot as a file
        plt.savefig(filename)
        print(f"Plot saved as {filename}")

        # Display the plot
        plt.show()

    def _tune_hyperparameters(self, model_name, model):
        """Tune hyperparameters using GridSearchCV"""
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=self.param_grids[model_name],
                                   cv=self.cv,
                                   scoring='roc_auc',
                                   n_jobs=-1,
                                   verbose=1)
        grid_search.fit(self.X, self.y)
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def _plot_model_roc(self, model_name, model):
        fpr_list, tpr_list, auc_list = [], [], []  # Reset lists for each model
        roc_auc_per_fold = []

        # Perform the k-fold cross-validation
        for fold, (train_idx, test_idx) in enumerate(self.cv.split(self.X, self.y)):
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]  # For pandas DataFrame
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]  # For pandas Series
            
            # Fit the model on the training data
            model.fit(X_train, y_train)
            
            # Get prediction probabilities for ROC curve (positive class probability)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Append the ROC curve data and AUC
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(roc_auc)
            roc_auc_per_fold.append(roc_auc)  # Store AUC for this fold
            
            # Plot ROC curve for the current fold with specific styling
            plt.plot(fpr, tpr, linestyle='-', color='C{}'.format(fold), label=f'{model_name} Fold {fold + 1} (AUC = {roc_auc:.2f})')

        # Calculate the mean ROC curve (average FPR and TPR across all folds)
        mean_fpr = np.linspace(0, 1, 100)  # 100 points for interpolation
        mean_tpr = np.zeros_like(mean_fpr)

        # Interpolate TPR values for each fold at the common FPR values
        for fpr, tpr in zip(fpr_list, tpr_list):
            mean_tpr += np.interp(mean_fpr, fpr, tpr)

        # Average the TPR values
        mean_tpr /= len(fpr_list)

        # Plot the mean ROC curve
        mean_auc = np.mean(roc_auc_per_fold)
        plt.plot(mean_fpr, mean_tpr, linestyle='-', lw=3, label=f'{model_name} Mean ROC (AUC = {mean_auc:.2f})')

        # Print the mean AUC for this model
        print(f'{model_name} Mean AUC across all folds: {mean_auc:.2f}')

    def _display_metrics(self, model_name, model):
        """Display standard classification metrics"""
        y_pred = model.predict(self.X)
        
        accuracy = accuracy_score(self.y, y_pred)
        precision = precision_score(self.y, y_pred)
        recall = recall_score(self.y, y_pred)
        f1 = f1_score(self.y, y_pred)
        roc_auc = roc_auc_score(self.y, model.predict_proba(self.X)[:, 1])
        
        print(f"\nMetrics for {model_name}:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"ROC AUC: {roc_auc:.2f}")