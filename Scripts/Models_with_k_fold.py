import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.neural_network import MLPClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer
class ModelEvaluator:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.models = {
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "SVM": SVC(probability=True),  # SVM for classification
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
            "SVM": {
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
    def shap_summary_plot(self, model, feature_names):
        """Generate SHAP summary plot for a fitted model."""
        explainer = shap.Explainer(model, self.X)
        shap_values = explainer(self.X)
        shap.summary_plot(shap_values, self.X, feature_names=feature_names)

    def shap_force_plot(self, model, feature_names, instance_index):
        """Generate SHAP force plot for an individual prediction."""
        explainer = shap.Explainer(model, self.X)
        shap_values = explainer(self.X)
        shap.initjs()
        return shap.force_plot(explainer.expected_value, shap_values[instance_index].values, self.X.iloc[instance_index], feature_names=feature_names)

    def lime_explanation(self, model, feature_names, instance_index):
        """Generate LIME explanation for an individual prediction."""
        explainer = LimeTabularExplainer(
            self.X.values,
            feature_names=feature_names,
            class_names=["Class 0", "Class 1"],
            mode="classification"
        )
        exp = explainer.explain_instance(
            self.X.iloc[instance_index].values,
            model.predict_proba
        )
        exp.show_in_notebook(show_table=True)

    def plot_roc(self, filename='roc_curve.png'):
        plt.figure(figsize=(10, 8))
        for model_name, model in self.models.items():
            best_model = self._tune_hyperparameters(model_name, model)
            self._plot_model_roc(model_name, best_model)
            self._display_metrics(model_name, best_model)

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)
        plt.title(f'Mean ROC Curves for Selected Models', fontsize=16)
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Plot saved as {filename}")
        plt.show()

    def _tune_hyperparameters(self, model_name, model):
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=self.param_grids[model_name],
            cv=self.cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(self.X, self.y)
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def _evaluate_model(self, model_name, model):
        fpr_list, tpr_list = [], []
        auc_list, f1_list = [], []

        for train_idx, test_idx in self.cv.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

            # Train the model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Metrics
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            f1 = f1_score(y_test, y_pred)

            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(roc_auc)
            f1_list.append(f1)

            # Plot each fold
            plt.plot(fpr, tpr, linestyle='-', label=f'{model_name} Fold {len(auc_list)} (AUC={roc_auc:.2f})')

        # Calculate mean and SD of metrics
        mean_auc = np.mean(auc_list)
        sd_auc = np.std(auc_list)
        mean_f1 = np.mean(f1_list)
        sd_f1 = np.std(f1_list)

        # Plot mean ROC curve
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_list, tpr_list)], axis=0)
        plt.plot(mean_fpr, mean_tpr, linestyle='-', lw=3, label=f'{model_name} Mean ROC (AUC={mean_auc:.2f})')

        return mean_auc, sd_auc, mean_f1, sd_f1

    def _plot_model_roc(self, model_name, model):
        fpr_list, tpr_list, auc_list = [], [], []  
        roc_auc_per_fold, f1_per_fold = [], []

        for fold, (train_idx, test_idx) in enumerate(self.cv.split(self.X, self.y)):
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            f1 = f1_score(y_test, y_pred)

            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(roc_auc)
            roc_auc_per_fold.append(roc_auc)
            f1_per_fold.append(f1)

            plt.plot(fpr, tpr, linestyle='-', label=f'{model_name} Fold {fold + 1} (AUC = {roc_auc:.2f})')

        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(mean_fpr)
        for fpr, tpr in zip(fpr_list, tpr_list):
            mean_tpr += np.interp(mean_fpr, fpr, tpr)

        mean_tpr /= len(fpr_list)
        mean_auc = np.mean(roc_auc_per_fold)
        sd_auc = np.std(roc_auc_per_fold)
        mean_f1 = np.mean(f1_per_fold)
        sd_f1 = np.std(f1_per_fold)

        plt.plot(mean_fpr, mean_tpr, linestyle='-', lw=3, label=f'{model_name} Mean ROC (AUC = {mean_auc:.2f})')
        print(f"{model_name}: Mean AUC = {mean_auc:.2f}, SD AUC = {sd_auc:.2f}, Mean F1 = {mean_f1:.2f}, SD F1 = {sd_f1:.2f}")

    def _display_metrics(self, model_name, model):
        y_pred = model.predict(self.X)
        y_pred_proba = model.predict_proba(self.X)[:, 1]

        cm = confusion_matrix(self.y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = accuracy_score(self.y, y_pred)
        precision = precision_score(self.y, y_pred)
        recall = recall_score(self.y, y_pred)
        specificity = tn / (tn + fp)
        sensitivity = recall
        f1 = f1_score(self.y, y_pred)
        roc_auc = roc_auc_score(self.y, y_pred_proba)

        print(f"\nMetrics for {model_name}:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall (Sensitivity): {sensitivity:.2f}")
        print(f"Specificity: {specificity:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"ROC AUC: {roc_auc:.2f}")