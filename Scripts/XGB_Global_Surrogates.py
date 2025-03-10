import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import shap
import lime
import lime.lime_tabular
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.lines as mlines
from scipy.stats import spearmanr

class MLAlgorithms:
    def __init__(self, data, target_column='target'):
        self.data = data
        self.target_column = target_column
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.best_xgb = None
        self.best_gb = None
        self.label_encoder = LabelEncoder()

    def preprocess_features(self):
        """Prepares features and target for modeling."""
        # Ensure the date column is in datetime format
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')

        # Drop unnecessary columns (check if they exist first)
        columns_to_drop = ['ALLSKY_SFC_LW_DWN', 'TS', 'T2M_RANGE', 'T2MDEW ', 'GWETROOT']
        columns_to_drop = [col for col in columns_to_drop if col in self.data.columns]
        self.data = self.data.drop(columns=columns_to_drop, errors='ignore')

        # Feature engineering: create time-based features
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        self.data['day'] = self.data['date'].dt.day
        self.data['weekday'] = self.data['date'].dt.weekday
        self.data['day_of_year'] = self.data['date'].dt.dayofyear

        # Check for missing target values
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' is missing from the dataset.")
        
        # Drop rows with missing target values
        self.data = self.data.dropna(subset=[self.target_column])

        # Prepare features and target
        feature_columns = [col for col in self.data.columns if col not in [self.target_column, 'date']]
        self.X = self.data[feature_columns].fillna(self.data[feature_columns].mean())  # Fill missing features with mean

        # Ensure categorical target and encode labels
        self.y = self.data[self.target_column].astype(str)
        self.label_encoder.fit(self.y)
        self.y = self.label_encoder.transform(self.y)

        # Ensure continuous class labels starting from 0
        unique_labels = np.unique(self.y)
        new_labels = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.y = np.array([new_labels[label] for label in self.y])

        # Check the range of the class labels
        unique_classes = np.unique(self.y)
        print(f"Unique classes found in target: {unique_classes}")

        # Scale the features
        self.X = self.scaler.fit_transform(self.X)

    def train_xgboost(self):
        """Trains XGBoost classifier and returns accuracy."""
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.best_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        self.best_xgb.fit(X_train, y_train)
        y_pred = self.best_xgb.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"XGBoost Accuracy: {accuracy:.4f}")
        return accuracy

    def train_gradient_boosting(self):
        """Trains Gradient Boosting classifier and returns accuracy."""
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.best_gb = GradientBoostingClassifier()
        self.best_gb.fit(X_train, y_train)
        y_pred = self.best_gb.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Gradient Boosting Accuracy: {accuracy:.4f}")
        return accuracy

    def train_decision_tree_surrogate(self, model_type="xgb"):
        """Trains a Decision Tree as a global surrogate model."""
        if model_type == "xgb" and self.best_xgb is not None:
            model = self.best_xgb
        elif model_type == "gb" and self.best_gb is not None:
            model = self.best_gb
        else:
            raise ValueError(f"Model '{model_type}' not trained yet!")

        X_train, _, y_train, _ = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        dt = DecisionTreeClassifier(max_depth=4)

        # Train decision tree on XGBoost/GB predictions
        pseudo_labels = model.predict(X_train)
        dt.fit(X_train, pseudo_labels)
        print(f"Decision Tree trained as surrogate for {model_type.upper()}")
        return dt

    def interpret_model(self, model_type="xgb"):
        """Generate interpretations using SHAP and LIME."""
        if model_type == "xgb" and self.best_xgb is not None:
            model = self.best_xgb
        elif model_type == "gb" and self.best_gb is not None:
            model = self.best_gb
        else:
            raise ValueError(f"Model '{model_type}' not trained yet!")

        # SHAP interpretation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X)
        shap.summary_plot(shap_values, self.X, plot_type="bar")

        # LIME interpretation
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(self.X, training_labels=self.y, mode='classification')
        lime_explanation = explainer_lime.explain_instance(self.X[0], model.predict_proba)
        lime_explanation.show_in_notebook()

    def calculate_spearman_fidelity(self, model_accuracy, surrogate_fidelity):
        """Calculate Spearman Fidelity for model vs surrogate."""
        spearman_fidelity = spearmanr(model_accuracy, surrogate_fidelity)
        return spearman_fidelity.correlation

    def plot_accuracy_vs_fidelity(self, accuracies, surrogate_accuracies, files):
        """Plot Accuracy vs Fidelity for models and surrogate models."""
        # Generate abbreviated names for datasets
        dataset_names = [os.path.basename(file).split('_')[2] for file in files]  # Extracts dataset name part
        dataset_abbr = [name[:3].upper() for name in dataset_names]  # Short abbreviation of first 3 letters
        
        # Define a list of distinct colors for each dataset
        colors = ['darkblue', 'darkred', 'green', 'purple', 'orange', 'brown', 'pink']
        markers = ['o', 'X', 's', 'D', '^', 'v', '<']  # Different marker styles
        
        # Set a refined style using Seaborn for a more professional look
        sns.set(style="whitegrid", palette="muted")

        # Prepare the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create lists for legend handles
        legend_handles = []

        # Create a list to store the results for the table
        table_data = []

        # Scatter plot for accuracies vs surrogate accuracies with advanced markers and colors
        for i, (acc, surrogate_acc) in enumerate(zip(accuracies, surrogate_accuracies)):
            color = colors[i % len(colors)]  # Cycle through color list
            marker_xgb = markers[0]  # XGBoost always uses 'o' marker
            marker_gb = markers[1]  # Gradient Boosting always uses 'X' marker

            # Plot XGBoost and Gradient Boosting with different colors and markers
            scatter_xgb = ax.scatter(acc[0], surrogate_acc[0], color=color, marker=marker_xgb, s=100, edgecolors='black', alpha=0.8)
            scatter_gb = ax.scatter(acc[1], surrogate_acc[1], color=color, marker=marker_gb, s=100, edgecolors='black', alpha=0.8)

            # Add handles to the legend
            legend_handles.append(mlines.Line2D([0], [0], marker=marker_xgb, color='w', markerfacecolor=color, markersize=10, label=f'{dataset_abbr[i]} XGB'))
            legend_handles.append(mlines.Line2D([0], [0], marker=marker_gb, color='w', markerfacecolor=color, markersize=10, label=f'{dataset_abbr[i]} GB'))

            # Calculate Spearman Fidelity
            spearman_fidelity_xgb = self.calculate_spearman_fidelity(acc[0], surrogate_acc[0])
            spearman_fidelity_gb = self.calculate_spearman_fidelity(acc[1], surrogate_acc[1])

            # Append the dataset and surrogate results to the table_data list
            table_data.append([
                dataset_abbr[i],  # Dataset
                "XGBoost",        # Model (XGBoost)
                acc[0],           # XGB Accuracy
                surrogate_acc[0], # XGB Fidelity
                spearman_fidelity_xgb,  # XGB Spearman Fidelity
                "Gradient Boosting",  # Model (GB)
                acc[1],           # GB Accuracy
                surrogate_acc[1], # GB Fidelity
                spearman_fidelity_gb   # GB Spearman Fidelity
            ])

        # Create a DataFrame for the table
        table_df = pd.DataFrame(table_data, columns=[
                "Dataset", "Model 1", "Accuracy 1", "Fidelity 1", "Spearman Fidelity 1",
               "Model 2", "Accuracy 2", "Fidelity 2", "Spearman Fidelity 2"
                   ])


        # Save table to CSV
        table_df.to_csv('../Data/surrogate/model_accuracies_fidelity.csv', index=False)

        # Plot customizations
        ax.set_xlabel('Accuracy', fontsize=14)
        ax.set_ylabel('Fidelity', fontsize=14)
        ax.set_title('Accuracy vs Fidelity Comparison', fontsize=16)
        ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), title='Dataset & Model', fontsize=12)

        # Save the plot to a file
        plt.tight_layout()
        plt.savefig('../Data/surrogate/accuracy_vs_fidelity_plot.png', format='png')
        plt.show()



