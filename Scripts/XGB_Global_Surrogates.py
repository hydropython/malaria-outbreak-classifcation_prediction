import pandas as pd
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import spearmanr
import matplotlib.lines as mlines
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVR

class MLAlgorithms:
    def __init__(self, data, target_column='target'):
        # Ensure that `data` is a pandas DataFrame
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            try:
                self.data = pd.DataFrame(data)
            except Exception as e:
                raise ValueError("The provided data could not be converted into a DataFrame.") from e
        
        self.target_column = target_column
        self.X = None
        self.y = None
        self.scaler = StandardScaler()  # StandardScaler for feature scaling
        self.label_encoder = LabelEncoder()  # LabelEncoder for encoding categorical target values
        
        # Initialize models
        self.best_xgb = XGBClassifier()  # XGBoost classifier
        self.best_gb = GradientBoostingClassifier()  # Gradient Boosting classifier
        self.best_rf = RandomForestClassifier()  # Random Forest classifier
        self.best_svr = SVR()  # Support Vector Regressor (for regression tasks, modify if needed)

    def preprocess_features(self):
        """Prepares features and target for modeling."""
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
            print(f"Processed 'date' column: {self.data['date'].head()}")

        columns_to_drop = ['ALLSKY_SFC_LW_DWN', 'TS', 'T2M_RANGE', 'T2MDEW ', 'GWETROOT']
        columns_to_drop = [col for col in columns_to_drop if col in self.data.columns]
        self.data = self.data.drop(columns=columns_to_drop, errors='ignore')
        print(f"Dropped columns: {columns_to_drop}")

        if 'date' in self.data.columns:
            self.data['year'] = self.data['date'].dt.year
            self.data['month'] = self.data['date'].dt.month
            self.data['day'] = self.data['date'].dt.day
            self.data['weekday'] = self.data['date'].dt.weekday
            self.data['day_of_year'] = self.data['date'].dt.dayofyear
            print(f"Extracted date features: {self.data[['year', 'month', 'day', 'weekday', 'day_of_year']].head()}")

        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' is missing from the dataset.")
        
        self.data = self.data.dropna(subset=[self.target_column])
        print(f"Data after dropping rows with missing target values: {self.data.head()}")

        feature_columns = [col for col in self.data.columns if col not in [self.target_column, 'date']]
        self.X = self.data[feature_columns].fillna(self.data[feature_columns].mean())
        print(f"Selected feature columns: {feature_columns}")

        self.y = self.data[self.target_column].astype(str)
        self.label_encoder.fit(self.y)
        self.y = self.label_encoder.transform(self.y)

        unique_labels = np.unique(self.y)
        new_labels = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.y = np.array([new_labels[label] for label in self.y])

        unique_classes = np.unique(self.y)
        print(f"Unique classes found in target: {unique_classes}")

        self.X = self.scaler.fit_transform(self.X)
        print(f"Feature data after standardization: {self.X[:5]}")

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
    
    def train_random_forest(self):
        """Trains Random Forest classifier and returns accuracy."""
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.best_rf = RandomForestClassifier()
        self.best_rf.fit(X_train, y_train)
        y_pred = self.best_rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Random Forest Accuracy: {accuracy:.4f}")
        return accuracy

    def train_svr(self):
        """Trains Support Vector Regression (SVR) and returns R-squared score."""
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.best_svr = SVR()
        self.best_svr.fit(X_train, y_train)
        y_pred = self.best_svr.predict(X_test)
        r2_score = self.best_svr.score(X_test, y_test)
        print(f"SVR R-squared: {r2_score:.4f}")
        return r2_score

    def train_decision_tree_surrogate(self, model_type):
        # Select the appropriate model based on model_type
        if model_type == "xgb":
            model = self.best_xgb
        elif model_type == "gb":
            model = self.best_gb
        elif model_type == "rf":
            model = self.best_rf
        elif model_type == "svr":
            model = self.best_svr
        else:
            raise ValueError(f"Model type {model_type} not recognized.")
        
        # Train the model first (if not already trained)
        model.fit(self.X, self.y)

        if isinstance(model, SVR):  # Check if the model is a regression model (SVR)
            # For SVR, skip Decision Tree surrogate as it's for classification
            print(f"Skipping surrogate for {model_type} as it is a regression model.")
            return None  # Return None for SVR, or handle as needed
        
        # Create a Decision Tree surrogate model (for classification)
        dt = DecisionTreeClassifier(max_depth=3)  # Decision Tree for classification
        dt.fit(self.X, model.predict(self.X))  # Train Decision Tree on model predictions (classification)

        # Compute surrogate accuracy (how well the Decision Tree mimics the original model)
        surrogate_predictions = dt.predict(self.X)
        
        # Compute classification accuracy (percentage of correct predictions)
        accuracy = accuracy_score(model.predict(self.X), surrogate_predictions)

        return accuracy
    def plot_feature_importance(self, model_type, feature_names):
        """Plots feature importance for surrogate model."""
        if model_type == 'rf':
            feature_importance = self.best_rf.feature_importances_
        elif model_type == 'svr':
            feature_importance = self.best_svr.coef_  # Assuming you're using SVR with linear kernel
        elif model_type == 'xgb':
            feature_importance = self.best_xgb.feature_importances_
        elif model_type == 'gb':
            feature_importance = self.best_gb.feature_importances_

        # Convert feature importance to a DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })

        # Sort the features by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot the feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f"Feature Importance for {model_type} Model")
        plt.show()

    def calculate_spearman_fidelity(self, accuracies, surrogate_accuracies):
        # Calculate Spearman correlation between accuracies and surrogate_accuracies
        spearman_corr, _ = spearmanr(accuracies, surrogate_accuracies)
        return spearman_corr

    def save_results_to_csv(self, accuracies, surrogate_accuracies, files, filename='model_accuracy_results.csv'):
        """Save model accuracy, surrogate accuracy, and Spearman correlation per dataset to CSV."""
        
        # Collect results into a list of dictionaries for each dataset
        results = []
        for i, file in enumerate(files):
            dataset_results = {
                'Dataset': file,
                'XGBoost Accuracy': accuracies['xgb'][i],
                'Gradient Boosting Accuracy': accuracies['gb'][i],
                'SVR Accuracy': accuracies['svr'][i],
                'Random Forest Accuracy': accuracies['rf'][i],
                'XGBoost Surrogate Accuracy': surrogate_accuracies['xgb'][i],
                'Gradient Boosting Surrogate Accuracy': surrogate_accuracies['gb'][i],
                'SVR Surrogate Accuracy': surrogate_accuracies['svr'][i],
                'Random Forest Surrogate Accuracy': surrogate_accuracies['rf'][i],
            }

            # Compute Spearman correlation for this dataset
            for model in ['xgb', 'gb', 'svr', 'rf']:
                try:
                    spearman_corr, _ = spearmanr([accuracies[model][i]], [surrogate_accuracies[model][i]])
                    dataset_results[f"{model.upper()} Spearman Correlation"] = round(spearman_corr, 4)
                except ValueError:
                    dataset_results[f"{model.upper()} Spearman Correlation"] = "N/A"  # Handle missing/insufficient data

            results.append(dataset_results)

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)

        # Save as CSV
        results_df.to_csv(filename, index=False)
        print(f"✅ Results saved to {filename}")

    def plot_accuracy_vs_fidelity(self, accuracies, surrogate_accuracies, files, save_path="publication_plot"):
        """Plot Accuracy vs Fidelity with publication-quality formatting and high-resolution saving."""
        
        try:
            # Extract dataset names from filenames
            dataset_names = []
            for file in files:
                try:
                    base_name = os.path.basename(file).split('_')
                    dataset_name = base_name[2] if len(base_name) > 2 else base_name[-1]
                except Exception as e:
                    print(f"Warning: Unable to extract dataset name from file '{file}': {e}")
                    dataset_name = "Unknown"
                dataset_names.append(dataset_name)

            # Generate unique abbreviations for datasets
            dataset_abbr = []
            abbr_counts = Counter()
            for name in dataset_names:
                abbr = name[:3].upper()  # Take first 3 letters
                abbr_counts[abbr] += 1
                abbr = f"{abbr}{abbr_counts[abbr]}" if abbr_counts[abbr] > 1 else abbr
                dataset_abbr.append(abbr)

            # Assign unique colors to datasets
            distinct_colors = sns.color_palette("Set2", len(dataset_abbr))

            # Define markers for models
            markers = {'xgb': 'o', 'gb': 'x', 'rf': '*'}

            # Set up the figure
            sns.set(style="whitegrid")
            fig, ax = plt.subplots(figsize=(12, 7), dpi=300)  # High DPI for quality

            dataset_handles = []  # Store dataset legend handles
            model_handles = []    # Store model legend handles

            # Shade the area where good results are happening (e.g., accuracy > 0.75 and fidelity > 0.85)
            ax.axhspan(0.95, 1, color='green', alpha=0.1, label='Good Results Area')
            #ax.axvspan(0.70, 1, color='green', alpha=0.1)

            best_model = None
            best_dataset = None
            best_acc = -1
            best_fidelity = -1

            # Plot each dataset
            for i, file in enumerate(files):
                color = distinct_colors[i]  # Unique color for each dataset
                dataset_label = dataset_abbr[i] if dataset_abbr[i] not in [h.get_label() for h in dataset_handles] else ""

                # Plot each model
                for model, marker in markers.items():
                    try:
                        acc = accuracies[model][i]
                        fidelity = surrogate_accuracies[model][i]

                        if acc is None or fidelity is None:
                            continue  # Skip plotting if any value is None

                        # Check for the best model-dataset pair
                        if acc > best_acc and fidelity > best_fidelity:
                            best_acc = acc
                            best_fidelity = fidelity
                            best_model = model
                            best_dataset = dataset_abbr[i]

                        # Plot each point
                        ax.scatter(float(acc), float(fidelity), marker=marker, color=color, s=90, label=f"{model.upper()}_{dataset_abbr[i]}")

                        # Label all points
                        ax.text(float(acc), float(fidelity), f"{model.upper()}_{dataset_abbr[i]}", fontsize=9, ha='right', va='bottom')

                    except (KeyError, IndexError, ValueError, TypeError) as e:
                        print(f"Warning: Issue with model '{model}' at index {i}: {e}")

                # Add dataset legend handle only once per dataset
                if dataset_label:
                    dataset_handles.append(ax.scatter([], [], marker='o', color=color, label=dataset_abbr[i]))

            # Highlight the best model-dataset pair in red with a bounding box
            if best_model and best_dataset:
                ax.scatter(float(best_acc), float(best_fidelity), marker=markers[best_model], color='red', s=150, edgecolor='black', linewidth=2)
                ax.text(float(best_acc), float(best_fidelity), f"**BEST**: {best_model.upper()}_{best_dataset}", 
                        fontsize=11, color='red', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))

            # Model legend (below the graph)
            model_handles.append(mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=10, label="XGB"))
            model_handles.append(mlines.Line2D([], [], color='red', marker='x', linestyle='None', markersize=10, label="GB"))
            model_handles.append(mlines.Line2D([], [], color='purple', marker='*', linestyle='None', markersize=10, label="RF"))

            # Labels and title
            ax.set_xlabel("Accuracy", fontsize=12, fontweight='bold')
            ax.set_ylabel("Fidelity", fontsize=12, fontweight='bold')
            ax.set_title("Model Accuracy vs. Fidelity", fontsize=12, fontweight='bold')

            # Setting axis limits
            ax.set_xlim(0.70, 1)
            ax.set_ylim(0.80, 1)

            # Grid and background styling
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_facecolor('white')

            # Combine dataset and model legends below the graph
            legend1 = ax.legend(handles=model_handles, title="Models", loc='upper center', frameon=True, fontsize=12, bbox_to_anchor=(0.5, -0.15), ncol=3)
            legend2 = ax.legend(handles=dataset_handles, title="Datasets", loc='right center', frameon=True, fontsize=12, bbox_to_anchor=(0.5, -0.25), ncol=len(dataset_handles))

            # Add both legends to the plot
            ax.add_artist(legend1)
            ax.add_artist(legend2)

            # Adjust layout for publication quality
            plt.tight_layout(rect=[0, 0.15, 1, 1])  # Extra bottom space for legends

            # 📌 SAVE FIGURE IN MULTIPLE FORMATS FOR PUBLICATION
            plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")   # High-quality PNG
            plt.savefig(f"{save_path}.pdf", format="pdf", bbox_inches="tight")  # Vector PDF
            plt.savefig(f"{save_path}.svg", format="svg", bbox_inches="tight")  # Scalable SVG
            
            # Show the plot
            plt.show()

            print(f"✅ Publication-quality figure saved as '{save_path}.png', '{save_path}.pdf', and '{save_path}.svg'")

        except Exception as e:
            print(f"Error: An unexpected issue occurred: {e}")


