import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MalariaPreprocessor:
    def __init__(self, data_path, target_column, date_column='date'):
        """
        Initializes the preprocessor with the data path and target column.
        """
        self.data_path = data_path
        self.target_column = target_column
        self.date_column = date_column
        self.data = None
        self.q3_values = None

    def load_data(self):
        """
        Loads the dataset from the provided path.
        """
        self.data = pd.read_csv(self.data_path)
        self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
        print(f"Data loaded successfully with shape: {self.data.shape}")

    def preprocess(self):
        """
        Preprocesses the data by adding 'month', 'week', and 'target' columns based on Q3 (third quartile).
        """
        # Extract year, month, and week
        self.data['year'] = self.data[self.date_column].dt.year
        self.data['month'] = self.data[self.date_column].dt.month
        self.data['week'] = self.data[self.date_column].dt.isocalendar().week

        # Calculate the third quartile (Q3) for each (month, week)
        self.q3_values = self.data.groupby(['month', 'week'])[self.target_column].quantile(0.75)

        # Classify as higher or lower than Q3
        def classify_cases(row):
            q3_cases = self.q3_values.loc[row['month'], row['week']]
            return 1 if row[self.target_column] > q3_cases else 0

        self.data['target'] = self.data.apply(classify_cases, axis=1)
        print("Data preprocessing completed. Added 'target' column based on Q3 classification.")

    def visualize_target_distribution(self):
        """
        Visualizes the distribution of the target variable (higher or lower cases).
        """
        sns.countplot(x='target', data=self.data, palette='viridis')
        plt.title('Distribution of Malaria Cases (Higher=1, Lower=0)')
        plt.xlabel('Case Classification')
        plt.ylabel('Count')
        plt.show()

    def visualize_monthly_trends(self):
        """
        Visualizes monthly trends for malaria cases.
        """
        monthly_data = self.data.groupby('month')[self.target_column].mean().reset_index()

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=monthly_data, x='month', y=self.target_column, marker='o', color='green')
        plt.title('Monthly Trends in Malaria Cases')
        plt.xlabel('Month')
        plt.ylabel('Average Cases')
        plt.xticks(range(1, 13))
        plt.grid(alpha=0.5)
        plt.show()

    def visualize_weekly_trends(self):
        """
        Visualizes weekly trends in malaria cases across all months using a heatmap.
        """
        # Group data by month and week, calculating the mean cases
        weekly_data = self.data.groupby(['month', 'week'])[self.target_column].mean().reset_index()

        # Use pivot to reshape the DataFrame for a heatmap
        pivot_table = weekly_data.pivot(index='month', columns='week', values=self.target_column)

        # Create a heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_table, cmap='YlGnBu', annot=False, cbar_kws={'label': 'Average Cases'})
        plt.title('Weekly Trends in Malaria Cases')
        plt.xlabel('Week')
        plt.ylabel('Month')
        plt.show()

    def get_classification_summary(self):
        """
        Returns a summary of classifications and Q3 thresholds.
        """
        # Count the number of Class 1 (Higher) and Class 0 (Lower) cases
        class_counts = self.data['target'].value_counts()
        class_1_count = class_counts.get(1, 0)  # Default to 0 if not present
        class_0_count = class_counts.get(0, 0)  # Default to 0 if not present

        # Prepare the Q3 thresholds for each (month, week)
        q3_thresholds = self.q3_values.reset_index().rename(columns={self.target_column: 'Q3'})

        # Create the summary
        summary = {
            "Class 1 Count (Higher)": class_1_count,
            "Class 0 Count (Lower)": class_0_count,
            "Q3 Thresholds": q3_thresholds  # Include the DataFrame here
        }
        return summary

    def save_with_classification(self):
        """
        Saves the updated dataset with the 'Q3_classification' column to the original file.
        """
        self.data.to_csv(self.data_path.replace(".csv", "_Q3.csv"), index=False)
        print(f"Updated data with 'Q3_classification' saved to: {self.data_path.replace('.csv', '_Q3.csv')}")