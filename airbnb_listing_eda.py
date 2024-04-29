import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class AirbnbListingEDA:

    def __init__(self):
        self.file_path = None
        self.data = None

    def read_file_path(self):
        """
        Prompt the user to input the file path.
        """
        while True:
            try:
                file_path = input("Enter the path to the data file: ")
                if not os.path.isfile(file_path):
                    raise FileNotFoundError("File not found.")
                self.file_path = file_path
                break
            except FileNotFoundError as e:
                print("Error:", e)

    def read_data(self):
        """
        Read data from the specified file based on its format (CSV, Excel, or JSON).
        """
        if self.file_path.endswith('.csv'):
            self.data = pd.read_csv(self.file_path)
        elif self.file_path.endswith('.xlsx') or self.file_path.endswith('.xls'):
            self.data = pd.read_excel(self.file_path)
        elif self.file_path.endswith('.json'):
            self.data = pd.read_json(self.file_path)
        else:
            raise ValueError("Unsupported file format. Supported formats: CSV, Excel, JSON.")
        return self.data

    def clean_data(self):
        """
        Perform data cleaning operations on the loaded dataset.
        """
        print("Summation of null values before cleaning: \n", self.data.isna().sum(), "\n")
        columns_to_drop = ['neighbourhood_group', 'license']
        self.data.drop(columns_to_drop, axis=1, inplace=True)
        self.data.fillna(method='ffill', inplace=True)
        self.data['price'] = self.data['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)
        print("Summation of null values after cleaning: \n", self.data.isna().sum(), "\n")
        print("Data shape after cleaning:", self.data.shape)
        self.data.info()
        return self.data

    def visualize_data(self):
        """
        Visualize the data to identify patterns and relationships between variables.
        """
        plt.figure(figsize=(8, 6))
        sns.histplot(data=self.data, x="price", kde=True)
        plt.title("Price Distribution of Airbnb listings in the selected city")
        plt.xlabel("Price (in USD)")
        plt.ylabel("Count")
        plt.show()

        plt.figure(figsize=(8, 6))
        sns.countplot(data=self.data, x='room_type')
        plt.title("Distribution of Airbnb Listing Property Types")
        plt.xlabel("Property Type")
        plt.ylabel("Count")
        plt.show()

        plt.figure(figsize=(10, 8))
        sns.boxplot(data=self.data, x='neighbourhood', y='price')
        plt.title("Price by Neighbourhood")
        plt.xlabel("Neighbourhood")
        plt.ylabel("Price (in USD)")
        plt.xticks(rotation=90)
        plt.show()

        plt.figure(figsize=(10, 8))
        self.data['year_month'] = pd.to_datetime(self.data['last_review']).dt.to_period('M')
        availability = self.data.groupby(['year_month'])['availability_365'].mean().reset_index()
        availability['year_month'] = availability['year_month'].dt.to_timestamp()
        plt.plot(availability['year_month'], availability['availability_365'])
        plt.title("Average Availability of Airbnb listings over time")
        plt.xlabel("Year/Month")
        plt.ylabel("Availability (in days)")
        plt.xticks(rotation=90)
        plt.show()

    def conduct_analysis(self):
        """
        Conduct statistical analysis to validate the trends and patterns observed in the data.
        """
        numerical_data = self.data.select_dtypes(include=[np.number])
        corr_matrix = numerical_data.corr()
        sns.heatmap(corr_matrix, annot=True)
        plt.title("Correlation Matrix of Key Variables in the Dataset")
        plt.show()

    def suggest_business_plan(self):
        """
        Suggest a business plan based on the insights gained from the analysis.
        """
        high_demand_neighborhoods = self.data['neighbourhood'].value_counts().idxmax()
        high_demand_property_types = self.data['room_type'].value_counts().idxmax()
        peak_times = self.data[self.data['availability_365'] < self.data['availability_365'].mean()]['year_month'].value_counts().idxmax()
        avg_price_neighborhood = self.data[self.data['neighbourhood'] == high_demand_neighborhoods]['price'].mean()
        avg_price_property_type = self.data[self.data['room_type'] == high_demand_property_types]['price'].mean()
        recommendation = f"Based on the analysis, recommend targeting the neighborhood {high_demand_neighborhoods} with an average price of {avg_price_neighborhood} or property type {high_demand_property_types} with an average price of {avg_price_property_type} for higher demand and a better price-to-performance ratio. Additionally, consider increasing marketing efforts during {peak_times} when availability is lower and demand is higher."
        return recommendation


if __name__ == "__main__":
    airbnb_eda = AirbnbListingEDA()
    airbnb_eda.read_file_path()
    df = airbnb_eda.read_data()
    
    print("\nDataFrame Information:")
    df.info()
    print("\n\n")

    data_cleaned = airbnb_eda.clean_data()
    print("")
    print(data_cleaned)
    print("")

    airbnb_eda.visualize_data()
    
    airbnb_eda.conduct_analysis()
    print(airbnb_eda.suggest_business_plan())
    print("")
