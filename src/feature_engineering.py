'''
feature_engineering.py

Feature engineering script for TP ApMq2 - CEIA.

DESCRIPTION:

Data cleaninig and Features Engineering of the Big Mark data for the prediction of sales by product 
of each particular stores

Input files names must be 'Train_BigMart.csv' and 'Test_BigMart.csv'
Output files manes: 'outdata_train.csv', 'outdata_test.csv'

Imput data variables:

- Item_Identifier: product name or identifier.
- Item_Weight: product weight (in grams).
- Item_Fat_Content: classification of the product in terms of fats contained.
- Item_Visibility: product visibility scoring.
- Item_Type: product type.
- Item_MRP: Máximum retailed price. 
- Outlet_Identifier: store identifier.
- Outlet_Establishment_Year: launch year of the store.
- Outlet_Size: store size
- Outlet_Location_Type: store classification by location.
- Outlet_Type: store type 
- Item_Outlet_Sales: ventas del producto en cada observacion

Output data variables:

- Item_Weight : (dtype:float)
- Item_Visibility : (dtype:float)
- Item_MRP : labels - (1, 2, 3, 4)
- Outlet_Establishment_Year : (dtype:int)
- Outlet_Size : (0, 1, 2) - {'High': 2,'Medium': 1,'Small': 0}
- Outlet_Location_Type : (0, 1, 2) - {'Tier 1': 2,'Tier 2': 1,'Tier 3': 0}
- Outlet_Type_Grocery Store : (0, 1)
- Outlet_Type_Supermarket Type1 : (0, 1)
- Outlet_Type_Supermarket Type2 : (0, 1)
- Outlet_Type_Supermarket Type3 : (0, 1)

AUTHOR: Juan Ignacio Ribet
DATE: 31-Jul-2023
'''

# Imports
import os
import pandas as pd

def replace_data(dataframe: pd.DataFrame, columns:str, dic_data_repalce: dict):
    dataframe[columns] = dataframe[columns].replace(dic_data_repalce)

class FeatureEngineeringPipeline():
    ''' 
    Data cleaninig and Features Engineering class
    '''

    def __init__(self, output_path, input_path):
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        '''
        Read raw data from csv file. 
        The files names must be 'Train_BigMart.csv' and 'Test_BigMart.csv' for train and test data.
        
        -return pandas_df: The desired DataLake table as a DataFrame.
        -rtype: pd.DataFrame.
        '''

        for filename in os.listdir(self.input_path):
            if filename == 'Train_BigMart.csv':
                data_train = pd.read_csv(self.input_path + '/' + filename)
                data_train['Set'] = 'train'
            elif filename == 'Test_BigMart.csv':
                data_test = pd.read_csv(self.input_path + '/' + filename)
                data_test['Set'] = 'test'

        data = pd.concat([data_train, data_test], ignore_index=True, sort=False)

        variables = ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
            'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year', 
            'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type','Item_Outlet_Sales','Set']

        missing_col = [var for var in variables if var not in data.columns]

        #Check if there is any missing expected column for the data transformation function.
        if missing_col:
            print(f'Error: Colums missing in the dataset:  {missing_col}')
        else:
            pandas_df = data[variables]

            return pandas_df

        return None

    def data_transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Data transformation.
        '''
        # Determine the age of the Establisment by the year 2019.
        data['Outlet_Establishment_Year'] = 2020 - data['Outlet_Establishment_Year']

        # Fill null values in productos 'Item_Weight'. Imputation of similar cases.
        items = list(data[data['Item_Weight'].isnull()]['Item_Identifier'].unique())
        for item in items:
            moda = (data[data['Item_Identifier'] == item][['Item_Weight']]).mode().iloc[0,0]
            data.loc[data['Item_Identifier'] == item, 'Item_Weight'] = moda

        # Fill null values in Outlet_Size.
        outlets = list(data[data['Outlet_Size'].isnull()]['Outlet_Identifier'].unique())
        for outlet in outlets:
            data.loc[data['Outlet_Identifier'] == outlet, 'Outlet_Size'] =  'Small'

        # Coding máximum retailed price by labels.
        data['Item_MRP'] = pd.qcut(data['Item_MRP'], 4, labels = [1, 2, 3, 4])

        # Drop colums: 'Item_Type', 'Item_Fat_Content', 'Item_Identifier', 'Outlet_Identifier'.
        data = data.drop(columns=['Item_Type',
                                  'Item_Fat_Content',
                                  'Item_Identifier', 
                                  'Outlet_Identifier'])

        # Modify object dtype varibles. Coding of ordinal variables.
        replace_data(data, 'Outlet_Size', ({'High': 2,'Medium': 1,'Small': 0}))

        replace_data(data, 'Outlet_Location_Type', ({'Tier 1': 2,'Tier 2': 1,'Tier 3': 0}))

        df_transformed = pd.get_dummies(data, columns=['Outlet_Type'],dtype=int)

        return df_transformed

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        '''
        Files saved as csv format at the outputh_path location.
        -transformed_dataframe: pd.DataFrame.
        '''
        # Splitting the dataset in train and test.
        df_train = transformed_dataframe.loc[transformed_dataframe['Set'] == 'train']
        df_test = transformed_dataframe.loc[transformed_dataframe['Set'] == 'test']

        # Drop columns with no data.
        df_train_final = df_train.copy()
        df_train_final.drop(columns=['Set'], inplace=True)
        df_test_final = df_test.copy()
        df_test_final.drop(columns=['Item_Outlet_Sales','Set'], inplace=True)

        # Save the datasets.
        df_train_final.to_csv(self.output_path + "/outdata_train.csv")
        df_test_final.to_csv(self.output_path + "/outdata_Test.csv")

    def run(self):
        ''' 
        Run Feature Engineering pipeline.
        '''
        data_frame = self.read_data()
        df_transformed = self.data_transformation(data_frame)
        self.write_prepared_data(df_transformed)


if __name__ == "__main__":
    FeatureEngineeringPipeline(input_path = '..\\Aprendizaje_Maq_2_CEIA\\data',
                            output_path = '..\\Aprendizaje_Maq_2_CEIA\\data').run()
