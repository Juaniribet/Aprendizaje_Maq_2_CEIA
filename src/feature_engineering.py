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

def replace_data(dataframe: pd.DataFrame, column:str, dic_data_repalce: dict):
    '''
    Funtion to code ordinal variables of the 'dataframe'.
    column : column name to modify to code ordinal values
    dic_data_repalce: dtype: dict 
    '''
    dataframe[column] = dataframe[column].replace(dic_data_repalce)

class FeatureEngineeringPipeline():
    ''' 
    Data cleaninig and Features Engineering class
    input_path : Directory to load the data to be transform.
    output_path : Directory to save the data transformed.
    predict_data : If the data is for inference 'predict_data' should be 'True'. Default: False
    '''

    def __init__(self, output_path, input_path, predict_data = False):
        self.input_path = input_path
        self.output_path = output_path
        self.predict_data = predict_data

    def read_data(self) -> pd.DataFrame:
        '''
        Read raw data from csv file. 
        The files names must be 'Train_BigMart.csv' and 'Test_BigMart.csv' for train and test data.
        
        -return pandas_df: The desired DataLake table as a DataFrame.
        -rtype: pd.DataFrame.
        '''
        # For predict data:
        if self.predict_data:
            example = pd.read_json(self.input_path, typ='series')
            example = pd.DataFrame(data=[example.values], columns= example.index)

            return example

        # For train data:
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

        # Modify object dtype varibles. Coding of ordinal variables.
        replace_data(data, 'Outlet_Size', ({'High': 2,'Medium': 1,'Small': 0}))
        replace_data(data, 'Outlet_Location_Type', ({'Tier 1': 2,'Tier 2': 1,'Tier 3': 0}))

        # For predict data:
        if self.predict_data:
            # Coding máximum retailed price by labels.
            ranges = [(31.288999999999998, 94.012), (94.012, 142.247),
                      (142.247, 185.856), (185.856, 266.888)]
            data['Item_MRP'] = [ranges.index(r)+1 for r in ranges
                                if min(r) <= data['Item_MRP'][0] <= max(r)][0]

            #Drop colums: 'Item_Type', 'Item_Fat_Content'.
            data = data.drop(columns=['Item_Type', 'Item_Fat_Content',])
            outlet_type = ['Grocery Store','Supermarket Type1', 
                           'Supermarket Type2','Supermarket Type3']

            # similar get_dummies for predict data
            for outlet in outlet_type:
                data['Outlet_Type_'+outlet] = 0
                if outlet == data['Outlet_Type'][0]:
                    data['Outlet_Type_'+outlet] = 1
            data = data.drop(columns=['Outlet_Type'])

        # For train data:
        else:
            # Coding máximum retailed price by labels.

            data['Item_MRP'] = pd.qcut(data['Item_MRP'], 4, labels = [1, 2, 3, 4])

            # Drop colums: 'Item_Type', 'Item_Fat_Content', 'Item_Identifier', 'Outlet_Identifier'.
            data = data.drop(columns=['Item_Type',
                                    'Item_Fat_Content',
                                    'Item_Identifier', 
                                    'Outlet_Identifier'])

            data = pd.get_dummies(data, columns=['Outlet_Type'],dtype=int)

        df_transformed = data.copy()

        return df_transformed

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        '''
        Files saved as csv format at the outputh_path location.
        -transformed_dataframe: pd.DataFrame.
        '''
        # For predict data:
        if self.predict_data:
            transformed_dataframe.to_csv(self.output_path + "/example_df.csv")
        # For train data:
        else:
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

            print('¡¡DATA SUCCESSFULLY TRANSFORMED!!')

    def run(self):
        ''' 
        Run Feature Engineering pipeline.
        '''
        data_frame = self.read_data()
        df_transformed = self.data_transformation(data_frame)
        self.write_prepared_data(df_transformed)


if __name__ == "__main__":
    # Local base directorys where the scripts are saved
    base_path, _ = os.path.split(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(base_path, os.pardir))

    FeatureEngineeringPipeline(input_path = base_path + '\\data',
                            output_path = base_path + '\\results').run()
