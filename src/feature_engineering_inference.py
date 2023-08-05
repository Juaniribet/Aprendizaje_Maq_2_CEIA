'''
feature_engineering.py

Feature engineering script for inference data of the TP ApMq2 - CEIA.

DESCRIPTION:

Data cleaninig and Features Engineering of the Big Mark data for the prediction of sales by product 
of each particular stores

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
DATE: 05-Ago-2023
'''

# Imports
import os
import logging
import pandas as pd

path, _ = os.path.split(os.path.abspath(__file__))
logging.basicConfig(
    filename=os.path.abspath(os.path.join(
        path, os.pardir)) + '\\results\\logging_info.log',
    level=logging.INFO,
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


def replace_data(dataframe: pd.DataFrame, column: str, dic_data_repalce: dict):
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
    '''

    def __init__(self, output_path, input_path):
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        '''
        Read raw data from .json file.

        -return pandas_df: The desired DataLake table as a DataFrame.
        -rtype: pd.DataFrame.
        '''
        example = pd.read_json(self.input_path, typ='series')
        example = pd.DataFrame(
            data=[example.values], columns=example.index)
        pandas_df = example.copy()

        return pandas_df

    def data_transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Data transformation.
        '''
        # Determine the age of the Establisment by the year 2019.
        data['Outlet_Establishment_Year'] = 2020 - \
            data['Outlet_Establishment_Year']

        # Modify object dtype varibles. Coding of ordinal variables.
        replace_data(data, 'Outlet_Size',
                     ({'High': 2, 'Medium': 1, 'Small': 0}))
        replace_data(data, 'Outlet_Location_Type',
                     ({'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}))

        # Coding máximum retailed price by labels.
        ranges = [(31.288999999999998, 94.012), (94.012, 142.247),
                  (142.247, 185.856), (185.856, 266.888)]
        data['Item_MRP'] = [ranges.index(r)+1 for r in ranges
                            if min(r) <= data['Item_MRP'][0] <= max(r)][0]

        # Drop colums: 'Item_Type', 'Item_Fat_Content'.
        data = data.drop(columns=['Item_Type', 'Item_Fat_Content',])

        # similar get_dummies for predict data
        outlet_type = ['Grocery Store', 'Supermarket Type1',
                       'Supermarket Type2', 'Supermarket Type3']

        for outlet in outlet_type:
            data['Outlet_Type_'+outlet] = 0
            if outlet == data['Outlet_Type'][0]:
                data['Outlet_Type_'+outlet] = 1
        data = data.drop(columns=['Outlet_Type'])

        df_transformed = data.copy()

        return df_transformed

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        '''
        Files saved as csv format at the outputh_path location.
        -transformed_dataframe: pd.DataFrame.
        '''
        transformed_dataframe.to_csv(self.output_path + "/example_df.csv")

        logging.info('INFERENCE DATA SUCCESSFULLY TRANSFORMED!!')

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

    FeatureEngineeringPipeline(input_path=base_path + '\\Notebook\\example.json',
                               output_path=base_path + '\\results').run()
