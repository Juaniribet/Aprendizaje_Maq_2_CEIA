"""
feature_engineering.py

Feature engineering script for TP ApMq2 - CEIA.

DESCRIPCIÓN:

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

Item_Weight
Item_Visibility
Item_MRP
Outlet_Establishment_Year
Outlet_Size
Outlet_Location_Type
Outlet_Type_Grocery Store
Outlet_Type_Supermarket Type1
Outlet_Type_Supermarket Type2
Outlet_Type_Supermarket Type3

AUTOR: Juan Ignacio Ribet
FECHA: 21-Jul-2023
"""

# Imports
import pandas as pd

class FeatureEngineeringPipeline():
    ''' 
    Data cleaninig and Features Engineering pipeline class
    '''

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """
        Read raw data from csv file. 
        
        -return pandas_df: The desired DataLake table as a DataFrame
        -rtype: pd.DataFrame
        """
        data = pd.read_csv(self.input_path)

        variables = ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
                    'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year', 
                    'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

        missing_col = [var for var in variables if var not in data.columns]

        #Check if there is any missing expected column for the data transformation function.
        if missing_col:
            print(f'Error: Colums missing in the dataset:  {missing_col}')
        else:
            if 'Item_Outlet_Sales' in data.columns:
                variables.append('Item_Outlet_Sales')

            pandas_df = data[variables]

            return pandas_df
        
        return None

    def data_transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Data transformation 
        '''
        # Determine the age of the Establisment by the year 2019
        data['Outlet_Establishment_Year'] = 2020 - data['Outlet_Establishment_Year']

        # Fill null values in productos 'Item_Weight'. Imputation of similar cases
        try:
            productos = list(data[data['Item_Weight'].isnull()]['Item_Identifier'].unique())
            for producto in productos:
                moda = (data[data['Item_Identifier'] == producto][['Item_Weight']]).mode().iloc[0,0]
                data.loc[data['Item_Identifier'] == producto, 'Item_Weight'] = moda
        finally:
            pass

        # Fill null values in Outlet_Size
        outlets = list(data[data['Outlet_Size'].isnull()]['Outlet_Identifier'].unique())
        for outlet in outlets:
            data.loc[data['Outlet_Identifier'] == outlet, 'Outlet_Size'] =  'Small'

        # Coding máximum retailed price by labels
        data['Item_MRP'] = pd.qcut(data['Item_MRP'], 4, labels = [1, 2, 3, 4])

        # Drop colums: 'Item_Type', 'Item_Fat_Content', 'Item_Identifier', 'Outlet_Identifier'
        data = data.drop(columns=['Item_Type',
                                  'Item_Fat_Content',
                                  'Item_Identifier', 
                                  'Outlet_Identifier'])

        # Modify object dtype varibles. Coding of ordinal variables.
        data['Outlet_Size'] = data['Outlet_Size'].replace({'High': 2,
                                                            'Medium': 1,
                                                            'Small': 0}
                                                            )
        data['Outlet_Location_Type'] = data['Outlet_Location_Type'].replace({'Tier 1': 2,
                                                                                        'Tier 2': 1,
                                                                                        'Tier 3': 0}
                                                                                        )
        df_transformed = pd.get_dummies(data, columns=['Outlet_Type'],dtype=int)

        return df_transformed

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        '''
        Files saved as csv format at the outputh_path location.
        -transformed_dataframe: pd.DataFrame
        '''

        transformed_dataframe.to_csv(self.output_path)

    def run(self):
        ''' 
        Run pipeline
        '''

        data_frame = self.read_data()
        df_transformed = self.data_transformation(data_frame)
        self.write_prepared_data(df_transformed)


if __name__ == "__main__":
    FeatureEngineeringPipeline(input_path = "C:/Users/juani/Documents/Especializacion IA/Aprendizaje de Maquina II/TP - AMq2/data/Train_BigMart.csv",
                               output_path = "C:/Users/juani/Documents/Especializacion IA/Aprendizaje de Maquina II/TP - AMq2/data/outdata_Train.csv").run()

    FeatureEngineeringPipeline(input_path = "C:/Users/juani/Documents/Especializacion IA/Aprendizaje de Maquina II/TP - AMq2/data/Test_BigMart.csv",
                            output_path = "C:/Users/juani/Documents/Especializacion IA/Aprendizaje de Maquina II/TP - AMq2/data/outdata_Test.csv").run()
