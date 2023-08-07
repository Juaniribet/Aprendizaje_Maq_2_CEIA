'''
feature_engineering.py

Feature engineering script for train data of the TP ApMq2 - CEIA.

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
DATE: 07-Ago-2023
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
    Replace values in a DataFrame column with specified dictionary values.

    :param dataframe: The DataFrame containing the column to be replaced.
    :type dataframe: pandas.DataFrame
    :param column: The name of the column to be replaced.
    :type column: str
    :param dic_data_replace: A dictionary mapping values to their replacements.
    :type dic_data_replace: dict

    This function replaces values in the specified column of the provided DataFrame
    with corresponding values from the provided dictionary.

    .. warning::
        This function modifies the input DataFrame in-place.
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
        Read data from csv files and return a pandas dataframe.

        :param self: The instance of the class.
        :return: DataFrame containing the raw data from the csv files.
        :rtype: pandas.DataFrame

        This method reads raw data from the csv files located at the path specified by
        the ``input_path`` attribute of the class instance. It then converts the data into
        a DataFrame and returns it.

        .. note::
            The ``input_path`` attribute of the class instance must be set before calling this 
            method.
        '''
        variables = ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
                     'Item_Type', 'Item_MRP', 'Outlet_Identifier',
                     'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
                     'Outlet_Type']

        for filename in os.listdir(self.input_path):
            _, file_format = os.path.splitext(filename)
            if file_format == '.csv':
                data = pd.read_csv(self.input_path + '/' + filename)
                if 'Item_Outlet_Sales' in data.columns:
                    data_train = data.copy()
                    data_train['Set'] = 'train'
                elif list(data.columns) == variables:
                    data_test = data.copy()
                    data_test['Set'] = 'test'

        data = pd.concat([data_train, data_test],
                         ignore_index=True, sort=False)

        variables.extend(['Item_Outlet_Sales', 'Set'])

        missing_col = [var for var in variables if var not in data.columns]

        # Check if there is any missing expected column for the data transformation function.
        if missing_col:
            error_messege = (f'Error: Colums missing in the dataset:  {missing_col}')
            logging.info(error_messege)
        else:
            pandas_df = data[variables]

        return pandas_df

    def data_transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Perform data transformation on input data.

        :param self: The instance of the class.
        :param data: The input DataFrame containing raw data.
        :type data: pandas.DataFrame

        :return: Transformed DataFrame representing the desired DataLake table.
        :rtype: pandas.DataFrame

        This method performs several transformations on the input data:

        1. Calculates the age of the establishment based on the year 2019.
        2. Fill null values in productos 'Item_Weight' using the mode of the 'Item_Weight'.
        3. Encodes 'Outlet_Size' using mapping ({'High': 2, 'Medium': 1, 'Small': 0}).
        4. Encodes 'Outlet_Location_Type' using mapping ({'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}).
        5. Codes 'Item_MRP' based on labels=[1, 2, 3, 4]
        6. Drops columns 'Item_Type' and 'Item_Fat_Content'.
        7. Encodes 'Outlet_Type' using pd.get_dummies function.
        8. Returns the transformed DataFrame.

        .. note::
            The function makes use of the `replace_data` function.
        '''
        # Determine the age of the Establisment by the year 2019.
        data['Outlet_Establishment_Year'] = 2020 - \
            data['Outlet_Establishment_Year']

        # Fill null values in productos 'Item_Weight'. Imputation of similar cases.
        items = list(data[data['Item_Weight'].isnull()]
                     ['Item_Identifier'].unique())
        for item in items:
            moda = (data[data['Item_Identifier'] == item]
                    [['Item_Weight']]).mode().iloc[0, 0]
            data.loc[data['Item_Identifier'] == item, 'Item_Weight'] = moda

        # Fill null values in Outlet_Size.
        outlets = list(data[data['Outlet_Size'].isnull()]
                       ['Outlet_Identifier'].unique())
        for outlet in outlets:
            data.loc[data['Outlet_Identifier'] ==
                     outlet, 'Outlet_Size'] = 'Small'

        # Modify object dtype varibles. Coding of ordinal variables.
        replace_data(data, 'Outlet_Size',
                     ({'High': 2, 'Medium': 1, 'Small': 0}))
        replace_data(data, 'Outlet_Location_Type',
                     ({'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}))

        # Coding máximum retailed price by labels.
        data['Item_MRP'] = pd.qcut(
            data['Item_MRP'], 4, labels=[1, 2, 3, 4])

        # Drop colums: 'Item_Type', 'Item_Fat_Content', 'Item_Identifier', 'Outlet_Identifier'.
        data = data.drop(columns=['Item_Type',
                                  'Item_Fat_Content',
                                  'Item_Identifier',
                                  'Outlet_Identifier'])

        data = pd.get_dummies(data, columns=['Outlet_Type'], dtype=int)

        df_transformed = data.copy()

        return df_transformed

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        '''
        Write prepared data to a CSV file.

        :param self: The instance of the class.
        :param transformed_dataframe: The DataFrame containing prepared data.
        :type transformed_dataframe: pandas.DataFrame

        This method writes the provided DataFrame containing prepared data to a CSV file.
        The file will be saved at the location specified by the ``output_path`` attribute
        of the class instance.

        :param transformed_dataframe: The DataFrame to be written.
        :type transformed_dataframe: pandas.DataFrame

        .. warning::
            This function will overwrite the CSV file if it already exists.

        .. note::
            The ``output_path`` attribute of the class instance must be set before calling this method.
        '''

        # Splitting the dataset in train and test.
        df_train = transformed_dataframe.loc[transformed_dataframe['Set'] == 'train']
        df_test = transformed_dataframe.loc[transformed_dataframe['Set'] == 'test']

        # Drop columns with no data.
        df_train_final = df_train.copy()
        df_train_final.drop(columns=['Set'], inplace=True)
        df_test_final = df_test.copy()
        df_test_final.drop(
            columns=['Item_Outlet_Sales', 'Set'], inplace=True)

        # Save the datasets.
        df_train_final.to_csv(self.output_path + "/outdata_train.csv")
        df_test_final.to_csv(self.output_path + "/outdata_Test.csv")

        logging.info('TRAIN DATA SUCCESSFULLY TRANSFORMED!!')

    def run(self):
        ''' 
        Execute the Feature Engineering pipeline.

        :param self: The instance of the class.
        :return: None

        This method orchestrates the execution of the Feature Engineering pipeline:

        1. Reads raw data using the :func:`read_data` method.
        2. Applies data transformation using the :func:`data_transformation` method.
        3. Writes the prepared data using the :func:`write_prepared_data` method.
        '''
        data_frame = self.read_data()
        df_transformed = self.data_transformation(data_frame)
        self.write_prepared_data(df_transformed)


if __name__ == "__main__":
    # Local base directorys where the scripts are saved
    base_path, _ = os.path.split(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(base_path, os.pardir))

    FeatureEngineeringPipeline(input_path=base_path + '\\data',
                               output_path=base_path + '\\results').run()
