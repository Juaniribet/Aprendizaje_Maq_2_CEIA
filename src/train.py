'''
train.py

Train script for TP ApMq2 - CEIA.

DESCRIPTION:

Train a Linear Regression model for Big Mark company to predict sales by product 
of a particular store.

Input data file:'outdata_train.csv' (from feature_engineering.py script).
Output model trained: sklearn Linear Regression (file: model_trained.pkl).


AUTHOR: Juan Ignacio Ribet.
DATE: 31-Jul-2023.
'''

# Imports
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression


class ModelTrainingPipeline():
    '''
    Train a Linear Regression model class.
    '''

    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self) -> pd.DataFrame:
        '''
        Read input file 'outdata_train.csv'.

        :return pandas_df: The desired DataLake table as a DataFrame.
        :rtype: pd.DataFrame.
        '''
        data = pd.read_csv(self.input_path, index_col=0)

        return data

    def model_training(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        '''
        Split data (70% train, 30% validation) and train the Linear Regression model.

        Returned metrics:   -TRAIN DATA: RMSE & R2.
                            -VALIDATION DATA: RMSE & R2.
                            -Intersection.
                            -Estimates Coefficient.
        '''
        model = LinearRegression()

        # Training and validation dataset split.
        features_data = dataframe.drop(columns='Item_Outlet_Sales')
        y_data = dataframe['Item_Outlet_Sales']
        x_train, x_val, y_train, y_val = train_test_split(features_data,
                                                          y_data,
                                                          test_size=0.3,
                                                          random_state=28)

        # Model Training.
        model.fit(x_train, y_train)

        # Root mean square errors and Coefficient of Determination.
        mse_train = metrics.mean_squared_error(y_train, model.predict(x_train))
        r2_train = model.score(x_train, y_train)
        print('Model Metrics:')
        print(f'TRAIN: RMSE: {(mse_train**0.5):.2f} - R2: {r2_train:.4f}')

        mse_val = metrics.mean_squared_error(y_val, model.predict(x_val))
        r2_val = model.score(x_val, y_val)
        print(f'VALIDATION: RMSE: {(mse_val**0.5):.2f} - R2: {r2_val:.4f}')

        print('\nModel Coefficients:')
        # Model constant.
        print(f'Intersection: {(model.intercept_):.2f}')

        # Model coefficients.
        coef = pd.DataFrame(x_train.columns, columns=['features'])
        coef['Estimates Coefficient'] = model.coef_
        print(coef, '\n')

        return model

    def model_dump(self, model_trained) -> None:
        '''
        Save trained model as 'model_trained.pkl' file.
        '''
        with open(self.model_path, 'wb') as file:
            pickle.dump(model_trained, file)

    def run(self):
        '''
        Run pipeline Training Pipeline.
        '''
        dataframe = self.read_data()
        model_trained = self.model_training(dataframe)
        self.model_dump(model_trained)


if __name__ == "__main__":
    # Local base directorys where the scripts are saved
    base_path, _ = os.path.split(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(base_path, os.pardir))

    ModelTrainingPipeline(input_path=base_path + '\\results\\outdata_train.csv',
                          model_path=base_path + '\\results\\model_trained.pkl').run()
