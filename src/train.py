"""
train.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
"""

# Imports
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression


class ModelTrainingPipeline():

    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING 
        
        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """
            
        data = pd.read_csv(self.input_path, index_col=0)
        
        return data

    
    def model_training(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        
        """
        
        seed = 28
        model = LinearRegression()

        # División de dataset de entrenaimento y validación
        features_data = dataframe.drop(columns='Item_Outlet_Sales')
        y_data = dataframe['Item_Outlet_Sales']
        x_train, x_val, y_train, y_val = train_test_split(features_data,
                                                          y_data,
                                                          test_size = 0.3,
                                                          random_state=seed)

        # Entrenamiento del modelo
        model.fit(x_train,y_train)

        # Predicción del modelo ajustado para el conjunto de validación
        pred = model.predict(x_val)

        # Cálculo de los errores cuadráticos medios y Coeficiente de Determinación (R^2)
        mse_train = metrics.mean_squared_error(y_train, model.predict(x_train))
        R2_train = model.score(x_train, y_train)
        print('Métricas del Modelo:')
        print('ENTRENAMIENTO: RMSE: {:.2f} - R2: {:.4f}'.format(mse_train**0.5, R2_train))

        mse_val = metrics.mean_squared_error(y_val, pred)
        R2_val = model.score(x_val, y_val)
        print('VALIDACIÓN: RMSE: {:.2f} - R2: {:.4f}'.format(mse_val**0.5, R2_val))

        print('\nCoeficientes del Modelo:')
        # Constante del modelo
        print('Intersección: {:.2f}'.format(model.intercept_))

        # Coeficientes del modelo
        coef = pd.DataFrame(x_train.columns, columns=['features'])
        coef['Coeficiente Estimados'] = model.coef_
        print(coef, '\n')
        
        return model

    def model_dump(self, model_trained) -> None:
        """
        COMPLETAR DOCSTRING
        
        """
        
        with open(self.model_path, 'wb') as file:
            pickle.dump(model_trained, file)
        
        return None

    def run(self):
    
        dataframe = self.read_data()
        model_trained = self.model_training(dataframe)
        self.model_dump(model_trained)

if __name__ == "__main__":

    ModelTrainingPipeline(input_path = "Aprendizaje_Maq_2_CEIA\data\outdata_Train.csv",
                          model_path = "Aprendizaje_Maq_2_CEIA\data\model_trained.pkl").run()