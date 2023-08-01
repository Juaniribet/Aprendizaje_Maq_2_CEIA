"""
predict.py

Prediction script for TP ApMq2 - CEIA.

DESCRIPTION: 
This scrip make the prediction of the item sales for a determined outlet.
It will return a csv file with the 'Item_Identifier', 'Outlet_Identifier' and
the Item_Outlet_Sales prediction.

AUTHOR: Juan Ignacio Ribet
DATE: 01-Ago-2023
"""

# Imports
import os
import pickle
import pandas as pd

class MakePredictionPipeline():
    '''
    Make prediction with the trained model class.
    '''

    def __init__(self, input_path, output_path, model_path: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
        self.item = None
        self.outlet = None
        self.model = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the data for predictions and save the 'Item_Identifier' and 'Outlet_Identifier.
        """
        data = pd.read_csv(self.input_path, index_col=0)
        # Save 'Item_Identifier' and 'Outlet_Identifier' for the return data.
        self.item = data['Item_Identifier']
        self.outlet = data['Outlet_Identifier']
        # Drop 'Item_Identifier' and 'Outlet_Identifier' to get the data 
        # ready to make the prediction
        data = data.drop(columns=['Item_Identifier', 'Outlet_Identifier'])

        return data

    def load_model(self) -> None:
        """
        Load the trained model.
        """
        with open(self.model_path, 'rb') as file:
            self.model = pickle.load(file)


    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make the sale prediction with the load data and the load model.
        """
        # Make the prediction.
        pred = self.model.predict(data)
        print(f'The predicted sales of the item {self.item[0]}',
              f'in the outlet {self.outlet[0]} are: {int(pred[0])}')

        # compile the prediction with the identifiers.
        new_data = pd.DataFrame({'Item_Identifier': self.item,
                                 'Outlet_Identifier': self.outlet,
                                 'Pred_Item_Outlet_Sales' : pred})

        return new_data


    def write_predictions(self, predicted_data: pd.DataFrame) -> None:
        """
        Save the the predction and the identifiers in an scv file.
        """
        predicted_data.to_csv(self.output_path)


    def run(self):
        '''
        Run pipeline Predict Pipeline.
        '''
        data = self.load_data()
        self.load_model()
        df_preds = self.make_predictions(data)
        self.write_predictions(df_preds)


if __name__ == "__main__":
    # Local base directorys where the scripts are saved
    base_path, _ = os.path.split(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(base_path, os.pardir))

    pipeline = MakePredictionPipeline(input_path = base_path + '\\results\\example_df.csv',
                                      output_path = base_path + '\\results\\predicted_data.csv',
                                      model_path = base_path + '\\results\\model_trained.pkl')
    pipeline.run()
