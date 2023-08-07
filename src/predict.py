"""
predict.py

Prediction script for TP ApMq2 - CEIA.

DESCRIPTION: 
This scrip make the prediction of the item sales for a determined outlet.
It will return a csv file with the 'Item_Identifier', 'Outlet_Identifier' and
the Item_Outlet_Sales prediction.

AUTHOR: Juan Ignacio Ribet
DATE: 07-Ago-2023
"""

# Imports
import os
import logging
import pickle
import pandas as pd

path, _ = os.path.split(os.path.abspath(__file__))
logging.basicConfig(
    filename= os.path.abspath(os.path.join(path, os.pardir)) + '\\results\\logging_info.log',
    level=logging.INFO,
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

class MakePredictionPipeline():
    """
    Make prediction with the trained model class.
    """

    def __init__(self, input_path, output_path, model_path: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
        self.item = None
        self.outlet = None
        self.model = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the data for predictions and save the 'Item_Identifier' and 'Outlet_Identifier'.

        :return: A pandas DataFrame containing the loaded data for predictions.
        :rtype: pd.DataFrame
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
        Load a trained machine learning model from the specified file path and assign it to
        the instance variable 'model'.
        """
        with open(self.model_path, 'rb') as file:
            self.model = pickle.load(file)

    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using the trained model on the provided input data.

        :param data: A pandas DataFrame containing the input data for making predictions.
        :type data: pd.DataFrame
        :return: A pandas DataFrame containing the predictions along with item and outlet 
        identifiers.
        :rtype: pd.DataFrame
        """
        # Make the prediction.
        pred = self.model.predict(data)
        messenge = (f'The predicted sales of the item {self.item[0]}',
              f'in the outlet {self.outlet[0]} are: {int(pred[0])}')
        
        logging.info(messenge)

        # compile the prediction with the identifiers.
        new_data = pd.DataFrame({'Item_Identifier': self.item,
                                 'Outlet_Identifier': self.outlet,
                                 'Pred_Item_Outlet_Sales': pred})

        return new_data

    def write_predictions(self, predicted_data: pd.DataFrame) -> None:
        """
        Write the predicted data to a CSV file at the specified output path.

        :param predicted_data: A pandas DataFrame containing the predicted data.
        :type predicted_data: pd.DataFrame
        """
        predicted_data.to_csv(self.output_path)

    def run(self):
        """
        Run the complete workflow for loading data, loading a trained model, making predictions,
        and writing the results.

        This function orchestrates the following steps:
        1. Load input data for predictions.
        2. Load a trained model.
        3. Generate predictions using the loaded model.
        4. Write the prediction results to an output file.
        """
        data = self.load_data()
        self.load_model()
        df_preds = self.make_predictions(data)
        self.write_predictions(df_preds)


if __name__ == "__main__":
    # Local base directorys where the scripts are saved
    base_path, _ = os.path.split(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(base_path, os.pardir))

    pipeline = MakePredictionPipeline(input_path=base_path + '\\results\\example_df.csv',
                                      output_path=base_path + '\\results\\predicted_data.csv',
                                      model_path=base_path + '\\results\\model_trained.pkl')
    pipeline.run()
