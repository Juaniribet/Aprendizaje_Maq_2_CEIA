"""
inference_pipeline.py

DESCRIPTION: Run feature_engineering.py for predict data and tpredict.py scripts of the 
TP ApMq2 - CEIA.

AUTHOR: Juan Ignacio Ribet
DATE: 01-Ago-2023
"""
import os
import subprocess

# Local base directorys where the scripts are saved
base_path, _ = os.path.split(os.path.abspath(__file__))
base_path = os.path.abspath(os.path.join(base_path, os.pardir))

# As it is for prediction: 'predict_data = True'
subprocess.run(
    ['Python', base_path + '\\src\\feature_engineering_inference.py'], check=True)

subprocess.run(['Python', base_path + '\\src\\predict.py'], check=True)
