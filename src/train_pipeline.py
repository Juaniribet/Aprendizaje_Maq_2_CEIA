"""
train_pipeline.py

DESCRIPTION: Run feature_engineering.py and train.py scripts of the TP ApMq2 - CEIA.

AUTHOR: Juan Ignacio Ribet
DATE: 01-Ago-2023
"""
import os
import subprocess
import feature_engineering as fe

# Local base directorys where the scripts are saved
base_path, _ = os.path.split(os.path.abspath(__file__))
base_path = os.path.abspath(os.path.join(base_path, os.pardir))

fe.FeatureEngineeringPipeline(input_path=base_path + '\\data',
                              output_path=base_path + '\\results').run()

subprocess.run(['Python', base_path + '\\src\\train.py'], check=True)
