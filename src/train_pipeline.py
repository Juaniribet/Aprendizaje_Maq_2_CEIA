"""
train_pipeline.py

DESCRIPTION: Run feature_engineering.py and train.py scripts of the TP ApMq2 - CEIA.

AUTHOR: Juan Ignacio Ribet
DATE: 31-Jul-2023
"""
import subprocess

subprocess.run(['Python', '..\\Aprendizaje_Maq_2_CEIA\\src\\feature_engineering.py'], check=True)

subprocess.run(['Python', '..\\Aprendizaje_Maq_2_CEIA\\src\\train.py'], check=True)
