import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

import os
from src.utils import load_obj

class PredictPipeline:
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path =  os.path.join("artifacts", 'proprocessor.pkl')

            logging.info("Loading Model")

            model = load_obj(file_path = model_path)
            preprocessor_path = load_obj(file_path = preprocessor_path)



        except Exception as e:
            raise CustomException(e, sys)