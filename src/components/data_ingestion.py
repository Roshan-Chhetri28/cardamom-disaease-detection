import os
import sys

from src.exception import CustomException
from src.logger import logging

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestion:
    def __init__(self):
        pass
    
    def initiate_data_ingestion(self):
        # D:\Cardamon_d\data\Test
        logging.info("Started DataIngestion Method")

        try:
            self.train_data_path = "D:/Cardamon_d/data/Train"
            self.test_data_path = "D:/Cardamon_d/data/Test"
            self.train_imagegen = ImageDataGenerator(
            rescale =  1./255,
            horizontal_flip=True,
            rotation_range=30,
            zoom_range=0.2,
            vertical_flip = True,
            width_shift_range = 0.2,
            height_shift_range = 0.2
            )
            
            self.test_imagegen = ImageDataGenerator(
                rescale = 1./255
            )

            self.train = self.train_imagegen.flow_from_directory(
                self.train_data_path,
                shuffle = True,
                target_size = (255, 255),
                batch_size = 32,
                class_mode = 'categorical'  
            )
            self.test = self.test_imagegen.flow_from_directory(
                self.test_data_path,
                shuffle = True,
                target_size = (255, 255),
                batch_size = 32,
                class_mode = 'categorical'  
            )
            return (
                self.train,
                self.test
            )

        except Exception as e:
            raise CustomException(e, sys)
    
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train, y_train, X_test, y_test = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()

    r2_score = model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)

    print(f"r2_score : {r2_score}")