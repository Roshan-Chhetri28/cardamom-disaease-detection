import sys
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
from PIL import ImageFile
import cv2 as cv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    @staticmethod
    def extract_colorHOG(image):
        """
        Extracts Color Histogram features from an image.
        """
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            
            # Ensure image is in the correct format
            if image.dtype == np.float32 or image.max() <= 1.0:
                image = (image * 255).astype('uint8')

            image = cv.cvtColor(image, cv.COLOR_RGB2HSV)

            # Calculate histograms for each HSV channel
            hist_h = cv.calcHist([image], [0], None, [256], [0, 256]).flatten()
            hist_s = cv.calcHist([image], [1], None, [256], [0, 256]).flatten()
            hist_v = cv.calcHist([image], [2], None, [256], [0, 256]).flatten()

            # Concatenate the histograms
            hist_features = np.concatenate((hist_h, hist_s, hist_v))
            
            return hist_features
        except Exception as e:
            raise CustomException(f"Error in extract_colorHOG: {e}", sys)

    def hog_dataset(self, generator):
        """
        Processes a dataset generator to extract Color HOG features and labels.
        """
        try:
            hist_features_list = []
            labels_list = []

            for batch_images, batch_labels in generator:
                for img, label in zip(batch_images, batch_labels):
                    hist_feature = self.extract_colorHOG(img)
                    hist_features_list.append(hist_feature)
                    labels_list.append(label)

                # Break if we've processed all samples in the generator
                if len(hist_features_list) >= generator.samples:
                    break

            return np.array(hist_features_list), np.array(labels_list)
        except Exception as e:
            raise CustomException(f"Error in hog_dataset: {e}", sys)

    def get_data_transformer_obj(self):
        """
        Creates and returns a data transformation pipeline object.
        """
        try:
            logging.info("Creating data transformation pipeline.")
            
            pipeline = Pipeline(
                steps=[
                    ('scaler', StandardScaler())
                ]
            )
            
            save_obj(self.data_transformation_config.preprocessor_obj_file_path, pipeline)
            
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data, test_data):
        """
        Applies the transformation pipeline to the dataset.
        """
        try:
            logging.info("Initiating data transformation.")

            # Extract features and labels
            X_train, y_train = self.hog_dataset(train_data)
            X_test, y_test = self.hog_dataset(test_data)
            
            logging.info("Applying the pipeline")
            pipeline = self.get_data_transformer_obj()
            train_transformed_features = pipeline.fit_transform(X_train)
            test_transformed_features = pipeline.fit_transform(X_test)
            
            logging.info('data transformation complete')
            return train_transformed_features, y_train, test_transformed_features, y_test
        except Exception as e:
            raise CustomException(e, sys)
