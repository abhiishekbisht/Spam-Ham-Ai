import sys
from typing import List, Tuple
import os
from pandas import DataFrame
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact

from src.exception import SpamhamException
from src.logger import logging
from src.utils.main_utils import MainUtils, load_numpy_array_data


class SpamhamDetectionModel:
    def __init__(self, preprocessing_object: object, encoder_object: object, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.encoder_object = encoder_object
        self.trained_model_object = trained_model_object

    def predict(self, X: DataFrame) -> DataFrame:
        logging.info("Entered predict method of SpamhamDetectionModel class")
        try:
            transformed_feature = self.preprocessing_object.transform(X)
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            raise SpamhamException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:
    def __init__(self, 
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.utils = MainUtils()

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            x_train, y_train, x_test, y_test = train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1]
            
            clf = LogisticRegression(C=5.0, max_iter=1000)
            clf.fit(x_train, y_train)
            
            preds = clf.predict(x_test)
            f1 = f1_score(y_test, preds, zero_division=0)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            
            preprocessing_obj = self.utils.load_object(file_path=self.data_transformation_artifact.transformed_vectorizer_object_file_path)
            encoder_object = self.utils.load_object(file_path=self.data_transformation_artifact.transformed_encoder_object_file_path)
             
            model_wrapper = SpamhamDetectionModel(
                preprocessing_object=preprocessing_obj,
                encoder_object=encoder_object,
                trained_model_object=clf
            )
            
            trained_model_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(trained_model_path, exist_ok=True)
            
            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model_wrapper
            )
            
            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=prec, recall_score=rec)
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )

            logging.info("Model training completed successfully")
            return model_trainer_artifact

        except Exception as e:
            raise SpamhamException(e, sys) from e

