import os
import sys
from dataclasses import dataclass
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler


@dataclass
class ModelTrainerConfig:
    trainer_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Splitting Training and Test data')
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1] 
            )
            base_estimator = DecisionTreeClassifier(max_depth=1)

            models = {
                "Logistic Regression":make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=2000, C=0.5)),
                "Logistic Regression":LogisticRegression(penalty="l2",C=1.0),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(estimator=base_estimator,n_estimators=50,algorithm='SAMME')
            }

            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score <0.6:
                raise CustomException("No best model Found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trainer_model_file_path,
                obj = best_model
            )
            return best_model
        except Exception as e:
            raise CustomException(e,sys)