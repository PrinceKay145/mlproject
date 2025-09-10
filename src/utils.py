import sys
import os
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, "wb")as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        # Iterate by model name to keep keys aligned and robust
        for model_name, model in models.items():
            param_grid = params.get(model_name, {})

            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions for evaluation
            y_test_pred = model.predict(X_test)
            # Train score computed but not used for selection; omit to avoid linter warning
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)