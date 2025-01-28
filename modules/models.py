import time
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import GridSearchCV


class Models:
    def __init__(self, model_name):
        self.param_grids = {
            "Logistic Regression": {
                "model": LogisticRegression(),
                "params": {
                    "C": [0.01, 1, 10],
                    "solver": ["liblinear", "sag", "lbfgs"]
                }
            },
            "Random Forest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 7, 10]
                }
            },
            "SVM": {
                "model": SVC(),
                "params": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf", "poly"]
                }
            },
            "XGBoost": {
                "model": XGBClassifier(eval_metric="mlogloss"),
                "params": {
                    "learning_rate": [0.01, 0.1, 0.3],
                    "n_estimators": [100, 300, 500]
                }
            }
        }

        if model_name not in self.param_grids:
            raise ValueError(f"Model {model_name} is not supported. "
                             f"Choose from: {list(self.param_grids.keys())}")

        self.model_name = model_name
        self.model_info = self.param_grids[model_name]
        self.model = self.model_info["model"]
        self.params = self.model_info["params"]

    def train_and_evaluate(self, X_train, y_train, X_val, y_val):

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.params,
            scoring='accuracy',
            cv=5,              
            n_jobs=-1,         
            return_train_score=True  
        )
        

        start_train = time.time()
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_train
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_best_score = grid_search.best_score_


        train_predictions = best_model.predict(X_train)
        train_score = accuracy_score(y_train, train_predictions)

        start_val = time.time()
        val_predictions = best_model.predict(X_val)
        val_score = accuracy_score(y_val, val_predictions)
        val_time = time.time() - start_val

        results = {
            "Algorithm": self.model_name,
            "Best Hyperparameters": best_params,
            "CV Mean Accuracy (Best)": cv_best_score,    
            "Training Accuracy (Full)": train_score,     
            "Validation Accuracy": val_score,
            "Total Training Time (s)": train_time,
            "Validation Prediction Time (s)": val_time
        }
        
        return results
