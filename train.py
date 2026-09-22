import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMClassifier

BASE_DIR = Path(__file__).parent
Path("artifacts").mkdir(exist_ok=True)
SEED = 42

# Customized transformer (encoding dan feature engineering)
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X):
        X_new = X.copy()
        
        # Encoding 
        if 'gender' in X_new.columns:
            X_new['gender'] = X_new['gender'].map({'Male': 1, 'Female': 0})
        if 'extracurricular_activities' in X_new.columns:
            X_new['extracurricular_activities'] = X_new['extracurricular_activities'].map({'Yes': 1, 'No': 0})
            
        # Feature Engineering / Extracting
        X_new['total_experience'] = (X_new['internship_count'] + X_new['live_projects'] + (X_new['work_experience_months'] / 12))
        X_new['skill_score_combined'] = (X_new['technical_skill_score'] + X_new['soft_skill_score']) / 2
        X_new['academic_score_avg'] = (X_new['ssc_percentage'] + X_new['hsc_percentage'] + X_new['degree_percentage'] + X_new['cgpa'] * 10) / 4
        X_new['engagement_score'] = (X_new['attendance_percentage'] + X_new['extracurricular_activities'])
        
        X_new.fillna(0, inplace=True)
        return X_new


def train():
    mlflow.set_experiment("MLflow-Pipeline")
    train_df = pd.read_csv(BASE_DIR / "train.csv")
    
    TARGET_CLF = 'placement_status'
    TARGET_REG = 'salary_package_lpa'
    
    X_train = train_df.drop(columns=[TARGET_CLF, TARGET_REG])
    y_clf_train = train_df[TARGET_CLF]
    y_reg_train = train_df[TARGET_REG]

    with mlflow.start_run() as run:
        # A. Classification Pipeline
        clf_configs = { 'LightGBM': 
           (
                LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, class_weight='balanced'),
                {'n_estimators': [100, 200], 'max_depth': [5, 8, -1], 'learning_rate': [0.05, 0.1, 0.2], 'num_leaves': [31, 63]}
            )
        }

        best_clf_score = -1
        best_clf_pipeline = None
        best_clf_name = ""

        for name, (model, params) in clf_configs.items():
            pipeline = Pipeline([
                ('fe', FeatureEngineer()),
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            
            pipeline_params = {f"classifier__{k}": v for k, v in params.items()}
            
            search = RandomizedSearchCV(pipeline, pipeline_params, n_iter=12, cv=5, scoring='f1', n_jobs=-1, random_state=SEED)
            search.fit(X_train, y_clf_train)
            
            print(f"[{name}] CV F1-Score: {search.best_score_:.4f}")
            
            if search.best_score_ > best_clf_score:
                best_clf_score = search.best_score_
                best_clf_pipeline = search.best_estimator_
                best_clf_name = name

        print(f"Classification Model: {best_clf_name}")

        # B. Regression Pipeline
        placed_mask = y_clf_train == 1
        X_reg_train = X_train[placed_mask].copy()
        y_reg_train_f = y_reg_train[placed_mask]

        reg_configs = {
            'Random Forest': (
                RandomForestRegressor(random_state=SEED, n_jobs=-1),
                {'n_estimators': [100, 200], 'max_depth': [6, 10, None], 'min_samples_split': [2, 5], 'max_features': ['sqrt', 'log2']}
            )
        }

        best_reg_score = float('inf')
        best_reg_pipeline = None
        best_reg_name = ""

        for name, (model, params) in reg_configs.items():
            pipeline = Pipeline([
                ('fe', FeatureEngineer()),
                ('scaler', StandardScaler()),
                ('regressor', model)
            ])
            
            pipeline_params = {f"regressor__{k}": v for k, v in params.items()}
            
            search = RandomizedSearchCV(pipeline, pipeline_params, n_iter=12, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=SEED)
            search.fit(X_reg_train, y_reg_train_f)
            
            mae_score = -search.best_score_ 
            print(f"[{name}] CV MAE: {mae_score:.4f}")
            
            if mae_score < best_reg_score:
                best_reg_score = mae_score
                best_reg_pipeline = search.best_estimator_
                best_reg_name = name

        print(f"Regression Model: {best_reg_name}")

        mlflow.log_metrics({"best_cv_clf_f1": best_clf_score,"best_cv_reg_mae": best_reg_score})
        mlflow.log_params({"best_clf_model": best_clf_name,"best_reg_model": best_reg_name})

        joblib.dump(best_clf_pipeline, "artifacts/best_clf_pipeline.pkl")
        joblib.dump(best_reg_pipeline, "artifacts/best_reg_pipeline.pkl")

        mlflow.sklearn.log_model(best_clf_pipeline, "clf_model")
        mlflow.sklearn.log_model(best_reg_pipeline, "reg_model")

        return run.info.run_id

if __name__ == "__main__":
    train()