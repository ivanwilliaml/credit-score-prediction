import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,mean_absolute_error, mean_squared_error, r2_score)
from pathlib import Path

BASE_DIR = Path(__file__).parent

def evaluate(run_id):
    test_df = pd.read_csv(BASE_DIR / "test.csv")
    
    X_test = test_df.drop(columns=['placement_status', 'salary_package_lpa'])
    y_clf_test = test_df['placement_status']
    y_reg_test = test_df['salary_package_lpa']

    # Evaluasi Classification Model
    clf_model = mlflow.sklearn.load_model(f"runs:/{run_id}/clf_model")
    
    y_clf_pred = clf_model.predict(X_test)
    y_clf_prob = clf_model.predict_proba(X_test)[:, 1]
    
    clf_acc = accuracy_score(y_clf_test, y_clf_pred)
    clf_f1 = f1_score(y_clf_test, y_clf_pred)
    clf_prec = precision_score(y_clf_test, y_clf_pred)
    clf_rec = recall_score(y_clf_test, y_clf_pred)
    clf_auc = roc_auc_score(y_clf_test, y_clf_prob)

    # Evaluasi Regression Model
    placed_mask = y_clf_test == 1
    X_test_placed = X_test[placed_mask]
    y_reg_test_placed = y_reg_test[placed_mask]

    reg_model = mlflow.sklearn.load_model(f"runs:/{run_id}/reg_model")
    
    y_reg_pred = reg_model.predict(X_test_placed)
    
    reg_mae = mean_absolute_error(y_reg_test_placed, y_reg_pred)
    reg_rmse = np.sqrt(mean_squared_error(y_reg_test_placed, y_reg_pred))
    reg_r2 = r2_score(y_reg_test_placed, y_reg_pred)
    
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            "test_clf_accuracy": clf_acc,"test_clf_f1": clf_f1,"test_clf_precision": clf_prec,"test_clf_recall": clf_rec,"test_clf_auc": clf_auc,
            "test_reg_mae": reg_mae,"test_reg_rmse": reg_rmse,"test_reg_r2": reg_r2
        })

    print("Evaluation Report")
    print("Classification - Placement Status")
    print(f"Best Classification Model : {type(clf_model.steps[-1][1]).__name__ if hasattr(clf_model, 'steps') else type(clf_model).__name__}")
    print(f"Accuracy : {clf_acc:.4f}")
    print(f"F1-Score : {clf_f1:.4f}")
    print(f"AUC-ROC  : {clf_auc:.4f}")
    print("")
    print("Regression - Salary Package (LPA)")
    print(f"Best Regression Model     : {type(reg_model.steps[-1][1]).__name__ if hasattr(reg_model, 'steps') else type(reg_model).__name__}")
    print(f"MAE      : {reg_mae:.4f}")
    print(f"RMSE     : {reg_rmse:.4f}")
    print(f"R² Score : {reg_r2:.4f}")

    return {
        "clf_f1": clf_f1, 
        "reg_mae": reg_mae
    }

if __name__ == "__main__":
    pass