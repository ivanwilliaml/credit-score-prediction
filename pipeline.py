from data_ingestion import ingest_data
from split import split_data
from train import train
from evaluation import evaluate

CLF_F1_THRESHOLD = 0.85
REG_MAE_THRESHOLD = 5.0

def run_pipeline():
    print("Step 1: Data Ingestion")
    ingest_data()

    print("Step 2: Split Data")
    split_data()

    print("Step 3: Training (Classification & Regression)")
    run_id = train()

    print("Step 4: Evaluation")
    metrics = evaluate(run_id)

    f1 = metrics["clf_f1"]
    mae = metrics["reg_mae"]

    if f1 >= CLF_F1_THRESHOLD and mae <= REG_MAE_THRESHOLD:
        print(f"Model approved for deployment | F1 ({f1:.3f}) | MAE ({mae:.3f})|")
    else:
        print(f"Model rejected. | F1 ({f1:.3f}) | MAE ({mae:.3f})|")

if __name__ == "__main__":
    run_pipeline()