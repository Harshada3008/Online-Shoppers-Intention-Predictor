"""Run the full CSV-based ML pipeline and print MLflow dashboard instructions."""

import os
from dotenv import load_dotenv

load_dotenv()

from ml.train import train
from ml.evaluate import evaluate


def main():
    data_path = os.getenv('DATA_PATH')
    print("Running project pipeline with the following configuration:")
    print(f"  DATA_PATH = {data_path}")
    print(f"  MODEL_PATH = {os.getenv('MODEL_PATH')}")
    print(f"  MLFLOW_TRACKING_URI = {os.getenv('MLFLOW_TRACKING_URI')}")
    print()

    try:
        model, _, _, best_name = train()
        print(f"Trained best model: {best_name}")
    except Exception as e:
        print("Error during training:", e)
        raise

    try:
        metrics = evaluate()
        print("Evaluation completed. Metrics:")
        print(f"  Accuracy: {metrics[0]:.4f}")
        print(f"  Precision: {metrics[1]:.4f}")
        print(f"  Recall: {metrics[2]:.4f}")
        print(f"  F1 Score: {metrics[3]:.4f}")
        print(f"  ROC AUC: {metrics[4]:.4f}")
    except Exception as e:
        print("Error during evaluation:", e)
        raise

    print()
    print("MLflow runs were logged to:", os.getenv('MLFLOW_TRACKING_URI'))
    print("To open the MLflow dashboard, run:")
    print("  mlflow ui --backend-store-uri ./mlruns --port 5000")
    print("Then open http://127.0.0.1:5000 in your browser.")


if __name__ == '__main__':
    main()
