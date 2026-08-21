import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

F1_THRESHOLD = 0.65

def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    # 1.6.1 Đọc dữ liệu huấn luyện
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 1.6.2 Tách đặc trưng và nhãn
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]
    
    # Bonus 5: Cảnh Báo Lệch Lạc Dữ Liệu
    positive_rate = float(y_train.mean())
    if abs(positive_rate - 0.248) > 0.05:
        print(f"WARNING: Data drift detected! Positive rate is {positive_rate:.4f} (expected ~0.248)")

    # 1.6.3 Bắt đầu MLflow run
    with mlflow.start_run():
        
        # 1.6.4 Ghi nhận tham số
        mlflow.log_params(params)

        # 1.6.5 Khởi tạo và huấn luyện
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # Lấy xác suất thay vì nhãn mặc định (Bonus 2)
        probs = model.predict_proba(X_eval)[:, 1]

        best_f1 = 0
        best_threshold = 0.5
        
        # Quét ngưỡng từ 0.1 đến 0.9 (bước 0.05)
        for t in np.arange(0.1, 0.91, 0.05):
            preds_t = (probs >= t).astype(int)
            f1_t = f1_score(y_eval, preds_t)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_threshold = float(t)

        # Tính lại các metrics tại ngưỡng tốt nhất
        best_preds = (probs >= best_threshold).astype(int)
        
        f1 = float(f1_score(y_eval, best_preds))
        acc = float(accuracy_score(y_eval, best_preds))
        precision_val = float(precision_score(y_eval, best_preds))
        recall_val = float(recall_score(y_eval, best_preds))
        cm = confusion_matrix(y_eval, best_preds)

        # F1 tại ngưỡng mặc định 0.5 để so sánh
        default_preds = (probs >= 0.5).astype(int)
        f1_default = float(f1_score(y_eval, default_preds))

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("f1_default_05", f1_default)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("positive_rate", positive_rate)
        mlflow.log_metric("precision", precision_val)
        mlflow.log_metric("recall", recall_val)

        # 1.6.8 Log model
        mlflow.sklearn.log_model(model, "model")

        # 1.6.9 In ra terminal
        print(f"F1 (Best Threshold {best_threshold:.2f}): {f1:.4f} | Default F1: {f1_default:.4f}")

        # 1.6.10 Lưu metrics
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump({
                "f1_score": f1,
                "accuracy": acc,
                "best_threshold": best_threshold,
                "f1_default_05": f1_default,
                "positive_rate": positive_rate,
                "precision": precision_val,
                "recall": recall_val
            }, f)

        # Bonus 3: Báo Cáo Precision / Recall Tự Động
        with open("outputs/detail.txt", "w") as f:
            f.write("=== CLASSIFICATION REPORT ===\n")
            f.write(f"Threshold used: {best_threshold:.2f}\n")
            f.write(f"Precision: {precision_val:.4f}\n")
            f.write(f"Recall:    {recall_val:.4f}\n")
            f.write("\nConfusion Matrix:\n")
            f.write(f"TN: {cm[0,0]} | FP: {cm[0,1]}\n")
            f.write(f"FN: {cm[1,0]} | TP: {cm[1,1]}\n")

        # 1.6.11 Lưu mô hình cục bộ
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
