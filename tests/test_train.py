import os
import json
import numpy as np
import pandas as pd
from src.train import train

FEATURE_NAMES = [
    "age", "workclass", "education_num", "marital_status", "occupation",
    "relationship", "sex", "capital_gain", "capital_loss", "hours_per_week",
]

def _make_temp_data(tmp_path):
    """
    Tạo dataset nhỏ với cùng schema Adult để sử dụng trong test.
    """
    rng = np.random.default_rng(0)
    n = 200
    
    # Tạo giá trị số ngẫu nhiên cho feature, và giá trị 0 hoặc 1 cho target
    X = rng.random((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 2, n)
    
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y
    
    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "holdout.csv"
    
    # Lưu ra files
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)
    
    return str(train_path), str(eval_path)

def test_train_returns_float(tmp_path):
    """Kiểm tra hàm train() trả về một số thực trong khoảng [0, 1]."""
    train_path, eval_path = _make_temp_data(tmp_path)
    
    res = train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )
    
    assert isinstance(res, float)
    assert 0.0 <= res <= 1.0

def test_report_file_created(tmp_path):
    """Kiểm tra file outputs/report.json được tạo sau khi huấn luyện."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )
    
    assert os.path.exists("outputs/report.json")
    with open("outputs/report.json", "r") as f:
        data = json.load(f)
    assert "f1_score" in data
    assert "accuracy" in data

def test_model_file_created(tmp_path):
    """Kiểm tra file models/model.joblib được tạo sau khi huấn luyện."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert os.path.exists("models/model.joblib")
