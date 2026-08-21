from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

# Đọc tên bucket từ biến môi trường (được đặt trong systemd service)
ARTIFACT_BUCKET = os.environ.get("ARTIFACT_BUCKET", "default-bucket")
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser("~/models/model.joblib")


def download_model():
    """Tải file model.joblib từ cloud storage về máy khi server khởi động."""
    try:
        client = storage.Client()
        bucket = client.bucket(ARTIFACT_BUCKET)
        blob = bucket.blob(MODEL_KEY)
        blob.download_to_filename(MODEL_PATH)
        print(f"Tải thành công model từ gs://{ARTIFACT_BUCKET}/{MODEL_KEY}")
    except Exception as e:
        print(f"Có lỗi khi khởi tạo tải model: {e}")


# Gọi hàm này khi module được import (chạy khi server khởi động)
download_model()
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    """Endpoint kiểm tra sức khỏe server. GitHub Actions dùng endpoint này để xác nhận triển khai thành công."""
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    """
    Endpoint suy luận.

    Đầu vào: JSON {"features": [f1, f2, ..., f10]}
    Đầu ra:  JSON {"prediction": <0|1>, "label": <"thu_nhap_thap"|"thu_nhap_cao">}
    """
    if len(req.features) != 10:
        raise HTTPException(status_code=400, detail="Expected 10 features (adult income)")

    # Numpy array từ đầu vào 
    pred = int(model.predict([req.features])[0])
    label = "thu_nhap_cao" if pred == 1 else "thu_nhap_thap"
    
    return {"prediction": pred, "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
