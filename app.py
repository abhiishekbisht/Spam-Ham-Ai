import os
import json
import asyncio
from typing import List, Optional
from fastapi import FastAPI, Request, Form, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from uvicorn import run as app_run
from dotenv import load_dotenv

load_dotenv()

from src.pipeline.prediction_pipeline import PredictionPipeline
from train_and_export import train_and_export_model

app = FastAPI(
    title="Spam & Ham ML Intelligence Platform",
    description="State-of-the-art Spam Detection & Text Analysis Engine powered by Machine Learning",
    version="1.0.0"
)

# Static & Templates setup
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
class SinglePredictRequest(BaseModel):
    text: str = Field(..., example="WINNER! You have won a free iPhone. Click here to claim.")

class BatchPredictRequest(BaseModel):
    messages: List[str] = Field(..., example=["Hey lunch today?", "URGENT: Password reset required"])

class TextStatsSchema(BaseModel):
    char_count: int
    word_count: int
    uppercase_pct: float
    url_count: int

class DetailedPredictionResponse(BaseModel):
    is_spam: bool
    label: str
    prediction: int
    spam_probability: float
    confidence_percent: float
    risk_level: str
    flagged_keywords: List[str]
    text_stats: TextStatsSchema

# Global pipeline instance
pipeline = PredictionPipeline()

# WEB ROUTES
@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse(request=request, name="prediction.html", context={"context": False})

@app.post("/predict")
async def predict_legacy_form(request: Request, input_text: Optional[str] = Form(None)):
    try:
        if not input_text:
            form = await request.form()
            input_text = form.get("input_text", "")
            
        res = pipeline.predict_detailed(input_text or "")
        return templates.TemplateResponse(
            request=request,
            name="prediction.html",
            context={
                "context": True,
                "prediction": res["prediction"],
                "detailed_result": res,
                "input_text": input_text
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="prediction.html",
            context={"context": False, "error": str(e)}
        )


# API ENDPOINTS (v1)
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "SpamHam-ML", "version": "2.0.0"}

@app.post("/api/v1/predict", response_model=DetailedPredictionResponse)
async def api_predict_single(payload: SinglePredictRequest):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    return pipeline.predict_detailed(payload.text)

@app.post("/api/v1/predict-batch")
async def api_predict_batch(payload: BatchPredictRequest):
    if not payload.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")
    results = [pipeline.predict_detailed(msg) for msg in payload.messages]
    return {"count": len(results), "results": results}

@app.post("/api/v1/upload-csv")
async def api_upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    try:
        import pandas as pd
        content = await file.read()
        import io
        df = pd.read_csv(io.BytesIO(content))
        text_col = None
        for col in df.columns:
            if "message" in col.lower() or "text" in col.lower() or "body" in col.lower() or "content" in col.lower():
                text_col = col
                break
        if not text_col:
            text_col = df.columns[0]

        messages = df[text_col].dropna().astype(str).tolist()
        results = [pipeline.predict_detailed(msg) for msg in messages[:200]]  # limit to top 200 for rapid display
        
        spam_count = sum(1 for r in results if r["is_spam"])
        ham_count = len(results) - spam_count
        
        return {
            "filename": file.filename,
            "total_processed": len(results),
            "spam_count": spam_count,
            "ham_count": ham_count,
            "spam_percentage": round((spam_count / max(len(results), 1)) * 100, 1),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")

@app.get("/api/v1/model-info")
async def api_model_info():
    metrics_path = "artifacts/model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {
        "model_name": "LogisticRegression / MultinomialNB",
        "status": "Model active",
        "note": "Train metrics artifact will refresh upon retraining."
    }

def _async_retrain_task():
    try:
        train_and_export_model()
        global pipeline
        pipeline = PredictionPipeline()
    except Exception as e:
        print(f"Background training failed: {e}")

@app.post("/api/v1/train")
@app.get("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(_async_retrain_task)
    return {
        "status": "success",
        "message": "Model retraining initiated in the background.",
        "timestamp": asyncio.get_event_loop().time()
    }

if __name__ == "__main__":
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", 8080))
    app_run("app:app", host=host, port=port, reload=True)

    
