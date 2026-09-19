import os
import sys
import json
import re
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union

from src.logger import logging
from src.exception import SpamhamException

DEFAULT_SPAM_TRIGGERS = [
    "free", "winner", "win", "prize", "cash", "claim", "urgent", "credit card", "bank",
    "password", "account", "verify", "mobile", "txt", "call", "click", "http", "https",
    "www", "guaranteed", "offer", "discount", "100%", "cheap", "meds", "viagra", "money",
    "congratulations", "bonus", "selected", "ringtone", "dating", "stop", "reply"
]

class PredictionPipeline:
    def __init__(self, model_path: str = "artifacts/best_model.pkl", metrics_path: str = "artifacts/model_metrics.json"):
        self.model_path = model_path
        self.metrics_path = metrics_path
        self._model = None
        self._spam_keywords = set(DEFAULT_SPAM_TRIGGERS)
        self._load_spam_keywords()

    def _load_spam_keywords(self):
        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, "r") as f:
                    data = json.load(f)
                    kw = data.get("top_spam_keywords", [])
                    if kw:
                        for k in kw:
                            clean_k = k.strip().lower()
                            if len(clean_k) > 2 and clean_k.isalnum():
                                self._spam_keywords.add(clean_k)
            except Exception as e:
                logging.warning(f"Could not load keywords from metrics: {e}")

    def get_trained_model(self):
        if self._model is not None:
            return self._model

        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self._model = pickle.load(f)
                return self._model
            except Exception as e:
                logging.error(f"Failed to load local model artifact: {e}")

        # Fallback to S3 estimator if available
        try:
            from src.ml.model.s3_estimator import SpamhamDetector
            from src.entity.config_entity import PredictionPipelineConfig
            prediction_config = PredictionPipelineConfig()
            self._model = SpamhamDetector(
                bucket_name=prediction_config.model_bucket_name,
                model_path=prediction_config.model_file_name
            )
            return self._model
        except Exception as e:
            raise SpamhamException(f"No trained model artifact found at {self.model_path} and S3 fallback failed: {e}", sys)

    def run_pipeline(self, input_data: Union[List[str], str]) -> List[int]:
        """
        Returns simple binary list [0 or 1] for backward compatibility.
        """
        try:
            if isinstance(input_data, str):
                input_data = [input_data]
            model = self.get_trained_model()
            
            # If standard sklearn pipeline
            if hasattr(model, "predict"):
                preds = model.predict(input_data)
                return [int(p) for p in preds]
            return [0] * len(input_data)
        except Exception as e:
            raise SpamhamException(e, sys) from e

    def predict_detailed(self, text: str) -> Dict[str, Any]:
        """
        Returns detailed prediction result with probability score, risk level,
        flagged keywords, and text analysis metrics.
        """
        try:
            text_str = str(text or "").strip()
            if not text_str:
                return {
                    "is_spam": False,
                    "label": "Ham",
                    "prediction": 0,
                    "spam_probability": 0.0,
                    "confidence_percent": 100.0,
                    "risk_level": "Safe",
                    "flagged_keywords": [],
                    "text_stats": {
                        "char_count": 0,
                        "word_count": 0,
                        "uppercase_pct": 0.0,
                        "url_count": 0
                    }
                }

            model = self.get_trained_model()
            spam_prob = 0.5
            
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([text_str])[0]
                # Assuming index 1 is spam
                spam_prob = float(probs[1])
            elif hasattr(model, "decision_function"):
                df_val = model.decision_function([text_str])[0]
                spam_prob = float(1.0 / (1.0 + np.exp(-df_val)))
            else:
                pred = self.run_pipeline([text_str])[0]
                spam_prob = 0.95 if pred == 1 else 0.05

            is_spam = spam_prob >= 0.5
            confidence_pct = round((spam_prob if is_spam else (1.0 - spam_prob)) * 100, 1)
            
            if spam_prob >= 0.75:
                risk_level = "High Risk"
            elif spam_prob >= 0.45:
                risk_level = "Moderate Risk"
            else:
                risk_level = "Safe"

            # Extract flagged keywords
            words_in_text = set(re.findall(r'\b[a-zA-Z0-9]+\b', text_str.lower()))
            flagged = sorted(list(words_in_text.intersection(self._spam_keywords)))

            # Text statistics
            words = text_str.split()
            char_count = len(text_str)
            word_count = len(words)
            letters = [c for c in text_str if c.isalpha()]
            upper_pct = round((sum(1 for c in letters if c.isupper()) / max(len(letters), 1)) * 100, 1)
            url_count = len(re.findall(r'https?://\S+|www\.\S+', text_str, re.IGNORECASE))

            return {
                "is_spam": is_spam,
                "label": "Spam" if is_spam else "Ham",
                "prediction": 1 if is_spam else 0,
                "spam_probability": round(spam_prob, 4),
                "confidence_percent": confidence_pct,
                "risk_level": risk_level,
                "flagged_keywords": flagged,
                "text_stats": {
                    "char_count": char_count,
                    "word_count": word_count,
                    "uppercase_pct": upper_pct,
                    "url_count": url_count
                }
            }

        except Exception as e:
            raise SpamhamException(e, sys) from e

            
            
        
            
        

 
        

        