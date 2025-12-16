from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
import joblib
import uvicorn
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download, snapshot_download
import torch
import os
import csv
import io

app = FastAPI(
    title="Kazakh Sentiment Analyzer API",
    description="API for analyzing sentiment of Kazakh-language reviews",
    version="1.0.0"
)

# CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== HUGGING FACE MODEL LOADING =====
HF_REPO = "jasikKarim/my-ml-models"

def get_model_path(filename: str) -> str:
    """Download and return path to a model file"""
    return hf_hub_download(repo_id=HF_REPO, filename=filename, repo_type="model")

def get_transformer_path(subfolder: str) -> str:
    """Download and return path to transformer model folder"""
    path = snapshot_download(repo_id=HF_REPO, allow_patterns=f"{subfolder}/*", repo_type="model")
    return os.path.join(path, subfolder)
# =====================================

# Request model
class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Review text in Kazakh")
    model: str = Field(default="svm", description="Model to use for prediction")

# Batch request model
class BatchReviewRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of review texts in Kazakh")
    model: str = Field(default="svm", description="Model to use for prediction")

# Response model
class SentimentResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    sentiment: str
    probabilities: Dict[str, float]
    predicted_rating: Optional[float] = None
    text: str
    model_used:  str

# Batch response model
class BatchSentimentResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    results: List[SentimentResponse]
    total:  int
    summary: Dict[str, int]
    model_used: str


# SVM Model class
class SVMModel:
    """Real SVM model loaded from joblib files"""
    
    def __init__(self):
        self.name = "SVM"
        try:
            self.model = joblib.load(get_model_path("SVM.joblib"))
            self.vectorizer = joblib.load(get_model_path("SVM_vector.joblib"))
            self.metadata = joblib.load(get_model_path("SVM_metadata.joblib"))
            print(f"✓ SVM model loaded successfully")
            print(f"✓ Vectorizer loaded successfully")
            print(f"✓ Metadata loaded successfully")
            
            if hasattr(self.metadata, 'get'):
                self.label_mapping = self.metadata.get('label_mapping', {0: 'negative', 1: 'neutral', 2: 'positive'})
            else:
                self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
                
        except Exception as e:
            print(f"Error loading SVM model: {e}")
            raise
    
    def predict(self, text: str) -> Dict:
        try:
            text_vectorized = self.vectorizer.transform([text])
            prediction = self.model.predict(text_vectorized)[0]
            
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(text_vectorized)[0]
            elif hasattr(self.model, 'decision_function'):
                decision = self.model.decision_function(text_vectorized)[0]
                exp_scores = np.exp(decision - np.max(decision))
                probas = exp_scores / exp_scores.sum()
            else:
                probas = np.zeros(len(self.label_mapping))
                probas[prediction] = 1.0
            
            sentiment = self.label_mapping.get(prediction, 'neutral')
            
            prob_dict = {}
            for idx, label in self.label_mapping.items():
                prob_dict[label] = float(probas[idx]) if idx < len(probas) else 0.0
            
            rating = (
                prob_dict.get('positive', 0) * 5.0 +
                prob_dict.get('neutral', 0) * 3.0 +
                prob_dict.get('negative', 0) * 1.0
            )
            
            return {
                'sentiment': sentiment,
                'probabilities': prob_dict,
                'predicted_rating': rating
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            raise


# Logistic Model class
class LogisticModel: 
    """Logistic model loaded from joblib files"""
    
    def __init__(self):
        self.name = "Logistic"
        try: 
            self.model = joblib.load(get_model_path("linear_model.joblib"))
            self.vectorizer = joblib.load(get_model_path("linear_vector.joblib"))
            self.metadata = joblib.load(get_model_path("linear_metadata.joblib"))
            print(f"✓ Logistic model loaded successfully")
            print(f"✓ Vectorizer loaded successfully")
            print(f"✓ Metadata loaded successfully")
            
            if hasattr(self.metadata, 'get'):
                self.label_mapping = self.metadata.get('label_mapping', {0: 'negative', 1: 'neutral', 2: 'positive'})
            else:
                self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
                
        except Exception as e:
            print(f"Error loading Logistic model:  {e}")
            raise
    
    def predict(self, text: str) -> Dict:
        try:
            text_vectorized = self.vectorizer.transform([text])
            prediction = self.model.predict(text_vectorized)[0]
            
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(text_vectorized)[0]
            elif hasattr(self.model, 'decision_function'):
                decision = self.model.decision_function(text_vectorized)[0]
                exp_scores = np.exp(decision - np.max(decision))
                probas = exp_scores / exp_scores.sum()
            else:
                probas = np.zeros(len(self.label_mapping))
                probas[prediction] = 1.0
            
            sentiment = self.label_mapping.get(prediction, 'neutral')
            
            prob_dict = {}
            for idx, label in self.label_mapping.items():
                prob_dict[label] = float(probas[idx]) if idx < len(probas) else 0.0
            
            rating = (
                prob_dict.get('positive', 0) * 5.0 +
                prob_dict.get('neutral', 0) * 3.0 +
                prob_dict.get('negative', 0) * 1.0
            )
            
            return {
                'sentiment':  sentiment,
                'probabilities':  prob_dict,
                'predicted_rating': rating
            }
            
        except Exception as e: 
            print(f"Prediction error: {e}")
            raise


# Transformer Model class
class TransformerModel:
    """XLM-RoBERTa model for Kazakh sentiment analysis"""
    
    def __init__(self):
        self.name = "XLM-RoBERTa"
        self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.max_length = 128
        self.device = torch.device("cpu")
        
        try: 
            model_path = get_transformer_path("transformer_model")
            print(f"Loading transformer model from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Transformer model loaded successfully")
            print(f"✓ Model parameters: {self.model.num_parameters():,}")
            print(f"✓ Device:  {self.device}")
            
        except Exception as e:
            print(f"Error loading transformer model: {e}")
            raise
    
    def predict(self, text: str) -> Dict:
        try: 
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()
            
            sentiment = self.label_mapping.get(pred, 'neutral')
            
            prob_dict = {
                'negative': float(probs[0][0].item()),
                'neutral': float(probs[0][1].item()),
                'positive': float(probs[0][2].item())
            }
            
            rating = (
                prob_dict['positive'] * 5.0 +
                prob_dict['neutral'] * 3.0 +
                prob_dict['negative'] * 1.0
            )
            
            return {
                'sentiment': sentiment,
                'probabilities': prob_dict,
                'predicted_rating': rating
            }
            
        except Exception as e:
            print(f"Transformer prediction error: {e}")
            raise


# New Aza Transformer Model class
class AzaTransformerModel:
    """Alternative XLM-RoBERTa model (Aza_model) for Kazakh sentiment analysis"""
    
    def __init__(self):
        self.name = "XLM-RoBERTa (Aza)"
        self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.max_length = 128
        self.device = torch.device("cpu")
        
        try:
            model_path = get_transformer_path("Aza_model")
            print(f"Loading Aza transformer model from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Aza transformer model loaded successfully")
            print(f"✓ Aza model parameters: {self.model.num_parameters():,}")
            print(f"✓ Aza device: {self.device}")
        except Exception as e:
            print(f"Error loading Aza transformer model: {e}")
            raise
    
    def predict(self, text: str) -> Dict:
        try:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()
            
            sentiment = self.label_mapping.get(pred, 'neutral')
            
            prob_dict = {
                'negative': float(probs[0][0].item()),
                'neutral': float(probs[0][1].item()),
                'positive': float(probs[0][2].item())
            }
            
            rating = (
                prob_dict['positive'] * 5.0 +
                prob_dict['neutral'] * 3.0 +
                prob_dict['negative'] * 1.0
            )
            
            return {
                'sentiment': sentiment,
                'probabilities': prob_dict,
                'predicted_rating': rating
            }
        except Exception as e:
            print(f"Aza transformer prediction error: {e}")
            raise


# Initialize models
models = {}

# Load SVM model
try:
    svm_model = SVMModel()
    models['svm'] = svm_model
    print(f"✓ SVM model initialized successfully")
except Exception as e: 
    print(f"✗ Failed to initialize SVM model: {e}")

# Load Logistic model
try:
    logistic_model = LogisticModel()
    models['logistic'] = logistic_model
    print(f"✓ Logistic model initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize Logistic model: {e}")

# Load Transformer model
try:
    transformer_model = TransformerModel()
    models['transformer'] = transformer_model
    print(f"✓ Transformer model initialized successfully")
except Exception as e: 
    print(f"✗ Failed to initialize Transformer model:  {e}")

# Load Aza Transformer model
try:
    aza_transformer_model = AzaTransformerModel()
    models['aza_transformer'] = aza_transformer_model
    print(f"✓ Aza transformer model initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize Aza transformer model: {e}")


@app.get("/")
async def root():
    return {
        "message": "Kazakh Sentiment Analyzer API",
        "version": "1.0.0",
        "available_models": list(models.keys()),
        "endpoints": {
            "predict": "/api/predict (POST)",
            "batch_predict": "/api/batch/predict (POST)",
            "batch_csv": "/api/batch/csv (POST)",
            "models":  "/api/models (GET)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if models else "degraded",
        "models_loaded": list(models.keys()),
        "total_models": len(models)
    }

@app.get("/api/models")
async def get_available_models():
    available_models = []
    
    if 'svm' in models: 
        available_models.append({
            "id": "svm",
            "name": "Support Vector Machine (SVM)",
            "description":  "SVM classifier with TF-IDF vectorization for Kazakh text",
            "type": "sklearn",
            "status": "loaded"
        })
    
    if 'logistic' in models: 
        available_models.append({
            "id": "logistic",
            "name": "Logistic Regression",
            "description": "Logistic regression classifier with TF-IDF vectorization for Kazakh text",
            "type": "sklearn",
            "status": "loaded"
        })
    
    if 'transformer' in models:
        available_models.append({
            "id": "transformer",
            "name": "XLM-RoBERTa",
            "description": "Fine-tuned XLM-RoBERTa model for Kazakh sentiment analysis",
            "type": "deep_learning",
            "status":  "loaded"
        })

    if 'aza_transformer' in models:
        available_models.append({
            "id": "aza_transformer",
            "name": "XLM-RoBERTa (Aza)",
            "description": "Alternative fine-tuned XLM-RoBERTa model (Aza_model), with improved neutral performance",
            "type": "deep_learning",
            "status": "loaded"
        })
    
    return {"models": available_models}

@app.post("/api/predict", response_model=SentimentResponse)
async def predict_sentiment(request: ReviewRequest):
    try:
        if not models:
            raise HTTPException(
                status_code=503,
                detail="Models not loaded.  Please check server logs."
            )
        
        if request.model not in models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model.  Choose from:  {list(models.keys())}"
            )
        
        selected_model = models[request.model]
        prediction = selected_model.predict(request.text)
        
        response = SentimentResponse(
            sentiment=prediction['sentiment'],
            probabilities=prediction['probabilities'],
            predicted_rating=prediction['predicted_rating'],
            text=request.text[: 100],
            model_used=selected_model.name
        )
        
        return response
        
    except HTTPException: 
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ===== BATCH ANALYSIS ENDPOINTS =====

@app.post("/api/batch/predict", response_model=BatchSentimentResponse)
async def batch_predict_sentiment(request: BatchReviewRequest):
    """Analyze multiple texts at once (max 100)"""
    try:
        if not models:
            raise HTTPException(
                status_code=503,
                detail="Models not loaded. Please check server logs."
            )
        
        if request.model not in models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Choose from: {list(models.keys())}"
            )
        
        selected_model = models[request.model]
        results = []
        summary = {"positive": 0, "neutral": 0, "negative": 0}
        
        for text in request.texts:
            if not text.strip():
                continue
                
            prediction = selected_model.predict(text)
            
            result = SentimentResponse(
                sentiment=prediction['sentiment'],
                probabilities=prediction['probabilities'],
                predicted_rating=prediction['predicted_rating'],
                text=text[: 100],
                model_used=selected_model.name
            )
            results.append(result)
            
            # Update summary
            sentiment_lower = prediction['sentiment'].lower()
            if sentiment_lower in summary:
                summary[sentiment_lower] += 1
        
        return BatchSentimentResponse(
            results=results,
            total=len(results),
            summary=summary,
            model_used=selected_model.name
        )
        
    except HTTPException:
        raise
    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/api/batch/csv")
async def batch_predict_csv(
    file: UploadFile = File(...),
    model: str = "svm"
):
    """
    Analyze reviews from CSV file. 
    CSV should have a column named 'text' or 'review' containing the reviews.
    Returns JSON with results and summary.
    """
    try:
        if not models: 
            raise HTTPException(
                status_code=503,
                detail="Models not loaded. Please check server logs."
            )
        
        if model not in models: 
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Choose from: {list(models.keys())}"
            )
        
        # Check file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="Please upload a CSV file."
            )
        
        # Read CSV file
        content = await file.read()
        decoded = content.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(decoded))
        
        # Find text column
        fieldnames = csv_reader.fieldnames or []
        text_column = None
        for col in ['text', 'review', 'comment', 'content', 'Text', 'Review', 'Comment', 'Content']:
            if col in fieldnames:
                text_column = col
                break
        
        if not text_column: 
            raise HTTPException(
                status_code=400,
                detail=f"CSV must have a column named 'text', 'review', 'comment', or 'content'.  Found columns: {fieldnames}"
            )
        
        selected_model = models[model]
        results = []
        summary = {"positive": 0, "neutral": 0, "negative": 0}
        
        row_count = 0
        max_rows = 100  # Limit to prevent overload
        
        for row in csv_reader:
            if row_count >= max_rows:
                break
                
            text = row.get(text_column, '').strip()
            if not text:
                continue
            
            prediction = selected_model.predict(text)
            
            result = {
                "text": text[: 100],
                "sentiment": prediction['sentiment'],
                "probabilities": prediction['probabilities'],
                "predicted_rating": prediction['predicted_rating'],
                "model_used": selected_model.name
            }
            results.append(result)
            
            # Update summary
            sentiment_lower = prediction['sentiment'].lower()
            if sentiment_lower in summary:
                summary[sentiment_lower] += 1
            
            row_count += 1
        
        return {
            "filename": file.filename,
            "results": results,
            "total": len(results),
            "summary":  summary,
            "model_used": selected_model.name,
            "limited":  row_count >= max_rows
        }
        
    except HTTPException:
        raise
    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"CSV processing error: {str(e)}")


@app.get("/api/stats")
async def get_model_stats():
    stats = {}
    
    if 'svm' in models:
        svm = models['svm']
        model_info = {
            "model_type": "SVM",
            "status": "loaded",
            "metadata": {}
        }
        
        if hasattr(svm, 'metadata'):
            if isinstance(svm.metadata, dict):
                model_info["metadata"] = svm.metadata
            else:
                model_info["metadata"] = {
                    "label_mapping": svm.label_mapping
                }
        
        stats['svm'] = model_info
    
    if 'logistic' in models:
        logistic = models['logistic']
        model_info = {
            "model_type": "Logistic",
            "status": "loaded",
            "metadata": {}
        }
        
        if hasattr(logistic, 'metadata'):
            if isinstance(logistic.metadata, dict):
                model_info["metadata"] = logistic.metadata
            else:
                model_info["metadata"] = {
                    "label_mapping":  logistic.label_mapping
                }
        
        stats['logistic'] = model_info
    
    if 'transformer' in models:
        transformer = models['transformer']
        stats['transformer'] = {
            "model_type": "XLM-RoBERTa",
            "status": "loaded",
            "device": str(transformer.device),
            "label_mapping": transformer.label_mapping,
            "max_length": transformer.max_length
        }

    if 'aza_transformer' in models:
        aza = models['aza_transformer']
        stats['aza_transformer'] = {
            "model_type": "XLM-RoBERTa (Aza)",
            "status": "loaded",
            "device": str(aza.device),
            "label_mapping": aza.label_mapping,
            "max_length": aza.max_length
        }
    
    return stats

if __name__ == "__main__": 
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )