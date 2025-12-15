from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Optional
import joblib
import uvicorn
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI(
    title="Kazakh Sentiment Analyzer API",
    description="API for analyzing sentiment of Kazakh-language reviews",
    version="1.0.0"
)

# CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Review text in Kazakh")
    model: str = Field(default="svm", description="Model to use for prediction")

# Response model
class SentimentResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    sentiment: str
    probabilities: Dict[str, float]
    predicted_rating: Optional[float] = None
    text: str
    model_used:  str

# SVM Model class
class SVMModel:
    """Real SVM model loaded from joblib files"""
    
    def __init__(self, model_path:  str, vectorizer_path: str, metadata_path: str):
        self.name = "SVM"
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.metadata = joblib.load(metadata_path)
            print(f"✓ SVM model loaded successfully")
            print(f"✓ Vectorizer loaded successfully")
            print(f"✓ Metadata loaded successfully")
            
            # Get label mapping from metadata if available
            if hasattr(self.metadata, 'get'):
                self.label_mapping = self.metadata.get('label_mapping', {0: 'negative', 1: 'neutral', 2: 'positive'})
            else:
                self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
                
        except Exception as e:
            print(f"Error loading SVM model:  {e}")
            raise
    
    def predict(self, text: str) -> Dict:
        """
        Predict sentiment using the real SVM model
        
        Args:
            text: Review text in Kazakh
            
        Returns:
            Dictionary with sentiment, probabilities, and predicted rating
        """
        try:
            # Vectorize the text
            text_vectorized = self.vectorizer.transform([text])
            
            # Get prediction
            prediction = self.model.predict(text_vectorized)[0]
            
            # Get probability scores if available
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(text_vectorized)[0]
            elif hasattr(self.model, 'decision_function'):
                # For SVM, convert decision function to probabilities
                decision = self. model.decision_function(text_vectorized)[0]
                # Apply softmax to convert to probabilities
                exp_scores = np. exp(decision - np.max(decision))
                probas = exp_scores / exp_scores.sum()
            else:
                # Fallback: create one-hot probabilities
                probas = np.zeros(len(self.label_mapping))
                probas[prediction] = 1.0
            
            # Map prediction to sentiment label
            sentiment = self.label_mapping.get(prediction, 'neutral')
            
            # Create probability dictionary
            prob_dict = {}
            for idx, label in self.label_mapping.items():
                prob_dict[label] = float(probas[idx]) if idx < len(probas) else 0.0
            
            # Calculate predicted rating (1-5 scale)
            rating = (
                prob_dict. get('positive', 0) * 5.0 +
                prob_dict.get('neutral', 0) * 3.0 +
                prob_dict. get('negative', 0) * 1.0
            )
            
            return {
                'sentiment': sentiment,
                'probabilities': prob_dict,
                'predicted_rating': rating
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            raise


# Transformer Model class
class TransformerModel:
    """XLM-RoBERTa model for Kazakh sentiment analysis"""
    
    def __init__(self, model_path: str):
        self.name = "XLM-RoBERTa"
        self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.max_length = 128
        self.device = torch.device("cpu")
        
        try: 
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
        """
        Predict sentiment using the transformer model
        
        Args:
            text: Review text in Kazakh
            
        Returns:
            Dictionary with sentiment, probabilities, and predicted rating
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self. model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()
            
            # Map prediction to sentiment label
            sentiment = self.label_mapping.get(pred, 'neutral')
            
            # Create probability dictionary
            prob_dict = {
                'negative': float(probs[0][0].item()),
                'neutral':  float(probs[0][1].item()),
                'positive':  float(probs[0][2].item())
            }
            
            # Calculate predicted rating (1-5 scale)
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


# Initialize models
models = {}

# Load SVM model
try:
    models_dir = Path("models")
    svm_model = SVMModel(
        model_path=str(models_dir / "SVM.joblib"),
        vectorizer_path=str(models_dir / "SVM_vector.joblib"),
        metadata_path=str(models_dir / "SVM_metadata.joblib")
    )
    models['svm'] = svm_model
    print(f"✓ SVM model initialized successfully")
except Exception as e: 
    print(f"✗ Failed to initialize SVM model: {e}")

# Load Transformer model
try: 
    models_dir = Path("models")
    transformer_model = TransformerModel(
        model_path=str(models_dir / "transformer_model")
    )
    models['transformer'] = transformer_model
    print(f"✓ Transformer model initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize Transformer model: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Kazakh Sentiment Analyzer API",
        "version":  "1.0.0",
        "available_models": list(models.keys()),
        "endpoints": {
            "predict": "/api/predict (POST)",
            "models": "/api/models (GET)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }

@app. get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if models else "degraded", 
        "models_loaded": list(models.keys()),
        "total_models": len(models)
    }

@app.get("/api/models")
async def get_available_models():
    """Get list of available models"""
    available_models = []
    
    if 'svm' in models: 
        available_models.append({
            "id": "svm",
            "name": "Support Vector Machine (SVM)",
            "description":  "SVM classifier with TF-IDF vectorization for Kazakh text",
            "type": "sklearn",
            "status": "loaded"
        })
    
    if 'transformer' in models:
        available_models.append({
            "id":  "transformer",
            "name":  "XLM-RoBERTa",
            "description": "Fine-tuned XLM-RoBERTa model for Kazakh sentiment analysis",
            "type": "deep_learning",
            "status":  "loaded"
        })
    
    return {"models": available_models}

@app.post("/api/predict", response_model=SentimentResponse)
async def predict_sentiment(request: ReviewRequest):
    """
    Predict sentiment for a given review using the specified model. 
    
    Args:
        request: ReviewRequest containing text and model choice
        
    Returns:
        SentimentResponse with sentiment, probabilities, and rating
    """
    try: 
        # Check if models are loaded
        if not models:
            raise HTTPException(
                status_code=503,
                detail="Models not loaded. Please check server logs."
            )
        
        # Validate model selection
        if request. model not in models:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model.  Choose from:  {list(models.keys())}"
            )
        
        # Get the selected model
        selected_model = models[request.model]
        
        # Get prediction from model
        prediction = selected_model. predict(request.text)
        
        # Create response
        response = SentimentResponse(
            sentiment=prediction['sentiment'],
            probabilities=prediction['probabilities'],
            predicted_rating=prediction['predicted_rating'],
            text=request.text[: 100],  # Truncate for response
            model_used=selected_model.name
        )
        
        return response
        
    except HTTPException: 
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/stats")
async def get_model_stats():
    """
    Get model statistics and metadata
    """
    stats = {}
    
    if 'svm' in models:
        svm = models['svm']
        model_info = {
            "model_type": "SVM",
            "status": "loaded",
            "metadata": {}
        }
        
        # Extract metadata if available
        if hasattr(svm, 'metadata'):
            if isinstance(svm.metadata, dict):
                model_info["metadata"] = svm.metadata
            else:
                model_info["metadata"] = {
                    "label_mapping": svm.label_mapping
                }
        
        stats['svm'] = model_info
    
    if 'transformer' in models:
        transformer = models['transformer']
        stats['transformer'] = {
            "model_type": "XLM-RoBERTa",
            "status": "loaded",
            "device": str(transformer.device),
            "label_mapping": transformer.label_mapping,
            "max_length": transformer.max_length
        }
    
    return stats

if __name__ == "__main__": 
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes
    )