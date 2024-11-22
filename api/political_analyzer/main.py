from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Define request/response models
class TextRequest(BaseModel):
    text: str 

class LibertyResponse(BaseModel):
    personal_liberty: int
    economic_liberty: int
    personal_liberty_probabilities: list
    economic_liberty_probabilities: list

# Model definition (matches training code)
class LibertyPredictor(nn.Module):
    def __init__(self, pretrained_model_name):
        super(LibertyPredictor, self).__init__()
        self.distilbert = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc_personal = nn.Linear(self.distilbert.config.hidden_size, 3)
        self.fc_economic = nn.Linear(self.distilbert.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        x = self.dropout(pooled_output)
        personal_liberty = self.fc_personal(x)
        economic_liberty = self.fc_economic(x)
        return personal_liberty, economic_liberty

# Initialize FastAPI app
app = FastAPI(
    title="Liberty Prediction API",
    description="API for predicting personal and economic liberty scores from Spanish text using DistilBERT",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None
MAX_LEN = 128
PRETRAINED_MODEL = "dccuchile/distilbert-base-spanish-uncased"
MODEL_PATH = "manifesto_bal_distilbert_optimized_best_model.pth"

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, device
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, clean_up_tokenization_spaces=True)
    
        # Initialize and load model
        model = LibertyPredictor(PRETRAINED_MODEL).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        
        print(f"Model and tokenizer loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

def predict_text(text: str):
    # Tokenize input text
    encoded_text = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_token_type_ids=False,
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    with torch.no_grad():
        personal_liberty, economic_liberty = model(input_ids, attention_mask)
        
        # Get probabilities using softmax
        personal_probs = torch.nn.functional.softmax(personal_liberty, dim=1)
        economic_probs = torch.nn.functional.softmax(economic_liberty, dim=1)
        
        # Get predictions (converting from 0,1,2 to -1,0,1)
        personal_pred = torch.argmax(personal_liberty, dim=1).item() - 1
        economic_pred = torch.argmax(economic_liberty, dim=1).item() - 1
        
        # Convert probability tensors to lists
        personal_probs = personal_probs.cpu().numpy().tolist()[0]
        economic_probs = economic_probs.cpu().numpy().tolist()[0]

    return {
        "personal_liberty": personal_pred,
        "economic_liberty": economic_pred,
        "personal_liberty_probabilities": personal_probs,
        "economic_liberty_probabilities": economic_probs
    }

@app.post("/predict", response_model=LibertyResponse)
async def predict(request: TextRequest):
    try:
        if not model or not tokenizer:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        prediction = predict_text(request.text)
        return prediction
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    if model and tokenizer:
        return {"status": "healthy", "device": str(device)}
    return {"status": "model not loaded"}

@app.get("/hello")
async def hello():
    return {"message": "Hello world"}
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)