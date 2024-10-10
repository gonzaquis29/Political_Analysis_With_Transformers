import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

FILE_NAME = "poli_bias_optimized"
PRETRAINED_MODEL = "dccuchile/bert-base-spanish-wwm-uncased"

# Load and preprocess data
data_corpus_pre = pd.read_csv('../../../data/corpus/political_bias_preprocessed.csv')
data_corpus_pre['text_processed'] = data_corpus_pre['text_processed'].astype(str)
train_df, eval_df = train_test_split(data_corpus_pre, test_size=0.2, random_state=42)

# Model definition
class LibertyPredictor(nn.Module):
    def __init__(self, pretrained_model_name):
        super(LibertyPredictor, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc_personal = nn.Linear(self.bert.config.hidden_size, 3)
        self.fc_economic = nn.Linear(self.bert.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        x = self.dropout(pooled_output)
        personal_liberty = self.fc_personal(x)
        economic_liberty = self.fc_economic(x)
        return personal_liberty, economic_liberty

# Dataset definition
class LibertyDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text_processed
        self.targets = self._process_targets(dataframe[['libertad_personal_score', 'libertad_economica_score']].values)
        self.max_len = max_len

    def _process_targets(self, scores):
        def score_to_class(score):
            if score < -0.33:
                return 0
            elif score < 0.33:
                return 1
            else:
                return 2
        return [[score_to_class(personal), score_to_class(economic)] for personal, economic in scores]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }

# Hyperparameters
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-04
L1_LAMBDA = 1e-5
L2_LAMBDA = 1e-4
PATIENCE = 5

# Tokenizer and model initialization
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LibertyPredictor(PRETRAINED_MODEL).to(device)

# Freeze BERT layers except the last two
for name, param in model.bert.named_parameters():
    if 'layer.10' in name or 'layer.11' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Dataset and DataLoader creation
train_dataset = LibertyDataset(train_df.reset_index(drop=True), tokenizer, MAX_LEN)
eval_dataset = LibertyDataset(eval_df.reset_index(drop=True), tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2)
eval_loader = DataLoader(eval_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=1)

# Optimizer and loss function
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training function with L1 and L2 regularization
def train(model, train_loader, optimizer, criterion, device, l1_lambda, l2_lambda):
    model.train()
    total_loss = 0
    all_personal_preds, all_economic_preds = [], []
    all_personal_targets, all_economic_targets = [], []

    for _, data in enumerate(train_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        personal_liberty, economic_liberty = model(ids, mask)
        loss = criterion(personal_liberty, targets[:, 0]) + criterion(economic_liberty, targets[:, 1])

        # L1 regularization
        l1_reg = torch.tensor(0., requires_grad=True)
        for param in model.parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        loss = loss + l1_lambda * l1_reg

        # L2 regularization
        l2_reg = torch.tensor(0., requires_grad=True)
        for param in model.parameters():
            l2_reg = l2_reg + torch.norm(param, 2)
        loss = loss + l2_lambda * l2_reg

        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_personal_preds.extend(torch.argmax(personal_liberty, dim=1).cpu().numpy())
        all_economic_preds.extend(torch.argmax(economic_liberty, dim=1).cpu().numpy())
        all_personal_targets.extend(targets[:, 0].cpu().numpy())
        all_economic_targets.extend(targets[:, 1].cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    personal_f1 = f1_score(all_personal_targets, all_personal_preds, average='weighted')
    economic_f1 = f1_score(all_economic_targets, all_economic_preds, average='weighted')

    return avg_loss, personal_f1, economic_f1

# Evaluation function
def evaluate(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_personal_preds, all_economic_preds = [], []
    all_personal_targets, all_economic_targets = [], []

    with torch.no_grad():
        for _, data in enumerate(eval_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            personal_liberty, economic_liberty = model(ids, mask)
            loss = criterion(personal_liberty, targets[:, 0]) + criterion(economic_liberty, targets[:, 1])
            total_loss += loss.item()

            all_personal_preds.extend(torch.argmax(personal_liberty, dim=1).cpu().numpy())
            all_economic_preds.extend(torch.argmax(economic_liberty, dim=1).cpu().numpy())
            all_personal_targets.extend(targets[:, 0].cpu().numpy())
            all_economic_targets.extend(targets[:, 1].cpu().numpy())

    avg_loss = total_loss / len(eval_loader)
    personal_f1 = f1_score(all_personal_targets, all_personal_preds, average='weighted')
    economic_f1 = f1_score(all_economic_targets, all_economic_preds, average='weighted')

    return avg_loss, personal_f1, economic_f1

# Training loop with early stopping
best_eval_loss = float('inf')
best_epoch = 0
best_metrics = {}
patience_counter = 0

with open(f'training_logs_{FILE_NAME}.txt', 'a') as log_file:
    for epoch in range(EPOCHS):
        train_loss, train_personal_f1, train_economic_f1 = train(model, train_loader, optimizer, criterion, device, L1_LAMBDA, L2_LAMBDA)
        eval_loss, eval_personal_f1, eval_economic_f1 = evaluate(model, eval_loader, criterion, device)
        
        log_message = (f'Epoch {epoch+1}/{EPOCHS}, '
                       f'Train Loss: {train_loss:.4f}, '
                       f'Train Personal F1: {train_personal_f1:.4f}, '
                       f'Train Economic F1: {train_economic_f1:.4f}, '
                       f'Validation Loss: {eval_loss:.4f}, '
                       f'Validation Personal F1: {eval_personal_f1:.4f}, '
                       f'Validation Economic F1: {eval_economic_f1:.4f}')
        
        log_file.write(log_message + '\n')
        print(log_message)

        # Early stopping
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epoch = epoch + 1
            best_metrics = {
                'train_loss': train_loss,
                'train_personal_f1': train_personal_f1,
                'train_economic_f1': train_economic_f1,
                'eval_loss': eval_loss,
                'eval_personal_f1': eval_personal_f1,
                'eval_economic_f1': eval_economic_f1
            }
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), f'{FILE_NAME}_best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Print and log the best model summary
    best_model_summary = (f"\nBest Model Summary:\n"
                          f"Best Epoch: {best_epoch}\n"
                          f"Best Validation Loss: {best_metrics['eval_loss']:.4f}\n"
                          f"Best Train Loss: {best_metrics['train_loss']:.4f}\n"
                          f"Best Train Personal F1: {best_metrics['train_personal_f1']:.4f}\n"
                          f"Best Train Economic F1: {best_metrics['train_economic_f1']:.4f}\n"
                          f"Best Validation Personal F1: {best_metrics['eval_personal_f1']:.4f}\n"
                          f"Best Validation Economic F1: {best_metrics['eval_economic_f1']:.4f}\n")
    
    print(best_model_summary)
    log_file.write(best_model_summary)

print("Training completed.")

# Load the best model
model.load_state_dict(torch.load(f'{FILE_NAME}_best_model.pth'))

# Prediction function
def predict(model, text, tokenizer, max_len, device):
    model.eval()
    encoded_text = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_token_type_ids=True,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    with torch.no_grad():
        personal_liberty, economic_liberty = model(input_ids, attention_mask)

    personal_score = torch.argmax(personal_liberty, dim=1).item() - 1
    economic_score = torch.argmax(economic_liberty, dim=1).item() - 1
    return personal_score, economic_score

# Example usage
texto_ejemplo = "conservador"
libertad_personal, libertad_economica = predict(model, texto_ejemplo, tokenizer, MAX_LEN, device)
print(f"Libertad personal: {libertad_personal}")
print(f"Libertad económica: {libertad_economica}")