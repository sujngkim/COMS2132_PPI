from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import data_processing_2 as dt
import pickle
import sklearn

class SiamesePPI(nn.Module):
    def __init__(self, vocab_size=21, embed_dim=128, hidden_dim=128):
        super(SiamesePPI, self).__init__()
        
        # 1. Embedding Layer: Converts amino acid integers to dense vectors
        # padding_idx=0 ensures the model ignores the zeros we added
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 2. Shared Encoder (The "Arm"): 
        # This CNN will learn to detect local patterns in the sequences
        self.encoder = nn.Sequential(
            # Input: (Batch, Embed_Dim, Length)
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            # Global Average Pooling: Converts variable/fixed lengths to 1 vector
            nn.AdaptiveAvgPool1d(1) 
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, 64),
            nn.ReLU(),
            nn.Dropout(0.3), # Prevents overfitting
            nn.Linear(64, 1),
            nn.Sigmoid() # Outputs a probability between 0 and 1
        )

    def forward_once(self, x):
        # This function processes one protein through the encoder
        x = self.embedding(x)        # (Batch, Length, Embed_Dim)
        x = x.transpose(1, 2)        # (Batch, Embed_Dim, Length) for Conv1d
        x = self.encoder(x)          # (Batch, Hidden_Dim, 1)
        x = x.view(x.size(0), -1)    # Flatten to (Batch, Hidden_Dim)
        return x

    def forward(self, protein_a, protein_b, interaction_score):
        feat_a = self.forward_once(protein_a)
        feat_b = self.forward_once(protein_b)
        combined = torch.cat((feat_a, feat_b), dim=1)
        if interaction_score.dim() == 1:
            interaction_score = interaction_score.unsqueeze(1)
        combined = torch.cat((combined, interaction_score), dim=1)
        return self.classifier(combined)
def open_dictionaries(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def main(): 
    data = open_dictionaries('processed_data.pkl')
    ppi_data_pos = dt.PPIDataset(data['pos_train'], data['prots'], 1.0)
    ppi_data_neg = dt.PPIDataset(data['neg_train'], data['prots'], 0.0)
    total_ppi_data = ConcatDataset([ppi_data_pos, ppi_data_neg]) # final dataset
    train_loader = DataLoader(total_ppi_data, batch_size=32, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiamesePPI().to(device)
    criterion = nn.BCELoss(reduction='none') # 'none' allows us to apply weights manually
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    num_epochs = 10
    for epoch in range(num_epochs):
        print("batch started")
        model.train()
        running_loss = 0.0
        n_batches = 0
        for seq_a, seq_b, labels, weights in train_loader:
            seq_a, seq_b = seq_a.to(device), seq_b.to(device)
            labels, weights = labels.to(device), weights.to(device)
            w_loss = weights.float() / weights.float().mean().clamp(min=1e-6)

            w_norm = (weights / (weights.max().detach().clamp(min=1e-6))).to(dtype=torch.float32)
            outputs = model(seq_a, seq_b, w_norm).squeeze()
            loss = criterion(outputs, labels)
            weighted_loss = (loss * w_loss).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += weighted_loss.item()
            n_batches += 1
        epoch_loss = running_loss / max(n_batches, 1)
        scheduler.step(epoch_loss)
        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}", flush=True)
    return model

def testing(model):
    data = open_dictionaries('processed_data.pkl')
    ppi_data_pos = dt.PPIDataset(data['pos_test'], data['prots'], 1.0)
    ppi_data_neg = dt.PPIDataset(data['neg_test'], data['prots'], 0.0)
    total_ppi_data = ConcatDataset([ppi_data_pos, ppi_data_neg]) # final dataset
    test_loader = DataLoader(total_ppi_data, batch_size=32, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # 1. Switch model to eval mode (disables Dropout/Batchnorm)

    all_preds = []
    all_labels = []

    print("Starting Final Evaluation...")

    # 2. Use torch.no_grad() to save memory and speed up the math
    with torch.no_grad():
        for seq_a, seq_b, labels, weights in test_loader:
            seq_a, seq_b = seq_a.to(device), seq_b.to(device)
            weights = weights.to(device)
            w_norm = weights / (weights.max().detach().clamp(min=1e-6))
            outputs = model(seq_a, seq_b, w_norm).squeeze()

            
            # Convert probabilities to binary 0/1
            preds = (outputs > 0.5).int().cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # 3. Final Metrics
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n--- FINAL MODEL PERFORMANCE ---")
    print(classification_report(all_labels, all_preds, target_names=['Non-Interaction', 'Interaction']))

def saving_model(model):
    model_path = "siamese_ppi_model.pth"
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":  
    trained_model = main()
    testing(trained_model)
    saving_model(trained_model)
