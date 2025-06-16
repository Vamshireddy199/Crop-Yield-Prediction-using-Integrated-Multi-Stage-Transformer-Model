#data preprocessing and model training
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data = pd.read_csv("yield_df.csv")
if 'Unnamed: 0' in data.columns:
    data.drop(columns=['Unnamed: 0'], inplace=True)


for col in data.select_dtypes(include=['number']).columns:
    data[col] = data[col].fillna(data[col].mean())


categorical_cols = ['Area', 'Item']
numeric_cols = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'hg/ha_yield']
target_col = 'hg/ha_yield'


label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le


joblib.dump(label_encoders, "label_encoders.pkl")


scaler = MinMaxScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])


joblib.dump(scaler, "scaler.pkl")


train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]


class CropYieldDataset(Dataset):
    def __init__(self, df):
        self.X = df.drop(columns=[target_col]).values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


train_loader = DataLoader(CropYieldDataset(train_data), batch_size=64, shuffle=True)
test_loader = DataLoader(CropYieldDataset(test_data), batch_size=64, shuffle=False)


class SmallTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.2),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x).squeeze()


input_dim = len(categorical_cols) + len(numeric_cols) - 1
model = SmallTransformer(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


best_loss = float('inf')
for epoch in range(100):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_model.pth")


model.load_state_dict(torch.load("best_model.pth"))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch).cpu().numpy()
        y_true.extend(y_batch.numpy())
        y_pred.extend(outputs)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n--- Evaluation ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
