#model evaluation and visualization
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerIMSTModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=2048):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
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


test_data_path = '/content/drive/MyDrive/IMST2/test_dataset.csv'
model_path = '/content/drive/MyDrive/IMST2/best_model.pth'
scaler_path = '/content/drive/MyDrive/IMST2/scaler (1).pkl'
encoders_path = '/content/drive/MyDrive/IMST2/label_encoders (1).pkl'

target_col = 'hg/ha_yield'
categorical_cols = ['Area', 'Item']
numerical_cols = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
all_features = categorical_cols + numerical_cols

df = pd.read_csv(test_data_path)
label_encoders = joblib.load(encoders_path)
scaler = joblib.load(scaler_path)

for col in categorical_cols:
    df[col] = label_encoders[col].transform(df[col])

scaled = scaler.transform(df[numerical_cols + [target_col]])
df[numerical_cols + [target_col]] = scaled

X_test = df[all_features].values.astype(np.float32)
y_test = df[target_col].values.astype(np.float32)

X_test_tensor = torch.tensor(X_test).to(device)
y_test_tensor = torch.tensor(y_test).to(device)

model = TransformerIMSTModel(input_dim=X_test.shape[1]).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


with torch.no_grad():
    y_pred_tensor = model(X_test_tensor).squeeze()
    y_pred = y_pred_tensor.cpu().numpy()
    y_true = y_test_tensor.cpu().numpy()

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
median_yield = np.median(y_true)
f1 = f1_score((y_true > median_yield).astype(int), (y_pred > median_yield).astype(int), zero_division=1)
accuracy = np.mean(np.abs(y_pred - y_true) / y_true <= 0.8)

print("\n--- Transformer Model Evaluation on Test Dataset ---")
print(f"MAE                : {mae:.4f}")
print(f"MSE                : {mse:.4f}")
print(f"RMSE               : {rmse:.4f}")
print(f"R² Score           : {r2:.4f}")
print(f"F1 Score           : {f1:.4f}")
print(f"Accuracy           : {accuracy * 100:.2f}%")



true_binary = (y_true > median_yield).astype(int)
pred_binary = (y_pred > median_yield).astype(int)

cm = confusion_matrix(true_binary, pred_binary)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Yield", "High Yield"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

thresholds = np.linspace(0.01, 1.0, 20)
accuracies = [np.mean(np.abs(y_pred - y_true) <= t) for t in thresholds]

plt.figure(figsize=(8, 5))
plt.plot(thresholds, accuracies, marker='o', linestyle='--', color='green')
plt.xlabel("Absolute Error Threshold")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Error Threshold")
plt.grid(True)
plt.tight_layout()
plt.show()


residuals = y_true - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=30, color="orange")
plt.title("Residual Distribution (True - Predicted)")
plt.xlabel("Residual")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 6))
plt.scatter(y_true, y_pred, alpha=0.6, color="blue")
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
plt.xlabel("True Yield")
plt.ylabel("Predicted Yield")
plt.title("True vs Predicted Yield")
plt.grid(True)
plt.tight_layout()
plt.show()
