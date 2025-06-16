#user input and prediction
import torch
import joblib
import numpy as np
import pandas as pd
from torch import nn


label_encoders = joblib.load("/content/drive/MyDrive/IMST2/label_encoders (1).pkl")
scaler = joblib.load("/content/drive/MyDrive/IMST2/scaler (1).pkl")

area_classes = label_encoders["Area"].classes_
item_classes = label_encoders["Item"].classes_


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


input_dim = 6
model = SmallTransformer(input_dim=input_dim)
model.load_state_dict(torch.load("/content/drive/MyDrive/IMST2/best_model.pth", map_location='cpu'))
model.eval()


def case_insensitive_match(user_input, classes):
    matches = [cls for cls in classes if user_input.lower() in cls.lower()]
    if not matches:
        raise ValueError(f"Invalid input '{user_input}'. Options: {classes}")
    return matches[0]


print(f"Available Areas: {list(area_classes)}")
print(f"Available Items: {list(item_classes)}")

user_area = input("Enter Area: ").strip()
user_item = input("Enter Item: ").strip()

matched_area = case_insensitive_match(user_area, area_classes)
matched_item = case_insensitive_match(user_item, item_classes)


area_encoded = label_encoders["Area"].transform([matched_area])[0]
item_encoded = label_encoders["Item"].transform([matched_item])[0]


year = float(input("Enter Year: "))
rainfall = float(input("Enter Average Rainfall (mm): "))
pesticides = float(input("Enter Pesticides Used (tonnes): "))
temp = float(input("Enter Average Temperature (°C): "))


columns = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'hg/ha_yield']
input_df = pd.DataFrame([[year, rainfall, pesticides, temp, 0.0]], columns=columns)
scaled = scaler.transform(input_df)[:, :-1]


features = np.concatenate([[area_encoded, item_encoded], scaled.flatten()]).astype(np.float32)
input_tensor = torch.tensor(features).unsqueeze(0)


with torch.no_grad():
    prediction_scaled = model(input_tensor).item()


input_with_prediction = np.append(scaled.flatten(), prediction_scaled).reshape(1, -1)
real_yield = scaler.inverse_transform(input_with_prediction)[0, -1]

print(f"\n Predicted Crop Yield (hg/ha): {real_yield:.2f}")
