# preprocess.py

import pandas as pd
import numpy as np
import os

file_path = "Data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
cleaned_folder = "Data/Cleaned"
os.makedirs(cleaned_folder, exist_ok=True)
cleaned_csv_path = os.path.join(cleaned_folder, "friday_cleaned.csv")

# CSV dosyasını oku
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Label dönüştür
if 'Label' in df.columns:
    df['Label'] = df['Label'].map({'BENIGN': 0, 'DDoS': 1})
    df = df[df['Label'].notna()]  # Harici etiketleri dışla
else:
    raise ValueError("'Label' sütunu bulunamadı.")

# Özellik seçimi
selected_features = [
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Mean", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Fwd Packet Length Std",
    "Flow Duration", "Flow IAT Mean", "Flow IAT Std",
    "Fwd IAT Mean", "Bwd IAT Mean",
    "Flow Bytes/s", "Flow Packets/s",
    "SYN Flag Count", "ACK Flag Count",
    "Packet Length Mean", "Average Packet Size",
    "Down/Up Ratio", "Label"
]

missing = [col for col in selected_features if col not in df.columns]
if missing:
    raise ValueError(f"Eksik sütunlar: {missing}")

df = df[selected_features]
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

# Temizlenmiş veriyi kaydet
df.to_csv(cleaned_csv_path, index=False)
print(f"✅ Temizlenmiş veri kaydedildi: {cleaned_csv_path}")

# Özellik isimlerini kaydet
feature_names_path = os.path.join("model", "features.txt")
os.makedirs("model", exist_ok=True)
with open(feature_names_path, "w") as f:
    for col in df.columns:
        if col != 'Label':
            f.write(col + "\n")
print(f"📝 Özellik isimleri kaydedildi: {feature_names_path}")
