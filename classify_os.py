import json
import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import logging
from imblearn.over_sampling import SMOTENC
import joblib

# Определяем нужные категории (ключи)
REQUIRED_KEYS = ["ip", "user_agent", "tls", "tcpip", "os_prediction"]
INPUT_FILE = r"C:\Users\Жанабек\OneDrive\Документы\jason\fingerprints_with_os_checked.json"
CLEANED_FILE = "cleaned_sample_data.json"
MODEL_FILE = "model.pkl"
ENCODER_FILE = "label_encoder.pkl"

# Очистка JSON-файла
def clean_json(input_file, output_file, required_keys):
    """Загружает JSON (или JSONL), удаляет шум и сохраняет только нужные поля."""
    with open(input_file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)  # Попробавали здесь загрузить как обычный JSON
        except json.JSONDecodeError:
            f.seek(0)  # Если ошибка, читаем как JSONL (построчно)
            data = [json.loads(line) for line in f if line.strip()]

    # Фильтруем записи, оставляя только нужное
    cleaned_data = [{key: record.get(key) for key in required_keys} for record in data]

    # Сохраняем очищенные данные
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=4)

    print(f" Очищено JSON сохранен в {output_file}")

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, dict):
                data = list(data.values())
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]
    return data

def preprocess_data(data):
    features = []
    labels = []
    ja3_encoder = LabelEncoder()
    ja3_hashes = [record.get("tls", {}).get("ja3_hash", "") for record in data]
    ja3_encoder.fit(ja3_hashes)
    
    for record in data:
        user_agent = record.get("user_agent", "").lower()
        os_pred = record.get("os_prediction", {}).get("highest", "unknown")
        mismatch_score = int(("iphone" in user_agent and "windows" in os_pred) or 
                             ("windows" in user_agent and "mac os" in os_pred))
        
        feature_vector = {
    "ja3_hash_encoded": ja3_encoder.transform([record.get("tls", {}).get("ja3_hash", "")])[0],
    "cipher_count": len(record.get("tls", {}).get("ciphers", [])),
    "extension_count": len(record.get("tls", {}).get("extensions", [])),
    "ttl": round(record.get("tcpip", {}).get("ip", {}).get("ttl", 0) / 10) * 10,
    "mss": record.get("tcpip", {}).get("tcp", {}).get("mss", 0),
    "window_size": record.get("tcpip", {}).get("tcp", {}).get("window", 0),
    "mismatch_score": mismatch_score,
    "http2_settings_count": len(record.get("http2", {}).get("sent_frames", []))  # ✅ Добавляем этот признак
}

        features.append(feature_vector)
        labels.append(os_pred)
    return pd.DataFrame(features), labels


def train_model(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, shuffle=True)
    smote = SMOTENC(categorical_features=[0], random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    model = XGBClassifier(n_estimators=100, eval_metric='mlogloss', verbosity=1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    logging.info(f"Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
    joblib.dump(model, "model.pkl")
    joblib.dump(le, "label_encoder.pkl")

    
    return model, le

def classify_os(model, le, input_data):
    df, _ = preprocess_data(input_data)
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)
    results = []
    
    for i, record in enumerate(input_data):
        top_3 = np.argsort(probabilities[i])[-3:][::-1]
        top_3_os = [(le.inverse_transform([idx])[0], float(probabilities[i][idx])) for idx in top_3]
        result = {
            "ip": record.get("ip", "unknown"),
            "predicted_os": top_3_os
        }
        true_os = record.get("os_prediction", {}).get("highest", "unknown")
        predicted_os = top_3_os[0][0]
        confidence = top_3_os[0][1]
        if true_os != predicted_os:
            penalty = 2 if confidence > 0.9 else 1
            logging.warning(f"Ошибка предсказания: IP {record.get('ip', 'unknown')}, Истинная ОС: {true_os}, Предсказанная ОС: {predicted_os} (уверенность {confidence:.4f})")
        
       
        results.append(result)
    return results


def generate_noisy_data():
    return [
        {"ip": "192.168.1.100", "user_agent": "Mozilla/5.0 (iPhone; Windows NT 10.0; Win64; x64)",
         "tls": {"ja3_hash": "random_hash_123", "ciphers": []}, "tcpip": {"ip": {"ttl": 64}, "tcp": {"mss": 1460}}, "os_prediction": {"highest": "Windows"}},
        {"ip": "10.0.0.200", "user_agent": "Mozilla/5.0 (Linux; iPhone OS 15_1 like Mac OS X)",
         "tls": {"ja3_hash": "random_hash_456", "ciphers": []}, "tcpip": {"ip": {"ttl": 50}, "tcp": {"mss": 1200}}, "os_prediction": {"highest": "Linux"}}
    ]

   # Сохранение ошибок для дообучения
    with open("errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=4)
    
    return results

def retrain_with_errors():
    if os.path.exists("errors.json"):
        logging.info("Дообучение модели на ошибках...")
        errors = load_json("errors.json")
        if errors:
            data = load_json("sample_data.json") + errors
            X, y = preprocess_data(data)
            train_model(X, y)

def main():
    logging.basicConfig(level=logging.INFO)
    input_file = r"C:\Users\Жанабек\OneDrive\Документы\jason\cleaned_sample_data.json"
    if os.path.exists("model.pkl") and os.path.exists("label_encoder.pkl"):
        model = joblib.load("model.pkl")
        le = joblib.load("label_encoder.pkl")
        test_data = load_json("test_data.json")
        results = classify_os(model, le, test_data)
        print("\n📌 Результаты теста:")
        for res in results:
            print(res)
        retrain_with_errors()
    else:
        data = load_json(input_file)
        X, y = preprocess_data(data)
        model, le = train_model(X, y)
        results = classify_os(model, le, data)
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logging.info("Результаты сохранены в output.json")

if __name__ == "__main__":
    main()
