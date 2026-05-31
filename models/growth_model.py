import os
import json
from typing import List, Tuple

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'data', 'plant_dataset.csv'))
MODEL_PATH = os.path.join(BASE_DIR, 'saved_model.keras')
WEIGHTS_PATH = os.path.join(BASE_DIR, 'model_weights.json')

CONDITION_MAP = {
    'Cukup': 0, 'Baik': 1, 'Sangat Baik': 2,
    'Sehat': 2, 'Cukup Baik': 1, 'Perlu Perhatian': 0
}

_MODEL_CACHE = None
_FEATURE_COLUMNS_CACHE = None
_WEIGHTS_CACHE = None


def load_dataset(csv_path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding='utf-8')
    expected_columns = {
        'plant',
        'age_days',
        'height_cm',
        'leaf_count',
        'humidity_pct',
        'temperature_c',
        'condition',
        'estimated_harvest_days',
    }
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in dataset: {sorted(missing)}')
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = df.copy()
    df['condition'] = df['condition'].map(CONDITION_MAP).fillna(0).astype(np.float32)
    df = pd.get_dummies(df, columns=['plant'], prefix='plant')

    base_features = [
        'age_days',
        'height_cm',
        'leaf_count',
        'humidity_pct',
        'temperature_c',
        'condition',
    ]
    plant_features = [col for col in df.columns if col.startswith('plant_')]
    feature_columns = base_features + plant_features

    X = df[feature_columns].astype(np.float32).to_numpy()
    y = df['estimated_harvest_days'].astype(np.float32).to_numpy()
    return X, y, feature_columns


def build_model(input_dim: int):
    import tensorflow as tf
    normalizer = tf.keras.layers.Normalization(axis=-1)
    model = tf.keras.Sequential(
        [
            normalizer,
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1),
        ]
    )
    return model, normalizer


def train_model(epochs: int = 120, batch_size: int = 8, validation_split: float = 0.2):
    import tensorflow as tf
    df = load_dataset()
    X, y, _ = preprocess_data(df)
    model, normalizer = build_model(X.shape[1])
    
    X_tensor = tf.convert_to_tensor(X)
    y_tensor = tf.convert_to_tensor(y)
    normalizer.adapt(X_tensor)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='mse',
        metrics=['mae'],
    )

    model.fit(X_tensor, y_tensor, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=2)
    save_model(model)
    return model


def save_model(model, model_path: str = MODEL_PATH) -> None:
    import tensorflow as tf
    global _MODEL_CACHE
    dirname = os.path.dirname(model_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    model.save(model_path)
    _MODEL_CACHE = model


def load_model(model_path: str = MODEL_PATH):
    import tensorflow as tf
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file not found: {model_path}')
        _MODEL_CACHE = tf.keras.models.load_model(model_path)
    return _MODEL_CACHE


def _get_weights():
    global _WEIGHTS_CACHE
    if _WEIGHTS_CACHE is None:
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(f"Model weights file not found: {WEIGHTS_PATH}")
        with open(WEIGHTS_PATH, 'r') as f:
            _WEIGHTS_CACHE = json.load(f)
    return _WEIGHTS_CACHE


def get_feature_columns() -> List[str]:
    global _FEATURE_COLUMNS_CACHE
    if _FEATURE_COLUMNS_CACHE is None:
        _, _, feature_columns = preprocess_data(load_dataset())
        _FEATURE_COLUMNS_CACHE = feature_columns
    return _FEATURE_COLUMNS_CACHE


def prepare_input_vector(
    age_days: float,
    height_cm: float,
    leaf_count: int,
    humidity_pct: float,
    temperature_c: float,
    condition: str,
    plant: str,
) -> np.ndarray:
    feature_columns = get_feature_columns()
    conditions = CONDITION_MAP.get(condition, 0)
    row = {
        'age_days': age_days,
        'height_cm': height_cm,
        'leaf_count': leaf_count,
        'humidity_pct': humidity_pct,
        'temperature_c': temperature_c,
        'condition': conditions,
    }

    for col in feature_columns:
        if col.startswith('plant_'):
            row[col] = 1.0 if col == f'plant_{plant}' else 0.0

    vector = np.array([[row[col] for col in feature_columns]], dtype=np.float32)
    return vector


def predict_harvest_days(
    age_days: float,
    height_cm: float,
    leaf_count: int,
    humidity_pct: float,
    temperature_c: float,
    condition: str,
    plant: str,
) -> float:
    features = prepare_input_vector(
        age_days,
        height_cm,
        leaf_count,
        humidity_pct,
        temperature_c,
        condition,
        plant,
    )
    
    # Load model weights from JSON
    w_data = _get_weights()
    norm_mean = np.array(w_data['normalization']['mean'], dtype=np.float32)
    norm_var = np.array(w_data['normalization']['variance'], dtype=np.float32)
    
    dense_keys = sorted([k for k in w_data.keys() if k.startswith('dense')])
    
    # 1. Normalization
    normalized = (features - norm_mean) / np.sqrt(norm_var + 1e-3)
    
    # 2. Dense layer 1 (ReLU)
    w1 = np.array(w_data[dense_keys[0]]['weights'], dtype=np.float32)
    b1 = np.array(w_data[dense_keys[0]]['bias'], dtype=np.float32)
    h1 = np.maximum(np.dot(normalized, w1) + b1, 0.0)
    
    # 3. Dense layer 2 (ReLU)
    w2 = np.array(w_data[dense_keys[1]]['weights'], dtype=np.float32)
    b2 = np.array(w_data[dense_keys[1]]['bias'], dtype=np.float32)
    h2 = np.maximum(np.dot(h1, w2) + b2, 0.0)
    
    # 4. Dense layer 3 / Output (Linear)
    w3 = np.array(w_data[dense_keys[2]]['weights'], dtype=np.float32)
    b3 = np.array(w_data[dense_keys[2]]['bias'], dtype=np.float32)
    prediction = float((np.dot(h2, w3) + b3).item())
    
    return float(max(prediction, 0.0))


def predict_growth_stage(height_cm: float, leaf_count: int, health_score: float) -> str:
    # Pure NumPy implementation of Sigmoid activation over linear regression score
    score = height_cm * 0.02 + leaf_count * 0.04 + health_score * 0.55 + 0.18
    probability = 1.0 / (1.0 + np.exp(-score))

    if probability < 0.35:
        return 'Tahap awal (persemaian & akar berkembang)'
    if probability < 0.65:
        return 'Tahap pertumbuhan (batang dan daun aktif)'
    return 'Tahap matang (buah mulai berkembang atau matang)'


def evaluate_plant_condition(age_days: float, temperature_c: float, humidity_pct: float) -> str:
    """
    Mengevaluasi kondisi tanaman berdasarkan umur, suhu, dan kelembapan.
    
    Args:
        age_days: Umur tanaman dalam hari
        temperature_c: Suhu dalam Celsius
        humidity_pct: Kelembapan dalam persen (0-100)
    
    Returns:
        Salah satu dari: 'Sehat', 'Cukup Baik', atau 'Perlu Perhatian'
    """
    # Parameter optimal untuk tanaman (cabai, tomat, terong)
    optimal_temp_min, optimal_temp_max = 25.0, 30.0
    optimal_humidity_min, optimal_humidity_max = 60.0, 85.0
    
    # Hitung deviasi dari nilai optimal
    temp_deviation = 0
    if temperature_c < optimal_temp_min:
        temp_deviation = optimal_temp_min - temperature_c
    elif temperature_c > optimal_temp_max:
        temp_deviation = temperature_c - optimal_temp_max
    
    humidity_deviation = 0
    if humidity_pct < optimal_humidity_min:
        humidity_deviation = optimal_humidity_min - humidity_pct
    elif humidity_pct > optimal_humidity_max:
        humidity_deviation = humidity_pct - optimal_humidity_max
    
    # Total skor deviasi (semakin rendah semakin baik)
    total_deviation = temp_deviation + humidity_deviation * 0.5
    
    # Penyesuaian berdasarkan umur tanaman
    age_penalty = 0
    if age_days < 7:
        age_penalty = 5  # Tahap awal, lebih sensitif
    elif age_days > 90:
        age_penalty = 2  # Tahap akhir, stabil
    
    # Total skor akhir
    total_score = total_deviation + age_penalty
    
    # Klasifikasi kondisi
    if total_score < 3:
        return 'Sehat'
    elif total_score < 7:
        return 'Cukup Baik'
    else:
        return 'Perlu Perhatian'


def generate_care_recommendations(
    condition: str,
    temperature_c: float,
    humidity_pct: float,
    age_days: float,
) -> dict:
    """
    Generate rekomendasi perawatan berdasarkan kondisi tanaman dan lingkungan.
    
    Args:
        condition: Status kondisi ('Sehat', 'Cukup Baik', 'Perlu Perhatian')
        temperature_c: Suhu dalam Celsius
        humidity_pct: Kelembapan dalam persen
        age_days: Umur tanaman dalam hari
    
    Returns:
        Dictionary berisi rekomendasi untuk: penyiraman, pupuk, dan tindakan khusus
    """
    recommendations = {
        'watering': [],
        'fertilizer': [],
        'special_actions': [],
    }
    
    # Rekomendasi penyiraman berdasarkan kelembapan
    if humidity_pct < 60:
        recommendations['watering'].append('Penyiraman perlu ditingkatkan - kelembapan terlalu rendah.')
        recommendations['watering'].append('Siram setiap hari atau 2 kali sehari di musim kering.')
    elif humidity_pct > 85:
        recommendations['watering'].append('Kurangi frekuensi penyiraman - kelembapan terlalu tinggi.')
        recommendations['watering'].append('Pastikan drainase baik untuk mencegah pembusukan akar.')
    else:
        recommendations['watering'].append('Kelembapan optimal - lanjutkan pola penyiraman reguler.')
    
    # Rekomendasi pemupukan berdasarkan usia dan kondisi
    if age_days < 14:
        recommendations['fertilizer'].append('Gunakan pupuk NPK seimbang untuk tahap awal pertumbuhan.')
        recommendations['fertilizer'].append('Berikan pupuk cair setiap 3-4 hari sekali.')
    elif age_days < 45:
        recommendations['fertilizer'].append('Tingkatkan nitrogen untuk pertumbuhan daun yang optimal.')
        recommendations['fertilizer'].append('Pupuk setiap minggu dengan konsentrasi moderat.')
    else:
        recommendations['fertilizer'].append('Tingkatkan potassium untuk pembentukan buah yang lebih baik.')
        recommendations['fertilizer'].append('Pupuk setiap 10-14 hari untuk hasil panen maksimal.')
    
    # Rekomendasi suhu
    if temperature_c < 20:
        recommendations['special_actions'].append('⚠️ Suhu terlalu rendah - pertumbuhan akan melambat.')
        recommendations['special_actions'].append('Pertimbangkan menggunakan mulsa atau pelindung tanaman.')
    elif temperature_c < 25:
        recommendations['special_actions'].append('Suhu agak rendah - tingkatkan paparan sinar matahari.')
    elif temperature_c > 32:
        recommendations['special_actions'].append('⚠️ Suhu terlalu tinggi - risiko stress panas.')
        recommendations['special_actions'].append('Berikan naungan parsial dan penyiraman ekstra.')
    else:
        recommendations['special_actions'].append('Suhu optimal untuk pertumbuhan tanaman.')
    
    # Rekomendasi berdasarkan kondisi umum
    if condition == 'Perlu Perhatian':
        recommendations['special_actions'].insert(0, '⚠️ PRIORITAS: Kondisi tanaman memerlukan perhatian segera!')
        recommendations['special_actions'].append('Periksa tanda-tanda penyakit atau hama.')
        recommendations['special_actions'].append('Pastikan penggantian udara yang baik di sekitar tanaman.')
    elif condition == 'Cukup Baik':
        recommendations['special_actions'].insert(0, 'Lanjutkan monitoring rutin dan sesuaikan perawatan.')
    else:  # Sehat
        recommendations['special_actions'].insert(0, '✓ Kondisi tanaman sangat baik - pertahankan pola perawatan saat ini.')
    
    return recommendations


def compute_health_score(
    age_days: float,
    height_cm: float,
    leaf_count: int,
    temperature_c: float,
    humidity_pct: float,
) -> float:
    """
    Compute a numerical health score (0-10) from available measurements.
    Heuristic uses temperature/ humidity proximity to optimal ranges and
    growth indicators (height and leaf count relative to age).
    """
    # Base score
    score = 5.0

    # Optimal ranges (match evaluate_plant_condition)
    optimal_temp_min, optimal_temp_max = 25.0, 30.0
    optimal_humidity_min, optimal_humidity_max = 60.0, 85.0

    # Temperature contribution (±2 points)
    if optimal_temp_min <= temperature_c <= optimal_temp_max:
        score += 2.0
    else:
        temp_dev = min(10.0, abs((temperature_c - (optimal_temp_min + optimal_temp_max) / 2)))
        score -= min(2.0, temp_dev * 0.2)

    # Humidity contribution (±1.5 points)
    if optimal_humidity_min <= humidity_pct <= optimal_humidity_max:
        score += 1.5
    else:
        hum_dev = min(40.0, abs(humidity_pct - (optimal_humidity_min + optimal_humidity_max) / 2))
        score -= min(1.5, hum_dev * 0.05)

    # Growth indicators: expected rough height and leaf_count per age
    expected_height = max(1.0, age_days * 0.5)
    height_ratio = height_cm / expected_height
    score += max(-2.0, min(2.0, (height_ratio - 1.0) * 1.5))

    expected_leaves = max(1.0, age_days * 0.3)
    leaf_ratio = leaf_count / expected_leaves
    score += max(-1.5, min(1.5, (leaf_ratio - 1.0) * 1.2))

    # Age sensitivity: very young plants are more sensitive (small penalty if extreme)
    if age_days < 7:
        score -= 0.5

    # Clamp to 0-10
    score = max(0.0, min(10.0, score))
    return float(round(score, 2))


if __name__ == '__main__':
    train_model()
