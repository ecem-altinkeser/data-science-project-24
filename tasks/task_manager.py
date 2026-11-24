import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from sklearn.utils import resample


# Açıklama: CSV dosyasından veriyi pandas DataFrame olarak oku.
# Input: "tasks/death_note_characters.csv"
# Output: pandas.DataFrame
# Kolonlar örneği: ['name', 'iq', 'aggression', 'justice_score', 'city', 'is_suspect']
def load_character_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=",")


# Açıklama: Tüm kolon adlarını küçük harfe çevir ve boşlukları alt çizgiyle değiştir.
# Input: pd.DataFrame(columns=["Name", "Justice Score", "Is Suspect"])
# Output: pd.DataFrame(columns=["name", "justice_score", "is_suspect"])
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


# Açıklama: Belirtilen sütundaki eksik değerleri ortalama ile doldur.
# Input: "justice_score"
# Output: Null değerler ortalama ile doldurulmuş DataFrame
def fill_missing_scores(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].fillna(df[column].mean())

    return df


# Açıklama: city sütununu one-hot encode et. Önce şehir adlarını küçük harfe çevir ("Tokyo" → "tokyo").
# Diğer sütunlar korunmalı, sadece city sütunu yeni sütunlara dönüşmeli.
# Input: "Tokyo", "Kyoto", "Osaka"
# Output: city_tokyo, city_kyoto, city_osaka sütunları
def encode_city_column(df: pd.DataFrame) -> pd.DataFrame:

    encoded = pd.get_dummies(df["city"], prefix="city", prefix_sep="_")
    encoded.columns = encoded.columns.str.lower()
    df = pd.concat([df, encoded], axis=1)
    df = df.drop("city", axis=1)
    return df


# Açıklama: iq sütunundaki z-skoru 3'ten büyük olanları tespit et ve ayrı bir sütunda “is_outlier” olarak işaretle.
# Output: Yeni sütun: is_outlier (True/False)
def detect_outliers_iq(df: pd.DataFrame) -> pd.DataFrame:

    df["is_outlier"] = zscore(df["iq"]) > 3

    return df


# Açıklama: aggression sütununu min-max normalizasyon ile 0-1 arasına getir.
# Output: Yeni sütun: aggression_scaled
def scale_aggression(df: pd.DataFrame) -> pd.DataFrame:

    df["aggression_scaled"] = (df["aggression"] - df["aggression"].min()) / (df["aggression"].max() - df["aggression"].min())

    return df


# Açıklama: Aşağıdaki şekilde bir “suspect_score” oluştur:
# Formül: df['suspect_score'] = 0.4 * df['iq'] + 0.6 * df['aggresion_scaled]
def generate_suspect_score(df: pd.DataFrame) -> pd.DataFrame:

    df['suspect_score'] = 0.4 * df['iq'] + 0.6 * df['aggression_scaled']

    return df



# Açıklama: is_suspect == 1 olanları resample ederek sayıca is_suspect == 0 kadar yap.
def upsample_suspects(df: pd.DataFrame) -> pd.DataFrame:
    suspect_1 = df[df["is_suspect"] == 1]
    suspect_0 = df[df["is_suspect"] == 0]

    resampled_0 = resample(suspect_0, n_samples=len(suspect_1))
    df = pd.concat([suspect_1, resampled_0])

    return df


# Açıklama: Suspect score’a göre en yüksek n kişiyi getir.
# Input: n = 5
# Output: İlk 5 kişi (yüksekten düşüğe sıralı)
def get_top_suspects(df: pd.DataFrame, n: int) -> pd.DataFrame:

    return df.nlargest(n, "suspect_score")


# Açıklama: Temizlenmiş ve puanlanmış veriyi CSV’ye yaz.
def export_results(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path)
