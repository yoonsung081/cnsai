import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. 데이터 로드 (기존과 동일)
print("데이터 로드를 시작합니다...")
traffic_path = 'C:/Users/000/OneDrive/Desktop/cnsai/서울도시고속도로 노선별 시간대별 교통량(2014)..csv'
air_quality_path1 = 'C:/Users/000/OneDrive/Desktop/cnsai/서울시 대기질 자료 제공_2012-2015.csv'
air_quality_path2 = 'C:/Users/000/OneDrive/Desktop/cnsai/서울시 대기질 자료 제공_2016-2019.csv'
air_quality_path3 = 'C:/Users/000/OneDrive/Desktop/cnsai/서울시 대기질 자료 제공_2020-2021.csv'
try:
    df_traffic = pd.read_csv(traffic_path, encoding='cp949')
    df_air1 = pd.read_csv(air_quality_path1, encoding='cp949')
    df_air2 = pd.read_csv(air_quality_path2, encoding='cp949')
    df_air3 = pd.read_csv(air_quality_path3, encoding='cp949')
    print("데이터 로드 완료.")
except Exception as e:
    print(f"데이터 로드 중 오류: {e}")

# 2. 데이터 전처리 및 병합 (기존과 동일)
print("데이터 전처리를 시작합니다...")
df_air = pd.concat([df_air1, df_air2, df_air3], ignore_index=True)
df_air.columns = ['datetime', 'region', 'pm10', 'pm25']
df_traffic.columns = ['route', 'time', 'volume', 'category', 'unused']
df_traffic = df_traffic.drop(columns=['unused', 'category'])
df_traffic['volume'] = pd.to_numeric(df_traffic['volume'], errors='coerce')
df_traffic.dropna(subset=['volume'], inplace=True)
df_traffic['time'] = df_traffic['time'].astype(str).str.zfill(2)
traffic_by_time = df_traffic.groupby('time')['volume'].mean().reset_index()
traffic_by_time.columns = ['hour', 'avg_volume']
df_air['datetime'] = pd.to_datetime(df_air['datetime'])
df_air['hour'] = df_air['datetime'].dt.strftime('%H')
df_merged = pd.merge(df_air, traffic_by_time, on='hour', how='left')
df_merged.dropna(subset=['pm10', 'avg_volume'], inplace=True)
df_merged = df_merged.sample(n=100000, random_state=42)
print("데이터 전처리 완료.")

# 3. 등급 분류를 위한 데이터 재구성 (Feature Engineering)
print("분류 모델을 위한 데이터 재구성을 시작합니다...")

# 3.1. 미세먼지 등급(pm10_level) 컬럼 생성
def get_pm10_level(pm10):
    if pm10 <= 30:
        return '좋음'
    elif pm10 <= 80:
        return '보통'
    elif pm10 <= 150:
        return '나쁨'
    else:
        return '매우 나쁨'
df_merged['pm10_level'] = df_merged['pm10'].apply(get_pm10_level)

# 3.2. 시간 관련 특징 추가
df_merged['month'] = df_merged['datetime'].dt.month
df_merged['day_of_week'] = df_merged['datetime'].dt.dayofweek # 0:월요일, 6:일요일
df_merged['hour'] = df_merged['hour'].astype(int)

print("데이터 재구성 완료.")
print("\n--- 등급별 데이터 분포 ---")
print(df_merged['pm10_level'].value_counts())

# 4. 분류 모델 학습 및 평가
print("\n분류 모델 학습 및 평가를 시작합니다...")

# 사용할 특징(X)과 목표(y) 설정
features = ['avg_volume', 'hour', 'month', 'day_of_week']
target = 'pm10_level'

X = df_merged[features]
y = df_merged[target]

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 모델 초기화 및 학습
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, labels=['좋음', '보통', '나쁨', '매우 나쁨'])

print("\n--- 모델 평가 결과 ---")
print(f"분류 정확도 (Accuracy): {accuracy:.4f}")
print("\n--- 상세 평가 보고서 ---")
print(report)