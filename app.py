import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gdown
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# 공유 가능한 링크에서 파일 ID를 추출합니다
# stream_df > nov.csv
# https://drive.google.com/file/d/1-1i_FLEQCP4MOL9VFg8fMwt2OLMscO7V/view?usp=sharing

def parse_timestamp(ts):
   if pd.isnull(ts):
       return np.nan
   if isinstance(ts, str):
       try:
           return datetime.strptime(ts, '%Y-%m-%d %p %I:%M')
       except ValueError:
           return np.nan
   return np.nan

@st.cache(allow_output_mutation=True)
def load_data():
  try:
      url = 'https://drive.google.com/uc?id=1-1i_FLEQCP4MOL9VFg8fMwt2OLMscO7V'
      output = 'combined_df.csv'
      gdown.download(url, output, quiet=False)
      df = pd.read_csv(output)
      return df
  except Exception as e:
      st.error(f'데이터 로딩 중 오류가 발생했습니다: {e}')
      return pd.DataFrame()  # 오류 발생시 빈 DataFrame 반환​

data = load_data()  # 데이터 로드

# 사용자로부터 날짜 범위 입력 받기
start_date = st.sidebar.date_input('시작 날짜', datetime(2023, 10, 15))
end_date = st.sidebar.date_input('종료 날짜', datetime(2023, 10, 31))

timestamp_cols = [col for col in data.columns if 'PV_Timestamp' in col]
sensor_dataframes = {}

for timestamp_col in timestamp_cols:
   # 타임스탬프 파싱 및 정렬
   data[timestamp_col] = data[timestamp_col].apply(parse_timestamp)
   data.sort_values(by=timestamp_col, inplace=True)

   # 센서 값 칼럼 이름 추출 및 해당 센서 데이터 선택
   value_col = timestamp_col.replace('Timestamp', 'Value')
   sensor_df = data[[timestamp_col, value_col]].dropna()
   sensor_df.fillna(method='ffill', inplace=True)

   # 사용자가 선택한 날짜 범위에 따라 데이터 필터링
   filtered_sensor_df = sensor_df[(sensor_df[timestamp_col] >= start_date) & (sensor_df[timestamp_col] <= end_date)]
   sensor_dataframes[timestamp_col] = filtered_sensor_df

# 시각화
st.subheader('선택한 기간 동안의 센서 값')
fig, ax = plt.subplots(figsize=(15, 10))

for timestamp_col, sensor_df in sensor_dataframes.items():
   value_col = timestamp_col.replace('Timestamp', 'Value')
   ax.plot(sensor_df[timestamp_col], sensor_df[value_col], label=value_col)

ax.set_xlabel('Timestamp')
ax.set_ylabel('Value')
ax.set_title('Sensor Values Over Time (Filtered)')
ax.legend()

st.pyplot(fig)
