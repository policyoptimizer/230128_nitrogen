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

@st.cache(allow_output_mutation=True)
def load_data():
   url = 'https://drive.google.com/uc?id=1-1i_FLEQCP4MOL9VFg8fMwt2OLMscO7V'
   output = 'combined_df.csv'
   gdown.download(url, output, quiet=False)
   df = pd.read_csv(output)
   df['FI_S_105.PV_Timestamp'] = pd.to_datetime(df['FI_S_105.PV_Timestamp'])  # 타임스탬프를 datetime 형식으로 변환
   return df

data = load_data()  # 데이터 로드

# 날짜 선택 위젯
start_date = st.sidebar.date_input('시작 날짜', value=pd.to_datetime('2023-01-01'))
end_date = st.sidebar.date_input('종료 날짜', value=pd.to_datetime('2023-01-31'))

# 필터링된 센서 데이터프레임 생성
filtered_dataframes = {}
for timestamp_col, df in sensor_dataframes.items():
   # 가정: 각 df는 timestamp_col을 포함하고 있으며, 이는 datetime으로 변환될 수 있습니다.
   filtered_df = df[(pd.to_datetime(df[timestamp_col]) >= start_date) &
                    (pd.to_datetime(df[timestamp_col]) <= end_date)]
   filtered_dataframes[timestamp_col] = filtered_df

# 시각화
st.subheader('선택한 기간 동안의 센서 값')
fig, ax = plt.subplots(figsize=(15, 10))

for timestamp_col, sensor_df in filtered_dataframes.items():
   value_col = timestamp_col.replace('Timestamp', 'Value')
   ax.plot(pd.to_datetime(sensor_df[timestamp_col]), sensor_df[value_col], label=value_col)

ax.set_xlabel('Timestamp')
ax.set_ylabel('Value')
ax.set_title('Sensor Values Over Time (Filtered)')
ax.legend()

st.pyplot(fig)