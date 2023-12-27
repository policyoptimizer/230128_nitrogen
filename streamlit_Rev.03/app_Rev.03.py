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
   df['timestamp'] = pd.to_datetime(df['timestamp'])  # 'timestamp'를 datetime 형식으로 변환 (필요한 경우)
   return df

data = load_data()  # 데이터 로드

# 날짜 선택 위젯
start_date = st.sidebar.date_input(
   '시작 날짜',
   min_value=min(data['timestamp']),
   max_value=max(data['timestamp']),
   value=min(data['timestamp'])
)

end_date = st.sidebar.date_input(
   '종료 날짜',
   min_value=min(data['timestamp']),
   max_value=max(data['timestamp']),
   value=max(data['timestamp'])
)

# 필터링된 데이터 프레임 생성 및 표시
filtered_df = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]
st.write(filtered_df)
