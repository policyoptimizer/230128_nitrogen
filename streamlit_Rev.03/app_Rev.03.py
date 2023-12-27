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

# 날짜 선택 위젯 수정
start_date = st.sidebar.date_input('시작 날짜', value=min(data['FI_S_105.PV_Timestamp'].dropna()))
end_date = st.sidebar.date_input('종료 날짜', value=max(data['FI_S_105.PV_Timestamp'].dropna()))

# 데이터 유형과 필터링 문제를 디버깅하기 위한 추가 정보 출력
st.write('선택한 시작 날짜:', start_date)
st.write('선택한 종료 날짜:', end_date)
st.write('데이터 유형 (시작 날짜):', type(start_date))
st.write('데이터 유형 (FI_S_105.PV_Timestamp):', data['FI_S_105.PV_Timestamp'].dtype)

try:
   # 필터링된 데이터 프레임 생성
   filtered_df = data[(data['FI_S_105.PV_Timestamp'] >= pd.to_datetime(start_date)) &
                      (data['FI_S_105.PV_Timestamp'] <= pd.to_datetime(end_date))]
   st.write(filtered_df)
except Exception as e:
   st.error(f"오류 발생: {e}")