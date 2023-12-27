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
   return pd.read_csv(output)

######
# file_id = '1-1i_FLEQCP4MOL9VFg8fMwt2OLMscO7V'
# url = f'https://drive.google.com/uc?id={file_id}'

# # 임시 파일 경로
# output = 'combined_df.csv'

# # Google 드라이브에서 파일 다운로드
# gdown.download(url, output, quiet=False)

# # 데이터 불러오기
# combined_df = pd.read_csv(output)

# Streamlit 앱에 데이터를 표시합니다 (예시)
# st.write(combined_df.head())
######

# # Streamlit 앱의 제목 설정
# st.title('최적 질소 건조 SP 설정')

# # 데이터 로드 (임시 데이터 또는 사용자가 업로드한 데이터를 사용할 수 있음)
# # combined_df = pd.read_csv('path/to/your/data.csv')

# # 데이터 로드 대신 임시 데이터 생성 (실제 앱에서는 실제 데이터를 로드해야 함)
# combined_df = pd.DataFrame({
#    'Timestamp': pd.date_range(start='1/1/2023', periods=100, freq='D'),
#    'FI_S_105.PV_Value': np.random.rand(100) * 100,
#    # ... 다른 센서 데이터 ...
# })

# # 1. 사용자가 특정 기간을 설정하면 그 기간동안의 시계열 그래프를 보여줌
# st.subheader('1. 특정 기간의 시계열 그래프')
# start_date = st.date_input('시작 날짜', value=pd.to_datetime('2023-01-01'))
# end_date = st.date_input('종료 날짜', value=pd.to_datetime('2023-01-31'))

# # 필터링된 데이터
# filtered_df = combined_df[(combined_df['Timestamp'] >= start_date) & (combined_df['Timestamp'] <= end_date)]

# # 시계열 그래프 그리기
# st.line_chart(filtered_df.set_index('Timestamp'))

# # 2. 사용자가 특정 기간을 설정하면 그 기간동안의 PI 와 FI 들의 센서들의 사분위수를 보여줌
# st.subheader('2. 특정 기간의 센서 데이터 사분위수')
# st.write(filtered_df.describe())

# # 3. 사용자가 독립변수를 설정하면 종속변수의 예측값을 추출
# st.subheader('3. 센서 데이터를 기반으로 예측하기')
# # 사용자 입력 필드
# fi_105_val = st.number_input('FI_S_105.PV_Value', value=50.0)
# # ... 다른 센서 값 입력 필드 ...

# # 예측을 위한 독립 변수 배열 생성
# input_data = np.array([[fi_105_val,  # FI_S_105.PV_Value
#                        # ... 다른 센서 값 ...
#                        ]])

# # 예측 모델 (이 부분은 실제 모델로 대체해야 함)
# linear_model = LinearRegression()
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# # 이 부분에 모델 학습 코드 추가 (예: linear_model.fit(X, y))

# # 예측 버튼
# if st.button('예측하기'):
#    predicted_linear = linear_model.predict(input_data)
#    predicted_rf = rf_model.predict(input_data)
#    st.write(f'선형 회귀 예측값: {predicted_linear[0]}')
#    st.write(f'랜덤 포레스트 예측값: {predicted_rf[0]}')
