import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 임의의 데이터 생성 및 모델 훈련
np.random.seed(0)
X = np.random.rand(100, 6)
y = np.random.rand(100, 4)
X_df = pd.DataFrame(X, columns=[f'PI_Sensor_{i}' for i in range(1, 7)])
y_df = pd.DataFrame(y, columns=[f'FI_Sensor_{i}' for i in range(1, 5)])
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Streamlit 앱
def predict_new_data(new_data):
   new_data_df = pd.DataFrame([new_data], columns=X_df.columns)
   predicted_values = model.predict(new_data_df)
   return pd.DataFrame(predicted_values, columns=y_df.columns)

st.title('PI to FI Sensor Value Prediction')

# 사용자 입력
input_values = []
for i in range(1, 7):
   value = st.slider(f'PI_Sensor_{i}', 0.0, 1.0, 0.5)
   input_values.append(value)

if st.button('Predict'):
   predicted_values = predict_new_data(input_values)
   st.write('Predicted FI Sensor Values:')
   st.write(predicted_values)
