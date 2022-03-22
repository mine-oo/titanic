import numpy as np
import pandas as pd
import altair as alt

url='https://www.salesanalytics.co.jp/h3lk'
train=pd.read_csv(url)

#predictive modeling
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
train_X = train[['Fare']]
train_y = train[['Survived']]
logreg.fit(train_X, train_y)

#graph1
fig = alt.Chart(train).mark_bar(size=60).encode(
    x='Survived:O',
    y='count()',
    column=alt.Column('Pclass')
).properties(
    width=150,
    height=150
).interactive()

#graph2
fig2 = alt.Chart(train).mark_bar(size=60).encode(
        x='Survived:O',
        y='mean(Age)',
        column=alt.Column('Pclass')
    ).properties(
        width=150,
        height=150
    ).interactive()


#application
import streamlit as st

st.write("""
  # My Web Application
  ## This app was created by me!
  このアプリケーションは機械学習について学ぶ際、チュートリアルとして
  よく用いられるタイタニック号における生存率の予想問題を扱ったものです。
    """)

#show graph
st.subheader('タイタニック号生存者(0:死亡1:生存)')
submit = st.button('グラフを表示する')
if submit == True:
    st.write(fig)


#Survival Probability Prediction
st.write('次に示すのは、タイタニック号における乗客の乗船料と生存率の関係を')
st.write('機械学習モデルに学習させ、予測モデルを構築したものになります')
st.subheader('Survival Probability')

#input
minValue = int(np.floor(train['Fare'].min()))
maxValue = int(np.floor(train['Fare'].max()))
startValue = int((maxValue+minValue)/2)
fareVlue = st.slider('please_select fare', min_value=minValue, max_value=maxValue,
step=1, value=startValue)

fareVlue_df = pd.DataFrame([fareVlue], columns=['Fare'])

#predicter
pred_probs= logreg.predict_proba(fareVlue_df)

#output

st.write('生存率の予測:', pred_probs[0,1])



 


