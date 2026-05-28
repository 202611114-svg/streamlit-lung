import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# 1. 페이지 설정 및 한글 폰트
st.set_page_config(page_title="폐암 위험군 분석", layout="wide")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 2. 레드 테마 CSS
st.markdown("""
    <style>
    .main { background-color: #fffafa; }
    h1 { color: #d32f2f; font-weight: 800; }
    .stButton>button {
        background-color: #d32f2f; color: white; border-radius: 5px;
        width: 100%; height: 3em; font-weight: bold; border: none;
    }
    .stButton>button:hover { background-color: #b71c1c; }
    .status-box { padding: 25px; border-radius: 15px; color: white; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# 3. 데이터 및 모델 로드
@st.cache_resource
def load_data():
    base_path = os.path.dirname(__file__)
    model = joblib.load(os.path.join(base_path, 'lung_model.pkl'))
    scaler = joblib.load(os.path.join(base_path, 'lung_scaler.pkl'))
    df = pd.read_csv(os.path.join(base_path, 'lung.csv'))
    
    # cluster 컬럼 자동 생성
    X = df[['나이', '담배여부', '알코올']]
    df['cluster'] = model.predict(scaler.transform(X))
    return model, scaler, df

model, scaler, df = load_data()

# 4. 사이드바 입력
with st.sidebar:
    st.title("📌 환자 데이터")
    age = st.number_input("나이", value=50)
    smoking = st.slider("담배 수치", 0.0, 10.0, 2.0)
    alcohol = st.slider("알코올 수치", 0.0, 10.0, 2.0)
    run_btn = st.button("결과 분석")

# 5. 메인 화면
st.title("🚨 폐암 위험군 정밀 분석")
st.write("입력된 지표를 바탕으로 환자의 군집을 분류합니다.")
st.divider()

if run_btn:
    # 예측
    new_data = pd.DataFrame([[age, smoking, alcohol]], columns=['나이', '담배여부', '알코올'])
    pred = model.predict(scaler.transform(new_data))[0]

    # [수정 완료] 군집 번호 매칭
    # 0: 매우 건강군, 1: 건강군, 2: 중간 그룹, 3: 강한 폐암 위험군 설정에 기반하여
    # 사용자 요청: "2번이 건강군이다"를 반영한 커스텀 매핑
    mapping = {
        0: {"label": "매우 건강군", "color": "#2E7D32", "desc": "가장 안전한 그룹입니다."},
        1: {"label": "중간 위험군", "color": "#F57C00", "desc": "주의가 필요한 단계입니다."},
        2: {"label": "건강군", "color": "#1976D2", "desc": "일반적인 건강 상태입니다."}, # 요청 반영
        3: {"label": "강한 폐암 위험군", "color": "#D32F2F", "desc": "매우 위험합니다. 정밀 검사가 필요합니다."}
    }
    
    res = mapping[pred]

    c1, c2 = st.columns([1, 1.2])

    with c1:
        st.subheader("分析 결과")
        st.markdown(f"""
            <div class="status-box" style="background-color: {res['color']};">
                <h2 style="margin:0;">{res['label']}</h2>
                <p style="font-size:1.2em;">현재 환자는 <b>군집 {pred}번</b>에 해당합니다.</p>
                <hr style="opacity:0.5;">
                <p>{res['desc']}</p>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.subheader("데이터 분포")
        fig, ax = plt.subplots()
        # 군집별 색상 (매핑 순서대로)
        colors = {0:'#2E7D32', 1:'#F57C00', 2:'#1976D2', 3:'#D32F2F'}
        
        for i in range(4):
            tmp = df[df['cluster'] == i]
            ax.scatter(tmp['나이'], tmp['담배여부'], c=colors[i], label=f"군집 {i}", alpha=0.4)
            
        ax.scatter(age, smoking, c='yellow', s=300, marker='*', edgecolor='black', label='내 위치')
        ax.set_xlabel("나이")
        ax.set_ylabel("담배 수치")
        ax.legend()
        st.pyplot(fig)
else:
    st.info("왼쪽 버튼을 눌러 분석을 시작하세요.")