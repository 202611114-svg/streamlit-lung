import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# 1. 페이지 설정
st.set_page_config(page_title="Lung Cancer AI Diagnosis", layout="wide", initial_sidebar_state="expanded")

# 2. 고해상도 디자인을 위한 커스텀 CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;700;900&display=swap');
    
    html, body, [class*="css"] { font-family: 'Pretendard', sans-serif; }
    
    /* 배경색 및 메인 타이틀 */
    .main { background-color: #fcfcfc; }
    .stTitle { color: #d32f2f; font-weight: 900; letter-spacing: -1px; }
    
    /* 진단 결과 카드 디자인 */
    .diagnosis-card {
        padding: 40px;
        border-radius: 20px;
        color: white;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 25px;
    }
    .diagnosis-label { font-size: 1.5rem; font-weight: 400; opacity: 0.9; }
    .diagnosis-name { font-size: 3.5rem; font-weight: 900; margin: 10px 0; }
    
    /* 사이드바 스타일링 */
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #eee; }
    .stButton>button {
        background: linear-gradient(135deg, #d32f2f 0%, #b71c1c 100%);
        color: white; border: none; padding: 15px; border-radius: 12px;
        font-weight: 700; font-size: 1.1rem; transition: 0.3s;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(211,47,47,0.4); }
    </style>
    """, unsafe_allow_html=True)

# 3. 데이터 로드 함수
@st.cache_resource
def load_resources():
    base_path = os.path.dirname(__file__)
    model = joblib.load(os.path.join(base_path, 'lung_model.pkl'))
    scaler = joblib.load(os.path.join(base_path, 'lung_scaler.pkl'))
    df = pd.read_csv(os.path.join(base_path, 'lung.csv'))
    
    # 전체 데이터 클러스터링 미리 수행
    X = df[['나이', '담배여부', '알코올']]
    df['cluster'] = model.predict(scaler.transform(X))
    return model, scaler, df

model, scaler, df = load_resources()

# 4. 사이드바 구성
with st.sidebar:
    st.markdown("<h2 style='color:#d32f2f;'>🏥 Patient Data</h2>", unsafe_allow_html=True)
    st.write("환자의 정보를 세부적으로 입력해주세요.")
    st.divider()
    
    age = st.number_input("나이 (Age)", min_value=1, max_value=120, value=50)
    smoking = st.slider("담배 수치 (Smoking Exposure)", 0.0, 10.0, 2.0, step=0.1)
    alcohol = st.slider("알코올 수치 (Alcohol Consumption)", 0.0, 10.0, 2.0, step=0.1)
    
    st.write("")
    run_btn = st.button("실시간 분석 시작")

# 5. 메인 레이아웃
st.title("🚨 Lung Cancer Risk AI Analysis")
st.write("인공지능 모델이 환자의 데이터를 기반으로 위험 군집을 분류합니다.")
st.divider()

if run_btn:
    # 데이터 예측
    new_data = pd.DataFrame([[age, smoking, alcohol]], columns=['나이', '담배여부', '알코올'])
    pred_idx = model.predict(scaler.transform(new_data))[0]

    # 군집 매핑 (사용자 요청: 2번이 건강군)
    mapping = {
        0: {"label": "매우 건강군", "color": "#2E7D32", "desc": "신체 지표가 매우 우수한 상태입니다."},
        1: {"label": "중간 위험군", "color": "#EF6C00", "desc": "주의가 필요한 단계입니다. 예방 조치를 권장합니다."},
        2: {"label": "일반 건강군", "color": "#1976D2", "desc": "정상 범위의 건강 상태를 유지하고 있습니다."},
        3: {"label": "강한 폐암 위험군", "color": "#D32F2F", "desc": "위험 수치가 매우 높습니다. 즉각적인 전문의 상담이 필요합니다."}
    }
    
    res = mapping[pred_idx]

    # 상단 결과 카드
    st.markdown(f"""
        <div class="diagnosis-card" style="background-color: {res['color']};">
            <div class="diagnosis-label">AI 분석 결과</div>
            <div class="diagnosis-name">{res['label']}</div>
            <div style="font-size: 1.2rem; opacity: 0.9;">현재 환자는 <b>{pred_idx}번 군집</b>에 배정되었습니다.</div>
            <p style="margin-top:15px; font-weight:bold;">{res['desc']}</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📍 군집 시각화 분석")
        # Plotly를 이용한 예쁜 산점도
        df['군집명'] = df['cluster'].map(lambda x: mapping[x]['label'])
        
        fig = px.scatter(
            df, x='나이', y='담배여부', color='군집명',
            color_discrete_map={v['label']: v['color'] for k, v in mapping.items()},
            opacity=0.5, template='plotly_white',
            labels={'나이': '환자 연령', '담배여부': '흡연 노출도'}
        )
        
        # 환자 본인 위치 추가 (큰 노란색 별)
        fig.add_trace(go.Scatter(
            x=[age], y=[smoking],
            mode='markers',
            marker=dict(color='yellow', size=25, symbol='star', line=dict(width=2, color='black')),
            name='분석 대상자'
        ))
        
        fig.update_layout(legend_title_text='군집 분류', margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 지표별 상세 분석")
        # 게이지 차트 등을 이용한 시각화
        st.write(f"현재 환자의 수치: 나이 {age}세 / 담배 {smoking} / 알코올 {alcohol}")
        
        # 각 지표별 막대 그래프
        metrics = pd.DataFrame({
            '지표': ['나이', '담배', '알코올'],
            '수치': [age, smoking * 10, alcohol * 10] # 스케일링 보정
        })
        fig_bar = px.bar(metrics, x='지표', y='수치', color='지표', 
                         color_discrete_sequence=['#bdbdbd', '#d32f2f', '#d32f2f'],
                         template='plotly_white')
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.warning(f"**알림:** 분석 데이터상 {res['label']} 그룹의 평균 흡연 수치는 {df[df['cluster']==pred_idx]['담배여부'].mean():.2f}입니다.")

else:
    # 분석 전 초기 화면
    st.markdown("""
        <div style="text-align: center; padding: 100px 0;">
            <img src="https://cdn-icons-png.flaticon.com/512/2865/2865766.png" width="150">
            <h2 style="color: #ccc;">왼쪽 사이드바에서 데이터를 입력하고 <br>분석 버튼을 눌러주세요.</h2>
        </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("본 AI 모델은 연구 목적으로 제작되었습니다. 정확한 진단은 반드시 의료기관을 방문하시기 바랍니다.")
