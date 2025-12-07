import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from datetime import datetime
import time

# ==========================================
# 1. 설정 및 모델 로드
# ==========================================
MODEL_PATH = "./urgency_model_focal_v4"    # 모델 경로
DB_FILE = "complaints_db.csv"                # DB 파일
ADMIN_PASSWORD = "admin"                     # 관리자 비번

st.set_page_config(page_title="캠퍼스 안전지킴이 통합관제", page_icon=None, layout="wide")

# 모델 캐싱
@st.cache_resource
def load_ai_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        return model, tokenizer, device
    except Exception as e:
        st.error(f"모델을 찾을 수 없습니다: {e}")
        return None, None, None

model, tokenizer, device = load_ai_model()

# ==========================================
# 2. 기능 함수
# ==========================================
def predict_urgency(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    pred_idx = torch.argmax(probs).item()
    confidence = probs[pred_idx].item() * 100
    return pred_idx, confidence

def load_db():
    # [중요] 컬럼 명시
    columns = ["이름", "접수시간", "내용", "긴급도", "확신도", "상태"]
    
    if not os.path.exists(DB_FILE):
        return pd.DataFrame(columns=columns)
    
    df = pd.read_csv(DB_FILE)
    
    # 기존 파일 호환성 처리 (이름 컬럼이 없으면 추가)
    if "이름" not in df.columns:
        df["이름"] = "익명"
        
    return df

def save_to_db(name, text, label, conf):
    df = load_db()
    new_data = pd.DataFrame({
        "이름": [name],
        "접수시간": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "내용": [text],
        "긴급도": [label],
        "확신도": [f"{conf:.1f}%"],
        "상태": ["접수"]
    })
    df = pd.concat([new_data, df], ignore_index=True)
    df.to_csv(DB_FILE, index=False)

# ==========================================
# 3. 화면 UI
# ==========================================
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'admin_logged_in' not in st.session_state:
    st.session_state['admin_logged_in'] = False
if 'show_login_input' not in st.session_state:
    st.session_state['show_login_input'] = False

def go_home(): 
    st.session_state['page'] = 'home'
    st.session_state['show_login_input'] = False

def go_user(): st.session_state['page'] = 'user'

def go_admin(): 
    if st.session_state['admin_logged_in']:
        st.session_state['page'] = 'admin'
    else:
        st.warning("로그인이 필요합니다.")

# ------------------------------------------
# [A] 메인 홈
# ------------------------------------------
if st.session_state['page'] == 'home':
    st.title("캠퍼스 안전지킴이 통합 시스템")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("학생 / 교직원")
        if st.button("민원 접수하기 (사용자)", use_container_width=True, type="primary"):
            go_user()
            st.rerun()
            
    with col2:
        st.warning("시설 관리팀")
        if st.button("관제 모니터링 (관리자)", use_container_width=True):
            if st.session_state['admin_logged_in']:
                go_admin()
                st.rerun()
            else:
                st.session_state['show_login_input'] = not st.session_state['show_login_input']

        if st.session_state['show_login_input']:
            with st.form("login_form"):
                password = st.text_input("관리자 비밀번호를 입력하세요", type="password")
                login_btn = st.form_submit_button("로그인")
                
                if login_btn:
                    if password == ADMIN_PASSWORD:
                        st.session_state['admin_logged_in'] = True
                        st.success("로그인 성공!")
                        time.sleep(0.5)
                        go_admin()
                        st.rerun()
                    else:
                        st.error("비밀번호가 틀렸습니다.")

# ------------------------------------------
# [B] 사용자 화면
# ------------------------------------------
elif st.session_state['page'] == 'user':
    st.button("홈으로", on_click=go_home)
    st.title("시설물 안전 민원 접수")
    
    with st.form("complaint_form", clear_on_submit=True):
        name_input = st.text_input("신고자 이름", placeholder="예: 홍길동 (전자공학과)")
        text_input = st.text_area("민원 내용", height=150, placeholder="위험 상황을 자세히 적어주세요.")
        
        submitted = st.form_submit_button("접수하기")
        
        if submitted:
            if not name_input.strip():
                st.error("이름을 입력해주세요!")
            elif not text_input.strip():
                st.error("민원 내용을 입력해주세요!")
            else:
                with st.spinner("AI가 분석 중입니다..."):
                    label, conf = predict_urgency(text_input)
                    save_to_db(name_input, text_input, label, conf)
                    time.sleep(1)
                
                st.success(f"'{name_input}'님의 민원이 접수되었습니다.")
                
                if label == 2:
                    st.error("긴급 민원입니다! 관리자에게 즉시 알림을 보냈습니다.")
                elif label == 1:
                    st.warning("우선 처리 대상으로 분류되었습니다.")
                else:
                    st.info("일반 민원으로 접수되었습니다.")

# ------------------------------------------
# [C] 관리자 화면 (수정됨)
# ------------------------------------------
elif st.session_state['page'] == 'admin':
    if not st.session_state['admin_logged_in']:
        go_home()
        st.rerun()

    col_head1, col_head2 = st.columns([8, 1])
    with col_head1:
        st.title("실시간 안전 관제 센터")
    with col_head2:
        if st.button("로그아웃"):
            st.session_state['admin_logged_in'] = False
            go_home()
            st.rerun()
            
    st.button("홈으로", on_click=go_home)

    df = load_db()
    
    if df.empty:
        st.info("현재 접수된 민원이 없습니다.")
    else:
        df['긴급도'] = pd.to_numeric(df['긴급도'], errors='coerce').fillna(0).astype(int)

        total = len(df)
        high_cnt = len(df[df['긴급도'] == 2])
        med_cnt = len(df[df['긴급도'] == 1])
        
        m1, m2, m3 = st.columns(3)
        m1.metric("총 접수", f"{total}건")
        m2.metric("긴급(High)", f"{high_cnt}건", delta_color="inverse")
        m3.metric("중간(Medium)", f"{med_cnt}건")

        if not df.empty and df.iloc[0]['긴급도'] == 2:
             st.toast(f"긴급 민원 발생! - {df.iloc[0]['이름']}", icon="🔥")

        st.markdown("### 실시간 접수 현황")
        if st.button("새로고침", use_container_width=True):
            st.rerun()

        # ----------------------------------------------------
        # [수정] 정보 표시 디자인 개선 (확실하게 보이도록)
        # ----------------------------------------------------
        for index, row in df.iterrows():
            urgency = row['긴급도']
            
            if urgency == 2:
                container = st.error
                badge = "🚨 [긴급]"
            elif urgency == 1:
                container = st.warning
                badge = "⚠️ [중간]"
            else:
                container = st.success
                badge = "✅ [일반]"
            
            # 컨테이너 시작
            with container(f"{badge} 조치 필요"):
                # 1. 메타 정보 (이름, 시간, 확신도)를 굵은 글씨로 상단에 배치
                st.markdown(f"**👤 작성자: {row['이름']} | 🕒 시간: {row['접수시간']} | 🤖 AI 확신도: {row['확신도']}**")
                st.divider() # 구분선
                
                # 2. 내용 및 버튼
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f"#### {row['내용']}")
                with c2:
                    if st.button("조치 완료", key=f"btn_{index}"):
                        st.write("처리됨")