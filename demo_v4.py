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
MODEL_PATH = "./urgency_model_focal_v4"
DB_FILE = "complaints_db.csv"
ADMIN_PASSWORD = "admin"

st.set_page_config(page_title="민원 분류 시스템", layout="wide")

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
    columns = ["제목", "이름", "접수시간", "내용", "긴급도", "확신도", "상태"]
    
    if not os.path.exists(DB_FILE):
        return pd.DataFrame(columns=columns)
    
    df = pd.read_csv(DB_FILE)
    
    if "제목" not in df.columns: df["제목"] = "제목 없음"
    if "이름" not in df.columns: df["이름"] = "익명"
        
    return df

def save_to_db(title, name, text, label, conf):
    df = load_db()
    new_data = pd.DataFrame({
        "제목": [title],
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
if 'page' not in st.session_state: st.session_state['page'] = 'home'
if 'admin_logged_in' not in st.session_state: st.session_state['admin_logged_in'] = False
if 'show_login_input' not in st.session_state: st.session_state['show_login_input'] = False

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
    st.title("민원 분류 시스템")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("사용자")
        if st.button("민원 접수하기", use_container_width=True, type="primary"):
            go_user()
            st.rerun()
            
    with col2:
        st.warning("관리자")
        if st.button("관리자 페이지", use_container_width=True):
            if st.session_state['admin_logged_in']:
                go_admin()
                st.rerun()
            else:
                st.session_state['show_login_input'] = not st.session_state['show_login_input']

        if st.session_state['show_login_input']:
            with st.form("login_form"):
                password = st.text_input("관리자 비밀번호를 입력하세요", type="password")
                login_btn = st.form_submit_button("로그인")
                if login_btn and password == ADMIN_PASSWORD:
                    st.session_state['admin_logged_in'] = True
                    st.success("로그인 성공!")
                    time.sleep(0.5)
                    go_admin()
                    st.rerun()
                elif login_btn:
                    st.error("비밀번호가 틀렸습니다.")

# ------------------------------------------
# [B] 사용자 화면
# ------------------------------------------
elif st.session_state['page'] == 'user':
    st.button("홈", on_click=go_home)
    st.title("민원 접수")
    
    with st.form("complaint_form", clear_on_submit=True):
        title_input = st.text_input("민원 제목", placeholder="예: 공학관 3층 복도 천장 누수")
        text_input = st.text_area("민원 내용", height=150, placeholder="상황을 자세히 적어주세요.")
        name_input = st.text_input("작성자 (이름/소속)", placeholder="예: 홍길동 (전자공학과)")
        
        submitted = st.form_submit_button("접수하기")
        
        if submitted:
            if not title_input.strip() or not text_input.strip() or not name_input.strip():
                st.error("모든 항목을 입력해주세요!")
            else:
                with st.spinner("AI가 내용을 분석 중입니다..."):
                    full_text = f"{title_input} {text_input}"
                    label, conf = predict_urgency(full_text)
                    save_to_db(title_input, name_input, text_input, label, conf)
                    time.sleep(1)
                
                if label == 2:
                    st.error(f"🔴 긴급 민원 접수! ({title_input})")
                elif label == 1:
                    st.warning(f"🟡 민원이 접수되었습니다. ({title_input})")
                else:
                    st.info(f"✅ 민원이 접수되었습니다. ({title_input})")

# ------------------------------------------
# [C] 관리자 화면 (자동 새로고침 추가됨)
# ------------------------------------------
elif st.session_state['page'] == 'admin':
    if not st.session_state['admin_logged_in']:
        go_home()
        st.rerun()

    col_head1, col_head2 = st.columns([8, 1])
    with col_head1:
        st.title("관리자 페이지")
    with col_head2:
        if st.button("로그아웃"):
            st.session_state['admin_logged_in'] = False
            go_home()
            st.rerun()
            
    st.button("홈", on_click=go_home)

    # [추가 기능] 실시간 모니터링 체크박스
    auto_refresh = st.checkbox("🔄 실시간 모니터링 (5초 자동 갱신)", value=False)

    df = load_db()
    
    if df.empty:
        st.info("현재 접수된 민원이 없습니다.")
    else:
        # 데이터 전처리 (오류 방지)
        df['긴급도'] = pd.to_numeric(df['긴급도'], errors='coerce').fillna(0).astype(int)
        df['제목'] = df['제목'].fillna("제목 없음")
        df['이름'] = df['이름'].fillna("익명")
        df['내용'] = df['내용'].fillna("내용 없음")

        # [통계]
        total = len(df)
        high_cnt = len(df[df['긴급도'] == 2])
        med_cnt = len(df[df['긴급도'] == 1])
        
        m1, m2, m3 = st.columns(3)
        m1.metric("총 접수", f"{total}건")
        m2.metric("🔴 긴급(High)", f"{high_cnt}건", delta_color="inverse")
        m3.metric("🟡 중간(Medium)", f"{med_cnt}건")

        # [알림]
        pending_urgent = df[(df['긴급도'] == 2) & (df['상태'] == '접수')]
        if not pending_urgent.empty:
             st.toast(f"🔴긴급 민원: {pending_urgent.iloc[0]['제목']}", icon="🚨")

        st.markdown("### 📋 민원 접수 목록")
        
        # 자동 모니터링이 꺼져있을 때만 수동 버튼 표시
        if not auto_refresh:
            if st.button("새로고침 🔄", use_container_width=True):
                st.rerun()

        # [정렬] 긴급도(내림차순) -> 접수시간(내림차순)
        df_sorted = df.sort_values(by=['긴급도', '접수시간'], ascending=[False, False])

        # [리스트 출력 Loop]
        for index, row in df_sorted.iterrows():
            urgency = row['긴급도']
            status = row['상태']
            
            # 긴급도별 아이콘 및 색상 설정
            if urgency == 2:
                icon = "🔴"
                label_text = "**[긴급]**"
                content_bg = "긴급 상황 내용"
            elif urgency == 1:
                icon = "🟡"
                label_text = "**[중간]**"
                content_bg = "민원 내용"
            else:
                icon = "🟢"
                label_text = "[일반]"
                content_bg = "민원 내용"

            # 리스트 제목 구성
            if status == "완료":
                display_title = f"✅ (완료) {row['제목']}"
            else:
                display_title = f"{icon} {row['제목']}"

            # Expander 시작
            with st.expander(display_title, expanded=False):
                
                # 1. 상단 정보 (작성자, 시간, 긴급도)
                st.markdown(f"### {label_text} {row['제목']}")
                
                meta_c1, meta_c2, meta_c3 = st.columns(3)
                with meta_c1:
                    st.markdown(f"**작성자:** {row['이름']}")
                with meta_c2:
                    st.markdown(f"**시간:** {row['접수시간']}")
                
                st.divider()
                
                # 2. 본문 내용 (가독성 위해 st.info 사용)
                st.markdown(f"**{content_bg}**")
                st.info(row['내용'], icon="📝")
                
                # 3. 조치 버튼
                if status == "접수":
                    # 버튼과 내용 사이 간격
                    st.write("") 
                    if st.button("조치 완료 처리", key=f"btn_finish_{index}", use_container_width=True):
                        df.at[index, '상태'] = '완료'
                        df.to_csv(DB_FILE, index=False)
                        st.success("상태가 '완료'로 변경되었습니다.")
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.success(f"조치 완료된 건입니다. ({row['접수시간']} 접수분)")

    # [핵심] 자동 새로고침 로직
    if auto_refresh:
        time.sleep(5)
        st.rerun()