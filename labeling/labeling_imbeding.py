import pandas as pd
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------
# 1. 설정 및 로드
# ---------------------------------------------------------
filename = 'final_result_with_campus_kookje.csv'
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"오류: '{filename}' 파일을 찾을 수 없습니다.")
    exit()

# 제목 + 내용 통합
df['full_text'] = df['title'].fillna('') + " " + df['content'].fillna('')

print("[System] AI 모델을 로딩 중입니다... (잠시만 기다려주세요)")
kiwi = Kiwi()
embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# ---------------------------------------------------------
# 2. 키워드 설정
# ---------------------------------------------------------

# [확정 긴급] 헷갈릴 여지가 없는 확실한 위험 단어들 ("불", "가스" 제외)
definite_high_nouns = {
    "누전", "합선", "스파크", "화재", "그을음", "보일러", 
    "감전", "침수", "누수", "붕괴", "폭발", "화상", "응급"
}

# [확정 긴급 구문] 문장에 포함되면 즉시 긴급
definite_high_phrases = [
    "타는냄새", "타는 냄새", "전기타는냄새", "전기 타는 냄새",
    "가스냄새", "가스 냄새", "가스샘", "가스 샘", "물새", "물 새"
]

# [검증 필요] 이 단어가 나오면 AI가 문맥을 확인해야 함
ambiguous_keywords = {"불", "가스", "연기", "위험","파손","유리","전선","사고"}

# [보통/고장]
medium_nouns = {
    "고장", "수리", "작동", "에어컨", "난방", "전등", "형광등",
    "교체", "단선", "불량", "오작동", "수정", "안됨", "안 됨"
}

# ---------------------------------------------------------
# 3. AI 판단 기준 (앵커 문장)
# ---------------------------------------------------------
# 모호한 단어가 나왔을 때, 이 문장들과 의미가 비슷해야만 긴급으로 인정
real_emergency_anchors = [
    "건물에 불이 났습니다 화재가 발생했어요",
    "가스가 누출되어 폭발할 위험이 있습니다",
    "연기가 자욱하고 매캐한 냄새가 납니다",
    "화재 경보기가 울리고 대피해야 합니다",
    "바닥이 파손되어 안전사고의 위험이 있습니다",
    "유리가 깨져서 다칠 위험이 있습니다",
    "전선이 노출되어 감전 위험이 있습니다",
    "사고로 이어질 수 있습니다"
]

# 기준 문장 벡터화
anchor_embeddings = embed_model.encode(real_emergency_anchors)

# ---------------------------------------------------------
# 4. 스마트 필터링 함수
# ---------------------------------------------------------
def classify_smart_filter(text):
    if not isinstance(text, str) or not text.strip():
        return 0

    # 1. [확정 긴급] 구문 체크 (가장 우선)
    for phrase in definite_high_phrases:
        if phrase in text:
            return 2
            
    # 형태소 분석
    tokens = kiwi.tokenize(text)
    nouns = set(token.form for token in tokens if token.tag.startswith('N'))
    
    # 2. [확정 긴급] 명사 체크
    if nouns & definite_high_nouns:
        return 2

    # 3. [검증 필요] "불, 가스" 등이 포함된 경우 -> AI 유사도 검사
    if nouns & ambiguous_keywords:
        input_embedding = embed_model.encode(text)
        scores = util.cos_sim(input_embedding, anchor_embeddings)
        max_score = float(scores.max())
        
        # 유사도가 0.5 이상이어야 진짜 긴급 상황으로 인정
        if max_score > 0.5:
            return 2
        else:
            # 유사도가 낮으면 긴급 아님 -> 아래 Medium 체크로 넘어감
            pass 

    # 4. [보통] 고장/수리 관련
    if nouns & medium_nouns:
        return 1
        
    # 5. 일반
    return 0

# ---------------------------------------------------------
# 5. 실행 및 저장
# ---------------------------------------------------------
print("데이터 분류를 시작합니다... (스마트 필터링 적용)")

df['emergency'] = df['full_text'].apply(classify_smart_filter)

# 결과 정리 (컬럼 재배치)
cols = [c for c in df.columns if c != 'emergency' and c != 'full_text'] + ['emergency']
df = df[cols]

# 결과 분포 출력
print("\n[라벨링 결과 분포]")
print(df['emergency'].value_counts().sort_index(ascending=False))

# 파일 저장
output_path = 'final_result_kookje.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n파일 저장 완료: {output_path}")