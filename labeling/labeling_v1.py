import pandas as pd
from kiwipiepy import Kiwi

# ---------------------------------------------------------
# 1. 데이터 로드
# ---------------------------------------------------------
filename = 'final_result_with_campus.csv'
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"오류: '{filename}' 파일을 찾을 수 없습니다.")
    exit()

# 제목 + 내용 통합
df['full_text'] = df['title'].fillna('') + " " + df['content'].fillna('')

# ---------------------------------------------------------
# 2. 키워드 설정 (요청사항 반영)
# ---------------------------------------------------------
kiwi = Kiwi()

# [2점: 긴급 (High)]
# (1) 형태소 분석(명사)으로 찾을 단어들 (단어 오탐지 방지용: 돈가스, 불편 등 제외)
high_nouns = {
    "누전", "합선", "스파크", "불", "연기", "화재", "그을음", 
    "가스", "보일러", "감전", "침수", "누수", "유수", "층간누수", 
    "위험", "붕괴", "파손", "균열"
}

# (2) 문장에 그대로 포함되어 있는지 확인할 구문들 (복합 단어, 동사형 등)
high_phrases = [
    "타는냄새", "타는 냄새", "전기타는냄새", "전기 타는 냄새",
    "가스냄새", "가스 냄새", "가스샘", "가스 샘",
    "물새", "물 새", "물 샘", 
    "위험함", "떨어짐", "깨짐"
]

# [1점: 중간 (Medium)] - 기존 유지
medium_nouns = {
    "고장", "수리", "작동", "에어컨", "난방", "전등", 
    "교체", "단선", "불량", "오작동", "수정"
}

# ---------------------------------------------------------
# 3. 분류 함수
# ---------------------------------------------------------
def classify_emergency_strict(text):
    if not isinstance(text, str) or not text.strip():
        return 0
        
    # 1단계: 문구(Phrase) 단순 매칭 확인 (가장 강력한 조건)
    # "타는 냄새", "물 새" 등을 여기서 잡습니다.
    for phrase in high_phrases:
        if phrase in text:
            return 2  # 긴급

    # 2단계: 형태소 분석 (명사 추출)
    tokens = kiwi.tokenize(text)
    nouns = set(token.form for token in tokens if token.tag.startswith('N'))
    
    # 3단계: 명사 키워드 매칭
    if nouns & high_nouns:
        return 2
        
    # 4단계: 중간(Medium) 확인
    if nouns & medium_nouns:
        return 1
        
    # 5단계: 일반(0)
    return 0

# ---------------------------------------------------------
# 4. 실행 및 저장
# ---------------------------------------------------------
print("데이터 분류 중... (예외 키워드 제외됨)")

df['emergency'] = df['full_text'].apply(classify_emergency_strict)

# full_text 삭제 및 컬럼 정렬 (emergency를 맨 끝으로)
cols = [c for c in df.columns if c != 'emergency' and c != 'full_text'] + ['emergency']
df = df[cols]

# 결과 분포 출력
print("\n[라벨링 결과 분포 (0:일반, 1:중간, 2:긴급)]")
print(df['emergency'].value_counts().sort_index(ascending=False))

# 파일 저장
output_path = 'final_result_labeled.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n파일 저장 완료: {output_path}")