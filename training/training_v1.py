import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# =========================
# 1. 설정값 (경로 및 모델 설정)
# =========================
# 사용자가 생성한 라벨링 파일 이름
CSV_PATH = "final_result_smart_filtered.csv" 

# 모델이 저장될 폴더명 (현재 폴더 내에 생성됨)
OUTPUT_DIR = "./new_emergency_model_v1"

MODEL_NAME = "klue/bert-base"   # 한국어 처리에 강한 KLUE-BERT 사용
NUM_LABELS = 3                  # 0:일반, 1:중간, 2:긴급

# =========================
# 2. Seed 고정 (재현성 확보)
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# =========================
# 3. Dataset 정의 (PyTorch)
# =========================
class EmergencyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # 토크나이징
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

# =========================
# 4. 데이터 로드 & 전처리
# =========================
def load_data(csv_path):
    # 파일 존재 여부 확인
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {csv_path}\n같은 폴더에 csv 파일이 있는지 확인해주세요.")

    # 한글 파일 읽기 (utf-8-sig)
    print(f"[INFO] 데이터를 로드합니다: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # 컬럼 확인 및 데이터 병합
    df["title"] = df["title"].fillna("")   # 제목이 비어있으면 공백 처리
    df["content"] = df["content"].fillna("") # 내용이 비어있으면 공백 처리
    
    # 제목과 내용을 합쳐서 학습 데이터(text)로 사용
    df["text"] = df["title"] + " " + df["content"]

    # 필수 데이터가 없는 행 제거 (text가 없거나, emergency 라벨이 없는 경우)
    df = df.dropna(subset=["text", "emergency"])
    df = df.reset_index(drop=True)

    # 데이터 분할 (Train: 70%, Valid: 15%, Test: 15%)
    # stratify 옵션을 사용하여 긴급도(0,1,2) 비율을 유지하며 나눔
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["emergency"]
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["emergency"]
    )

    print(f"전체 데이터 수: {len(df)}")
    print(f"Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}")

    return train_df, valid_df, test_df

# =========================
# 5. 성능 평가 함수 (Metric)
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    # macro avg: 클래스별 불균형을 고려하여 평균
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}

# =========================
# 6. 메인 실행 함수
# =========================
def main():
    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 디바이스: {device}")

    # 1) 데이터 로드
    train_df, valid_df, test_df = load_data(CSV_PATH)

    # 2) 토크나이저 & 모델 로드 (KLUE-BERT)
    print("[INFO] 모델과 토크나이저를 다운로드/로드합니다...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )
    model.to(device)

    # 3) Dataset 생성
    train_dataset = EmergencyDataset(
        train_df["text"].tolist(),
        train_df["emergency"].tolist(),
        tokenizer,
    )
    valid_dataset = EmergencyDataset(
        valid_df["text"].tolist(),
        valid_df["emergency"].tolist(),
        tokenizer,
    )
    test_dataset = EmergencyDataset(
        test_df["text"].tolist(),
        test_df["emergency"].tolist(),
        tokenizer,
    )

    # 4) 학습 파라미터 설정 (TrainingArguments)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,          # 모델 저장 경로
        do_train=True,
        do_eval=True,
        num_train_epochs=5,             # 학습 반복 횟수
        
        per_device_train_batch_size=8,  # 배치 사이즈
        per_device_eval_batch_size=8,
        
        learning_rate=2e-5,             # 학습률
        weight_decay=0.01,
        
        logging_steps=10,               # 로그 출력 빈도
        save_strategy="epoch",          # 에폭마다 저장
        eval_strategy="epoch",          # [수정됨] evaluation_strategy -> eval_strategy
        load_best_model_at_end=True,    # 학습 종료 후 가장 성능 좋은 모델 불러오기
        metric_for_best_model="f1_macro", # 성능 판단 기준
    )

    # 5) Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6) 학습 시작
    print("[INFO] 학습을 시작합니다...")
    trainer.train()

    # 7) 최종 테스트셋 평가
    print("=== Test 성능 평가 ===")
    test_metrics = trainer.evaluate(test_dataset)
    print(test_metrics)

    # 8) 모델 저장
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n[완료] 모델과 토크나이저가 저장되었습니다 → {OUTPUT_DIR}")

if __name__ == "__main__":
    main()