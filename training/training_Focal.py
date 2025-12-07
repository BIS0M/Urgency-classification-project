import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ==========================================
# 1. 설정값 (Configuration)
# ==========================================
# 학습에 사용할 데이터 파일명
CSV_PATH = "training\\final_result_plus_v2.csv" #경로수정해야함

# 모델이 저장될 폴더명
OUTPUT_DIR = "./urgency_model_focal_v4"

MODEL_NAME = "klue/bert-base"   # 한국어 성능이 좋은 KLUE-BERT
NUM_LABELS = 3                  # 0:일반, 1:중간, 2:긴급

# ==========================================
# 2. Seed 고정 (재현성 확보)
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==========================================
# 3. Focal Loss 정의 (핵심 1: 어려운 문제 집중)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: 모델의 예측값 (Logits)
        # targets: 실제 정답 (Labels)
        log_prob = F.log_softmax(inputs, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            targets, 
            weight=self.weight,
            reduction=self.reduction
        )

# ==========================================
# 4. Custom Trainer 정의 (핵심 2: 가중치 뻥튀기)
# ==========================================
class AdvancedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # -----------------------------------------------------------
        # [튜닝 포인트] 클래스별 가중치 설정
        # 0(일반): 1.0 (기준)
        # 1(중간): 1.5 (조금 더 신경 씀)
        # 2(긴급): 5.0 (매우 중요! 틀리면 벌점 5배)
        # -> 긴급을 자꾸 놓치면 5.0을 7.0, 10.0으로 더 올리세요.
        # -----------------------------------------------------------
        weights = torch.tensor([1.0, 1.5, 7.0], dtype=torch.float32)
        
        if torch.cuda.is_available():
            self.class_weights = weights.cuda()
        else:
            self.class_weights = weights
            
        # Focal Loss 장착
        self.loss_fct = FocalLoss(weight=self.class_weights, gamma=2.0)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Focal Loss로 오차 계산
        loss = self.loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# ==========================================
# 5. Dataset 정의
# ==========================================
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

# ==========================================
# 6. 데이터 로드 함수
# ==========================================
def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {csv_path}")

    print(f"[INFO] 데이터를 로드합니다: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # 전처리
    df["title"] = df["title"].fillna("")
    df["content"] = df["content"].fillna("")
    df["text"] = df["title"] + " " + df["content"]
    
    # 필수 데이터 확인
    if "emergency" not in df.columns:
        raise ValueError("CSV 파일에 'emergency' 컬럼이 없습니다.")
        
    df = df.dropna(subset=["text", "emergency"])
    df = df.reset_index(drop=True)

    # 데이터 분할 (Train / Valid / Test)
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["emergency"]
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["emergency"]
    )

    print(f"전체: {len(df)} | Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}")
    return train_df, valid_df, test_df

# ==========================================
# 7. 성능 평가 지표 (F1 Score 포함)
# ==========================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro") # Macro F1 중요!
    
    return {"accuracy": acc, "f1_macro": f1}

# ==========================================
# 8. 메인 실행 함수
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 디바이스: {device}")

    # 1) 데이터 로드
    train_df, valid_df, test_df = load_data(CSV_PATH)

    # 2) 토크나이저 & 모델 로드
    print("[INFO] 모델 준비 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    model.to(device)

    # 3) 데이터셋 생성
    train_dataset = EmergencyDataset(train_df["text"], train_df["emergency"], tokenizer)
    valid_dataset = EmergencyDataset(valid_df["text"], valid_df["emergency"], tokenizer)
    test_dataset = EmergencyDataset(test_df["text"], test_df["emergency"], tokenizer)

    # 4) 학습 파라미터 (F1 향상을 위한 튜닝)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        
        num_train_epochs=10,             # 충분히 학습 (5 -> 10)
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        
        learning_rate=1e-5,             # 천천히 꼼꼼하게 (2e-5 -> 1e-5)
        weight_decay=0.01,
        
        logging_steps=10,
        save_strategy="epoch",          # 에폭마다 저장
        eval_strategy="epoch",          # 에폭마다 평가
        
        load_best_model_at_end=True,    # 가장 성능 좋은 모델 선택
        metric_for_best_model="f1_macro", # F1 점수가 높은 모델을 최고로 침
        
        warmup_steps=50                 # 초반 안정화
    )

    # 5) Trainer 생성 (AdvancedTrainer 사용)
    trainer = AdvancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6) 학습 시작
    print("\n[INFO] 학습을 시작합니다 (Focal Loss + Class Weight 적용)...")
    trainer.train()

    # 7) 최종 평가
    print("\n=== 최종 Test 성능 평가 ===")
    test_metrics = trainer.evaluate(test_dataset)
    print(f"Accuracy : {test_metrics['eval_accuracy']:.4f}")
    print(f"F1 Macro : {test_metrics['eval_f1_macro']:.4f}")

    # 8) 저장
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n[완료] 모델이 저장되었습니다: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()