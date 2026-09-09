import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 1. 설정값 (Configuration)
# ==========================================
MODEL_PATH = "./urgency_model_focal_v4"  # 학습된 모델 경로
DATA_PATH = os.path.join("trainingset", "final_result_plus_v2.csv")  # 데이터 파일
BATCH_SIZE = 16

# 라벨 이름 (0, 1, 2 순서대로)
LABEL_NAMES = ['General (일반)', 'Medium (중간)', 'High (긴급)']

# 한글 폰트 설정 (그래프 깨짐 방지)
import platform
from matplotlib import font_manager, rc

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    try:
        font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
        rc('font', family=font_name)
    except:
        pass
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 데이터셋 및 모델 준비
# ==========================================
class SimpleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_len, padding='max_length', truncation=True)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def get_predictions(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    print("AI가 데이터를 분석(예측)하는 중...")
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

# ==========================================
# 3. 메인 실행
# ==========================================
if __name__ == "__main__":
    # 1) 데이터 로드
    if not os.path.exists(DATA_PATH):
        print(f"❌ 오류: 데이터 파일 '{DATA_PATH}'를 찾을 수 없습니다.")
        exit()
        
    df = pd.read_csv(DATA_PATH)
    df["title"] = df["title"].fillna("")
    df["content"] = df["content"].fillna("")
    df["text"] = df["title"] + " " + df["content"]
    df = df.dropna(subset=["text", "emergency"])
    
    # 2) 모델 로드
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 오류: 모델 폴더 '{MODEL_PATH}'를 찾을 수 없습니다.")
        exit()
        
    print(f"모델 로드 중: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    
    # 3) 예측 수행
    dataset = SimpleDataset(df['text'].values, df['emergency'].values, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    y_true, y_pred = get_predictions(model, dataloader, device)
    
    # 4) 혼동 행렬 시각화
    print("혼동 행렬 그리는 중...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, annot_kws={"size": 16})
    
    plt.xlabel('Predicted Label (모델 예측값)', fontsize=14)
    plt.ylabel('True Label (실제 정답)', fontsize=14)
    plt.title('Confusion Matrix (긴급도 분류 결과)', fontsize=18)
    
    # 이미지 저장
    save_path = "confusion_matrix_v3.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 시각화 완료! '{save_path}' 파일로 저장되었습니다.")
    
    # 5) 텍스트 리포트 출력 (정밀도, 재현율 확인용)
    print("\n" + "="*50)
    print("📋 상세 분류 리포트")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))
    
    plt.show()