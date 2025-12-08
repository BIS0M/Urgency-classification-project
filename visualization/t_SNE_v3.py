import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 1. 설정값 (Configuration)
# ==========================================
MODEL_PATH = "./urgency_model_focal_v4"  # 학습된 모델 경로
DATA_PATH = "final_result_plus_v2.csv"  # 데이터 파일
SAMPLE_SIZE = 1000  # 샘플링 개수

# 라벨 이름
LABEL_MAP = {0: 'Low (일반)', 1: 'Medium (중간)', 2: 'High (긴급)'}

# 한글 폰트 설정
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
# 2. 데이터셋 및 모델 로드
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

def get_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    
    print("AI가 데이터를 분석하여 특징(Embedding)을 추출하는 중...")
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].numpy()
            
            # BERT 모델의 마지막 히든 스테이트 추출
            outputs = model(input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.extend(cls_embeddings)
            labels.extend(batch_labels)
            
    return np.array(embeddings), np.array(labels)

# ==========================================
# 3. 메인 실행 로직
# ==========================================
if __name__ == "__main__":
    # 1) 데이터 로드
    if not os.path.exists(DATA_PATH):
        print(f"❌ 오류: 데이터 파일 '{DATA_PATH}'를 찾을 수 없습니다.")
        exit()
        
    print(f"데이터 로드 중: {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"CSV 읽기 오류: {e}")
        exit()
    
    df["title"] = df["title"].fillna("")
    df["content"] = df["content"].fillna("")
    df["text"] = df["title"] + " " + df["content"]
    
    if "emergency" not in df.columns:
        print("❌ 오류: CSV 파일에 'emergency' 컬럼이 없습니다.")
        exit()
        
    df = df.dropna(subset=["text", "emergency"])
    
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        print(f"데이터가 많아 {SAMPLE_SIZE}개만 랜덤 샘플링합니다.")
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
    
    # 2) 모델 및 토크나이저 로드
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 오류: 모델 폴더 '{MODEL_PATH}'를 찾을 수 없습니다.")
        exit()
        
    print(f"모델 로드 중: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        from transformers import AutoModelForSequenceClassification
        temp_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        if hasattr(temp_model, 'bert'):
            model = temp_model.bert
        elif hasattr(temp_model, 'roberta'):
            model = temp_model.roberta
        else:
            model = AutoModel.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        exit()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.to(device)
    
    # 3) 임베딩 추출
    dataset = SimpleDataset(df['text'].values, df['emergency'].values, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    embeddings, labels = get_embeddings(model, dataloader, device)
    print(f"임베딩 추출 완료. 데이터 크기: {embeddings.shape}")
    
    # 4) t-SNE 차원 축소 [수정됨: n_iter 제거]
    print("t-SNE 변환 중... (시간이 조금 걸릴 수 있습니다)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30) # n_iter 제거됨
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # 5) 시각화
    print("그래프 그리는 중...")
    
    vis_df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': [LABEL_MAP[l] for l in labels]
    })
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=vis_df, 
        x='x', y='y', 
        hue='label', 
        style='label',
        palette={'High (긴급)': 'red', 'Medium (중간)': 'orange', 'Low (일반)': 'green'},
        s=100, 
        alpha=0.7
    )
    
    plt.title('BERT 임베딩 t-SNE 시각화 (긴급도 분포)', fontsize=16)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='긴급도')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    save_path = "tsne_visualization_v3.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 시각화 완료! '{save_path}' 파일로 저장되었습니다.")
    plt.show()