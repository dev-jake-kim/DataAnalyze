# Demand Forecast Model Architecture

## 개요

울산 50개 노드의 시간별 택시 수요를 예측하는 Transformer 기반 모델.
과거 h시간의 격자 수요 데이터를 보고 다음 시간대 각 노드의 수요를 예측한다.

---

## 입력 / 출력

| 항목 | 형태 | 설명 |
|---|---|---|
| 입력 (cell_demands) | [B, h, 50, 49] long | 배치×이력×노드×셀, temporal_grid에서 추출한 정수 수요 (0–8) |
| 출력 (pred) | [B, 50] float | 다음 timestep의 노드별 예측 수요 |
| 레이블 (labels) | [B, 50] float | graph_data.json의 노드 단위 집계 수요 (0–21) |

---

## 데이터 흐름

```
temporal_grid.npy  [T=4368, 96, 78] int32
  + node 셀 좌표 (graph_data.json)
      │
      ▼  __getitem__
cell_demands [h, 50, 49]  ←─ window[t-h:t, cell_rows, cell_cols]
labels [50]               ←─ node_demands[t]
      │
      ▼  DemandForecastModel.forward
Step 1: NodeEmbedder
      │
      ▼
node_embed [B, h, 50, d]
      │
      ▼  Step 2: DemandPredictor
pred [B, 50]
```

---

## Step 1: NodeEmbedder (`models/node_embedder.py`)

각 timestep마다 50개 노드의 49개 셀 수요를 Transformer로 인코딩해 노드 임베딩을 생성.
**노드 번호 정보를 주지 않음** — 순수 공간·수요 맥락만 인코딩.

### 2D Positional Encoding

셀의 격자 좌표 (x, y)를 각각 d/2 차원의 사인파로 인코딩 후 concat.

```
x_pe = sinusoidal(cell.x, d//2)   →  [50, 49, d//2]
y_pe = sinusoidal(cell.y, d//2)   →  [50, 49, d//2]
cell_pe = concat(x_pe, y_pe)      →  [50, 49, d_model]
```

cell_pe는 학습 불가 buffer로 등록 (init 시 한 번만 계산).

### 처리 과정

```
cell_demands [B, h, 50, 49]
  → demand_embedding_table (nn.Embedding(9, d))  →  [B, h, 50, 49, d]
  + cell_pe (broadcast)                          →  [B, h, 50, 49, d]
  reshape → [B*h*50, 49, d]
  → TransformerEncoder1 (num_layers=2)
  → mean pool (dim=1)  →  [B*h*50, d]
  reshape  →  [B, h, 50, d]
```

---

## Step 2: DemandPredictor (`models/demand_predictor.py`)

노드 임베딩 이력 + 노드 ID를 받아 다음 timestep 수요를 예측.

```
node_embed [B, h, 50, d]
  + node_id_embed (nn.Embedding(50, d))[None, None]  broadcast
  + temporal_PE [h, d][None, :, None, :]             broadcast
  →  tokens [B, h, 50, d]
  reshape  →  [B, h*50, d]
  →  TransformerEncoder2 (num_layers=4)
  reshape  →  [B, h, 50, d]
  mean(dim=1)  →  [B, 50, d]
  →  Linear(d→1) + ReLU  →  pred [B, 50]
```

---

## Loss 함수

```
loss = MAE(y, pred) + γ · mean((y + pred)² / (y + 1))
```

- **MAE** term: 절대 오차를 균등하게 줄임
- **Penalty** term: 수요가 큰 구간에서 과소예측 시 페널티 강화 (분모 y+1이 작으면 페널티 증가)
- **γ (gamma)**: 두 항의 비율 조정, 기본값 0.01

---

## 데이터 분할 (시간 순서 유지)

| 분할 | 비율 | 인덱스 범위 | 샘플 수 |
|---|---|---|---|
| Train | 70% | 0 ~ 3032 | ~3033 |
| Val | 15% | 3057 ~ 3687 | ~631 |
| Test | 15% | 3712 ~ 4343 | ~632 |

미래 데이터 누출 없음 — train 레이블 최대 timestep = 3056, val 시작 = 3057.

---

## 하이퍼파라미터 (configs/transformer.yaml)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| d_model | 128 | 임베딩 차원 (반드시 짝수) |
| nhead | 8 | Attention head 수 |
| num_encoder_layers_1 | 2 | NodeEmbedder Transformer 레이어 수 |
| num_encoder_layers_2 | 4 | DemandPredictor Transformer 레이어 수 |
| dim_feedforward | 512 | FFN 내부 차원 |
| dropout | 0.1 | Dropout 비율 |
| gamma | 0.01 | Loss penalty 비율 |
| history_len | 24 | 이력 시간 수 (h) |

---

## 실행

```bash
# 학습
conda run -n DA python train_model.py --config-name=transformer

# CPU 디버그
conda run -n DA python train_model.py --config-name=transformer device=cpu \
  train.per_device_train_batch_size=2 train.num_train_epochs=1
```
