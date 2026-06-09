# Accident 車禍偵測 — 配置 & 訓練流程

> 台灣高速公路監控真實事故偵測，MobileNetV2 CNN 逐幀二分類，Ray 四階段完整流水線。

---

## 📋 快速總覽

| 階段 | 工具 | 任務 | 輸入 | 輸出 |
|------|------|------|------|------|
| **Stage 1** | Ray Data | **影片級切分** + 解碼縮放（防洩漏） | TAD frames/ (360 支影片) | train/val/test.npz |
| **Stage 2** | Ray Tune | CNN 超參搜尋（ASHA，GPU 分數共享） | train.npz | 最佳超參 |
| **Stage 3** | Ray Train | 微調 MobileNetV2 + 存檔 | train/val.npz + 超參 | accident_tad.pt |
| **Stage 4** | Ray Serve | TAD 測試影片逐幀推論展示 | accident_tad.pt + test 影片 | 儀表板右下角面板 |

---

## 🔑 核心設計：影片級切分（防洩漏）

### 問題
如果用「幀級隨機切分」會發生什麼？

```
❌ 錯誤（幀級隨機）
   影片 A → 幀 0,1,2,3,4,5,6,7,8,9
            ↓ 隨機切
   train: 幀 0,2,4,6,8    ← 同片高度相關
   test:  幀 1,3,5,7,9    ← 同片高度相關
   
   結果：模型「背」test 幀，test acc 虛假 99%

✅ 正確（影片級切分）
   影片 A → train
   影片 B → val
   影片 C → test
   
   train/val/test 不含同一支影片的幀
   → 誠實 test acc = 92%（未見過的影片）
```

### 實現
[src/data/accident_tad/pipeline.py](src/data/accident_tad/pipeline.py) 中 `_video_split()`：
- 按標籤分層（正類/負類各自切分）
- 以影片為單位（train 70% / val 15% / test 15% 的影片數）
- 同片所有幀落在同一 split

---

## 🔧 Stage 1 — 資料前處理（Ray Data）

### 目的
- 從 TAD frames/ 讀取 360 支影片，依「影片」切分成 train/val/test
- 均勻抽幀（每支影片 60 幀）避免不平衡
- 解碼、縮放到 224×224，存為 uint8
- 生成 videos.json（影片級評估對齊用）

### 資料結構（TAD）

```
TAD/frames/
├── normal/                          ← 負類 (0)，250 支
│   ├── Normal_001.mp4/
│   │   ├── 0.jpg, 1.jpg, ..., 234.jpg
│   └── Normal_250.mp4/
│       └── ...
└── abnormal/
    ├── 01_Accident_001.mp4/        ← 正類 (1)，110 支（只用這類）
    │   └── 0.jpg, 1.jpg, ...
    ├── 02_IllegalTurn_xxx/         ← 排除（其他異常類型）
    └── ...
```

### 執行

```bash
# 基本（預設 60 幀/影片）
docker compose exec ray-head python scripts/prepare_accident_tad.py

# 自訂抽幀數（調整資料量）
docker compose exec ray-head python scripts/prepare_accident_tad.py --k 40

# 自訂 TAD 路徑
docker compose exec ray-head python scripts/prepare_accident_tad.py \
    --root "/workspace/datasets/Traffic Anomaly Dataset/TAD/frames"
```

### 參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `--root` | `/workspace/datasets/Traffic Anomaly Dataset/TAD/frames` | TAD frames 來源 |
| `--out` | `/workspace/datasets/accident_tad_seq` | 輸出目錄 |
| `--k` | `60` | 每支影片均勻抽幀數（控總資料量 + 壓不平衡） |

### 輸出

```
accident_tad_seq/
├── train.npz       ← X (幀), y (標籤), vid (影片 id)
├── val.npz
├── test.npz
└── videos.json     ← {vid_id: {name, label}}  (評估對齊用)
```

### 關鍵設計

✓ **影片級切分** — 同片幀不跨 split（防洩漏核心）  
✓ **分層抽樣** — 正負類各自切（保持平衡）  
✓ **均勻抽幀** — K 幀/影片，避免 53 萬幀爆量  
✓ **uint8 低精度** — 標準化延到訓練（ImageNet mean/std）  

---

## 🎯 Stage 2 — 超參搜尋（Ray Tune）

### 目的
自動搜尋最佳的 backbone、微調策略、dropout、增強強度、學習率等。

### 執行

```bash
# 基本（12 samples，ASHA 早停，gpu=0.5 單卡跑 2 trial 並行）
docker compose exec ray-head python scripts/tune_accident_cnn.py

# 更多搜尋樣本（需 GPU 記憶體充足；0.5 = 2 trial 並行）
docker compose exec ray-head python scripts/tune_accident_cnn.py --samples 24

# 減少搜尋樣本（快速測試）
docker compose exec ray-head python scripts/tune_accident_cnn.py --samples 4

# 每 trial 更多 epoch（更精確，但更慢）
docker compose exec ray-head python scripts/tune_accident_cnn.py --epochs 15
```

### 參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `--data-dir` | `accident_tad_seq` | 訓練資料路徑 |
| `--samples` | `12` | 搜尋試驗數（ASHA 會提早砍） |
| `--epochs` | `10` | 每 trial 的訓練 epoch 數 |
| `--gpu-per-trial` | `0.5` | GPU 分數（0.5 = 單卡 2 trial） |
| `--run-name` | `accident_cnn_tune` | Ray Tune 執行名 |

### 搜尋空間

```python
backbone    : mobilenet_v2  或  resnet18
freeze      : True         或  False         (凍結主幹參數)
dropout     : 均勻 [0.0 ~ 0.4]
aug         : 均勻 [0.1 ~ 0.3]              (增強強度)
lr          : 對數 [1e-4 ~ 3e-3]            (學習率)
weight_decay: 對數 [1e-6 ~ 1e-3]            (L2 正則)
batch       : 32  或  64                    (批次大小)
```

### 物件存儲共享技巧

```
driver 端（執行一次）：
  arrays = load_arrays(data_dir)           ← 載入 2.3GB train array
  ray.put(arrays)                          ← 放進共享記憶體

各 trial 子程序：
  tune.with_parameters(trainable, data=arrays)
  ↓
  各 trial 讀同一份 arrays（零拷貝）
  ↓
  2 trial 不會各複製 2.3GB（只在 object store 1 份）
```

**優勢**：2 trial ≈ 2.3GB + 各 trial overhead，不會爆掉 8GB 記憶體。

### 預期輸出

```
=== 最佳超參 ===
backbone: mobilenet_v2
freeze: False
dropout: 0.25
aug: 0.18
lr: 7.5e-4
weight_decay: 3.2e-5
batch: 64

=== 最佳驗證指標 ===
f1: 0.835, precision: 0.85, recall: 0.82, acc: 0.92
```

---

## 🚂 Stage 3 — 正式訓練（Ray Train）

### 目的
用 Stage 2 最佳超參（或保守預設）微調 MobileNetV2，儲存最終權重。

### 執行

```bash
# 基本（預設保守超參）
docker compose exec ray-head python scripts/train_accident_cnn.py

# 用 Stage 2 最佳超參覆寫
docker compose exec ray-head python scripts/train_accident_cnn.py \
    --backbone mobilenet_v2 --freeze --dropout 0.25 --aug 0.18 \
    --lr 7.5e-4 --weight-decay 3.2e-5 --epochs 15

# 自訂訓練輪數
docker compose exec ray-head python scripts/train_accident_cnn.py \
    --epochs 20 --batch 64
```

### 參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `--data-dir` | `accident_tad_seq` | 訓練資料路徑 |
| `--backbone` | `mobilenet_v2` | 主幹網路（`mobilenet_v2` 或 `resnet18`） |
| `--freeze` | `True` | 凍結 ImageNet 預訓練層（微調模式） |
| `--dropout` | `0.2` | Dropout 比例 |
| `--aug` | `0.2` | 訓練時增強強度（0.0 ~ 1.0） |
| `--lr` | `1e-3` | 初始學習率 |
| `--weight-decay` | `1e-4` | L2 正則化係數 |
| `--batch` | `64` | 批次大小 |
| `--epochs` | `15` | 訓練輪數 |
| `--save-path` | `/workspace/ray_results/accident_tad_final/accident_tad.pt` | 權重存檔路徑 |
| `--run-name` | `accident_cnn_final_raytrain` | Ray Train 執行名 |

### 架構

```
ImageNet 預訓練 MobileNetV2
  ↓
  224×224 RGB 圖輸入
  ↓
[凍結 or 解凍] 主幹層
  ↓
  Global Average Pooling
  ↓
[自寫] 分類頭：Linear(1280) → Sigmoid → 二分類邏輯值
  ↓
  BCE Loss (有正負樣本權重調整)
```

### 訓練進度

```bash
# 即時監看
docker compose exec ray-head python scripts/monitor.py
# → http://localhost:8501

# 查看 log
docker compose exec ray-head tail -f ray_results/accident_tad_final_raytrain/worker_0/logs/worker_0.log
```

### 預期結果

- **訓練** 15 epoch 後存最佳權重（解凍微調 lr 1e-4）
- **測試影片級 ROC-AUC** ~ 0.96 / F1 ~ 0.91（評估用 eval_accident_tad.py，主指標）
- **最終權重** 存於 `ray_results/accident_tad_final/accident_tad.pt`

> 注：訓練用 F1（幀級），評估用 ROC-AUC（影片級，更重要）。

---

## 🔬 評估 — 影片級指標（最重要）

### 目的
驗證模型在**未見過的影片**上的效能（誠實評估）。

### 執行

```bash
# 自動計算 frame-level + video-level 所有指標
docker compose exec ray-head python scripts/eval_accident_tad.py

# 自訂聚合方式（預設 90 percentile）
docker compose exec ray-head python scripts/eval_accident_tad.py --agg-pct 85

# 自訂模型路徑
docker compose exec ray-head python scripts/eval_accident_tad.py \
    --model /workspace/ray_results/accident_tad_final/accident_tad.pt
```

### 參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `--model` | `accident_tad.pt` | 模型權重路徑 |
| `--data-dir` | `accident_tad_seq` | 資料集路徑（test.npz） |
| `--agg-pct` | `90` | 聚合分位數（取每片幀分數的第 90 百分位） |

### 輸出

```
=== TAD test 幀級(xxxx 幀，事故 xxxx)===
[PR-AUC] 0.8923  (隨機=0.3229)
[最佳F1] 0.8960 @thr0.05  P=0.9100 R=0.8820

=== TAD test 影片級(54 片，事故 17，聚合=每片第90百分位)===
[ROC-AUC] 0.9603   [PR-AUC] 0.9006
[最佳F1] 0.9090 @thr0.10  P=0.9380 R=0.8820
[混淆矩陣] TP=15 FP=1 FN=2 TN=36  acc=0.9440
```

### 指標說明

| 指標 | 意義 |
|------|------|
| **影片級 ROC-AUC** | 在 54 支測試影片上，分辨事故/正常的能力 |
| **影片級 F1** | 精準率與召回率的調和平均 |
| **混淆矩陣** | TP 15/17 (88% 事故抓到) + FP 1/37 (97% 正常沒誤報) |

---

## 🚀 Stage 4 — 推論展示（Ray Serve）

### 目的
在儀表板右下角展示 TAD **測試影片**的逐幀推論（與車流相機 demo 並列）。

### 機制

```
Ray Serve（同一 replica）
  ├─ 左側：5 路高公局 CCTV + YOLO 推論（Freeway）
  └─ 右下角：TAD 測試影片逐幀切換 + MobileNetV2 推論（Accident）
      • 每 2 秒換下一幀
      • 顯示模型預測分數 + 徽章（⚠️ 事故 / ✓ 正常）
```

### 自動選片邏輯

```python
# _build_tad_demo() 自動挑有「明確上升趨勢」的影片
# 目的：demo 呈現「模型逐漸提高信心」的視覺感
# 選出 ~3 事故片 + ~3 正常片，優先有明顯 onset 的

範例：
  事故片：[0.1, 0.2, 0.3, 0.5, 0.8, 0.95]  ← 逐漸升高（好看）
  正常片：[0.05, 0.08, 0.1, 0.12]          ← 持續低（好看）
```

### 啟動

```bash
# 啟動 Serve（前景，自動載入 accident_tad.pt + best.pt）
docker compose exec ray-head python scripts/serve_dashboard.py
#   注意：compose 沒有 ray-serve service，不能用 `docker compose up ray-serve`

# 驗證
# 瀏覽器開 http://localhost:8000
# 看右下角「車禍偵測」面板
```

### 儀表板區塊

```
┌─────────────────────────┬────────────────┐
│ 5 路高公局 CCTV         │ 車禍偵測 demo   │
│ + YOLO 車輛框           │ (TAD test 影片) │
│                         │                │
│ Freeway 訊號            │ 影片 1/6        │
│                         │ 幀 3/60         │
│                         │                │
│                         │ [幀預覽]       │
│                         │ 分數: 0.75     │
│                         │ ⚠️ 偵測到事故   │
└─────────────────────────┴────────────────┘
```

---

## 📊 完整工作流（快速清單）

```bash
# ❶ 準備環境
docker compose up -d                           # 啟動 3 節點叢集

# ❷ Stage 1：前處理（～5 分鐘，Ray Data 分散）
docker compose exec ray-head python scripts/prepare_accident_tad.py

# ❸ Stage 2：超參搜尋（可選，～30 分鐘，GPU 2 trial 並行）
docker compose exec ray-head python scripts/tune_accident_cnn.py --samples 12

# ❹ Stage 3：正式訓練（～10 分鐘，GPU）
docker compose exec ray-head python scripts/train_accident_cnn.py --epochs 15

# ❺ Stage 4：評估 + 上線
docker compose exec ray-head python scripts/eval_accident_tad.py  # 驗證指標
docker compose exec ray-head python scripts/serve_dashboard.py     # 啟動 demo（無 ray-serve service）

# ❻ 驗證
# 瀏覽器開 http://localhost:8000  → 右下角看 TAD demo 面板
# 瀏覽器開 http://localhost:8501 → MONITOR 看四階段進度 (車流 + 車禍 都亮)
```

---

## 🔍 常見問題 & 除錯

### Q1：為什麼要「影片級切分」？
**A** 防止資料洩漏。幀級隨機切會讓同片相鄰幀同時進 train/test，導致虛假高精度（99% 假象 → 實際 92%）。

### Q2：訓練中記憶體爆掉？
**A** 檢查 object store 的 data 有沒有洩漏。確保 tune.with_parameters 正確注入 arrays。或改用 `--gpu-per-trial 1.0` 單 trial 模式（慢但安全）。

### Q3：test 影片級 F1 比 val 高（0.88 vs 0.84）？
**A** 正常。val 和 test 來自不同影片分層，分佈可能不同。只要差距 < 10% 就無過擬合。

### Q4：模型怎麼判定事故？
**A** MobileNetV2 看完整幀，逐幀輸出 [0, 1] 的二分類分數。demo 用 90 percentile 聚合成「該片是否事故」。精確定位撞擊瞬間需要逐幀時刻標籤（TAD 沒有）。

### Q5：可以用 ResNet18 嗎？
**A** 可以，改 `--backbone resnet18`。但 MobileNetV2 推論更快（好處：real-time demo）。

---

## 📈 效能基準（最終結果）

| 指標 | 數值 |
|------|------|
| **影片級 ROC-AUC** | **0.960** |
| **影片級 PR-AUC** | **0.901** |
| **影片級 F1** | **0.909** |
| **訓練時間** | ～10 分鐘（15 epoch，RTX 3060 Ti） |
| **推論延遲** | < 10ms 單幀（GPU） |
| **Demo 刷新率** | 2 秒/幀（與 Freeway 同步） |

> **實戰意義**：17 支事故影片抓到 15 支（88%），37 支正常只誤報 1 支（97% 精準）。

---

## 📚 相關檔案

| 檔案 | 用途 |
|------|------|
| `scripts/prepare_accident_tad.py` | Stage 1 入口 |
| `scripts/tune_accident_cnn.py` | Stage 2 入口 |
| `scripts/train_accident_cnn.py` | Stage 3 入口 |
| `scripts/eval_accident_tad.py` | 評估指標（影片級） |
| `src/data/accident_tad/pipeline.py` | Ray Data 影片級切分邏輯 |
| `src/modeling/accident_cnn.py` | MobileNetV2 模型構建 |
| `src/train/accident_cnn/trainer.py` | 訓練迴圈、DataLoader |
| `src/serve/app.py` | Ray Serve + demo backend |
| `src/serve/dashboard.html` | 儀表板前端 |

---

> **簡報用重點**：影片級防洩漏設計 + ROC-AUC **0.96** + 混淆矩陣展示實戰能力。
