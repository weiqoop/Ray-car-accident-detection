# Freeway 車流偵測 — 配置 & 訓練流程

> 高公局即時 CCTV 單類車輛偵測，YOLO11s 模型，Ray 四階段完整流水線。

---

## 📋 快速總覽

| 階段 | 工具 | 任務 | 輸入 | 輸出 |
|------|------|------|------|------|
| **Stage 1** | Ray Data | 鏡頭級切分 + 增強 | freeway_yolo/ (CCTV 標註) | freeway_prepared/dataset.yaml |
| **Stage 2** | Ray Tune | 超參搜尋（ASHA） | freeway_prepared/dataset.yaml | 最佳超參 |
| **Stage 3** | Ray Train | 正式訓練 yolo11s | freeway_prepared/ + 超參 | best.pt |
| **Stage 4** | Ray Serve | 多路即時推論儀表板 | best.pt + 即時 CCTV | :8000 儀表板 |

---

## 🔧 Stage 1 — 資料前處理（Ray Data）

### 目的
- 以「**鏡頭為單位**」隔離測試集（防止同鏡頭洩漏）
- 對訓練集做離線劣化增強（可選）
- 輸出標準 ultralytics 資料夾結構

### 執行

```bash
# 基本執行（無離線增強）
docker compose exec ray-head python scripts/prepare_freeway.py

# 帶離線增強：每張 train 圖額外產生 2 種劣化變體
docker compose exec ray-head python scripts/prepare_freeway.py --aug 2

# 自訂 held-out 測試鏡頭（預設 CCTV-N1-S-93.080-M）
docker compose exec ray-head python scripts/prepare_freeway.py \
    --test-cam CCTV-N1-M-93.080-M --val-ratio 0.2
```

### 參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `--root` | `/workspace/datasets/freeway_yolo` | 原始 CCTV 標註資料 |
| `--out-root` | `/workspace/datasets/freeway_prepared` | 輸出目錄 |
| `--test-cam` | `CCTV-N1-S-93.080-M` | held-out 測試鏡頭名稱（整顆隔離） |
| `--val-ratio` | `0.2` | 驗證集比例（20%） |
| `--aug` | `0` | 每張 train 圖的離線劣化變體數 |

### 關鍵設計

✓ **鏡頭級隔離** — test 鏡頭完全不進訓練，避免同源偏差  
✓ **分散前處理** — Ray Data 跨 3 節點 CPU 平行  
✓ **離線增強可選** — 低畫質 freeway 不一定需要，ultralytics 訓練時已做即時增強  

---

## 🎯 Stage 2 — 超參搜尋（Ray Tune，可選）

### 目的
自動尋找最佳的學習率、權重衰減、增強強度等超參。

### 執行

```bash
# 預設 ASHA 搜尋，2 trial，GPU 平行
docker compose exec ray-head python scripts/tune_freeway.py

# 更多 trial（需 VRAM 充足）
docker compose exec ray-head python scripts/tune_freeway.py --num-trials 4

# 自訂搜尋空間
docker compose exec ray-head python scripts/tune_freeway.py \
    --lr0-range 0.001 0.05 --weight-decay-range 1e-5 1e-3
```

### 參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `--data` | `freeway_prepared/dataset.yaml` | 資料集路徑 |
| `--num-trials` | `2` | 搜尋試驗數（多並行） |
| `--epochs` | `30` | 每 trial 的訓練 epoch |
| `--lr0-range` | `0.001 0.1` | 學習率搜尋範圍 |
| `--weight-decay-range` | `1e-5 1e-3` | 權重衰減搜尋範圍 |
| `--run-name` | `freeway_tune` | Ray Tune 執行名 |

### 預期輸出

```
Best trial: trial_0_lr0=0.0072,wd=4.2e-4
Best metric (val mAP50): 0.823
```

---

## 🚂 Stage 3 — 正式訓練（Ray Train）

### 目的
用最佳超參（或保守預設）訓練 yolo11s，並保存最佳權重。

### 執行

```bash
# 基本（預設保守超參）
docker compose exec ray-head python scripts/train_freeway.py

# 用 Stage 2 最佳超參覆寫
docker compose exec ray-head python scripts/train_freeway.py \
    --lr0 0.0072 --weight-decay 4.2e-4 --epochs 100

# 自訂訓練輪數與批次
docker compose exec ray-head python scripts/train_freeway.py \
    --epochs 150 --batch 32 --patience 40
```

### 參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `--data` | `freeway_prepared/dataset.yaml` | 資料集路徑 |
| `--weights` | `yolo11s.pt` | 初始權重（容器已預載） |
| `--imgsz` | `640` | 訓練圖大小 |
| `--epochs` | `100` | 訓練輪數 |
| `--batch` | `16` | 批次大小 |
| `--patience` | `30` | 早停耐心（無進展 N epoch 停止） |
| `--name` | `freeway_final` | 存檔名稱（→ `ray_results/freeway_final/`） |
| `--lr0` | `0.01` | 初始學習率 |
| `--lrf` | `0.1` | 最終學習率係數 |
| `--weight-decay` | `5e-4` | L2 正則化係數 |
| `--hsv-v` | `0.03` | HSV 值增強強度 |
| `--scale` | `0.1` | 隨機縮放幅度 |
| `--fliplr` | `0.5` | 左右翻轉概率 |
| `--mosaic` | `0.1` | Mosaic 增強概率 |

### 訓練進度追蹤

```bash
# 即時監看 GPU 使用 + Ray Tune 進度
docker compose exec ray-head python scripts/monitor.py
# → 開瀏覽器 http://localhost:8501

# 或直接看訓練 log
docker compose exec ray-head tail -f ray_results/freeway_final/results.csv
```

### 預期結果

- **訓練時間** ～ 2–4 小時（視 GPU 與 epoch 數）
- **mAP50** ～ 0.84–0.85
- **mAP50-95** ～ 0.74–0.75
- **最佳權重** 存於 `ray_results/freeway_final/weights/best.pt`

---

## 🚀 Stage 4 — 推論展示（Ray Serve）

### 目的
5 路高公局 CCTV 即時推論，車流密度分級，展示在儀表板。

### 啟動 Serve

```bash
# 啟動 Serve（前景，自動載入 best.pt）
docker compose exec ray-head python scripts/serve_dashboard.py
#   注意：compose 沒有 ray-serve service，不能用 `docker compose up ray-serve`

# CPU 推論並釋出 GPU（邊訓練邊看 MONITOR 用）
docker compose exec ray-head python scripts/serve_dashboard.py --no-gpu
```

### 存取儀表板

- **主儀表板** → `http://localhost:8000/`
  - 5 路相機即時幀
  - YOLO 車輛邊框
  - 車流密度徽章（LOW/MED/HIGH）

- **叢集監控** → `http://localhost:8501/` (RAY MONITOR)
  - 3 節點負載
  - Ray 四階段進度（车流 + 車禍）

---

## 📊 完整工作流（快速清單）

```bash
# ❶ 準備環境
docker compose up -d                           # 啟動 3 節點叢集

# ❷ Stage 1：前處理（～10 分鐘，Ray Data 分散）
docker compose exec ray-head python scripts/prepare_freeway.py

# ❸ Stage 2：超參搜尋（可選，～1 小時）
docker compose exec ray-head python scripts/tune_freeway.py --num-trials 2

# ❹ Stage 3：正式訓練（～3 小時，GPU）
docker compose exec ray-head python scripts/train_freeway.py --epochs 100

# ❺ Stage 4：上線（正確指令；compose 無 ray-serve service）
docker compose exec ray-head python scripts/serve_dashboard.py   # 載入 best.pt 上線

# ❻ 驗證
# 瀏覽器開 http://localhost:8000  → 5 路相機即時跑
# 瀏覽器開 http://localhost:8501 → 監控 3 節點 + 進度
```

---

## 🔍 常見問題 & 除錯

### Q1：Stage 1 很慢？
**A** Ray Data 依賴磁碟讀，原始 freeway_yolo/ 很大會導致首次掃描慢。可用 `--root /data/detrac` 直接讀掛載。

### Q2：訓練中 GPU OOM？
**A** 減少批次大小 `--batch 8`，或關閉 Serve 釋放 GPU：
```bash
docker compose down ray-serve
# 訓練...
docker compose up ray-serve
```

### Q3：怎麼知道訓練進度？
**A** 查看 log：
```bash
docker compose exec ray-head tail -f ray_results/freeway_final/results.csv
```
或開監控儀表板 `http://localhost:8501`。

### Q4：可以用不同的預訓練權重嗎？
**A** 支援 `yolo11n/s/m/l/x` 等模型，會自動載入：
```bash
docker compose exec ray-head python scripts/train_freeway.py \
    --weights yolo11m.pt --epochs 100
```

---

## 📈 效能基準（最終結果）

| 指標 | 數值 |
|------|------|
| **mAP50** | **0.833** |
| **mAP50-95** | **0.712** |
| **訓練時間** | ～3 小時（RTX 3060 Ti） |
| **推論延遲** | < 50ms 單幀（GPU） |
| **線上相機數** | 5 路即時 |

---

## 📚 相關檔案

| 檔案 | 用途 |
|------|------|
| `scripts/prepare_freeway.py` | Stage 1 入口 |
| `scripts/tune_freeway.py` | Stage 2 入口 |
| `scripts/train_freeway.py` | Stage 3 入口 |
| `src/data/freeway/pipeline.py` | Ray Data 前處理邏輯 |
| `src/train/freeway/trainer.py` | TorchTrainer 訓練包裝 |
| `src/serve/app.py` | Ray Serve + 儀表板後端 |
| `src/serve/dashboard.html` | 儀表板前端 |
| `src/monitor/state.py` | RAY MONITOR 狀態推斷 |

---

> **簡報用重點**：鏡頭級隔離防洩漏 + mAP50 **0.833** + 5 路即時上線。
