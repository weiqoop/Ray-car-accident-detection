# 高速公路 CCTV 車流／車禍即時偵測系統

以 **Ray**（Data／Tune／Train／Serve）與 **YOLO11** 為核心、在 Docker 中建置的交通智慧
監控系統。針對台灣高速公路局（高公局）即時 CCTV（固定機位、MJPEG 串流），提供
**車流密度偵測**與**車禍事件偵測**兩條獨立流水線，並以即時儀表板與叢集監控呈現。

叢集以**三個節點**運行（1 head + 2 worker，同一台機器的多容器，僅透過網路溝通），
兩條流程**各自完整走 Ray 標準四階段**：`Ray Data → Ray Tune → Ray Train → Ray Serve`。

---

## 目錄

1. [系統總覽](#1-系統總覽)
2. [兩條流程的設計](#2-兩條流程的設計)
3. [環境需求](#3-環境需求)
4. [快速開始](#4-快速開始)
5. [車流偵測流程（Freeway）](#5-車流偵測流程freeway)
6. [車禍偵測流程（Accident）](#6-車禍偵測流程accident)
7. [部署架構](#7-部署架構)
8. [叢集監控（RAY MONITOR）](#8-叢集監控ray-monitor)
9. [專案結構](#9-專案結構)
10. [效能與現況](#10-效能與現況)
11. [指令速查](#11-指令速查)

---

## 1. 系統總覽

### 1.1 兩個模型、兩條流程

| 流程 | 模型 | 型態 | 訓練資料 | 狀態 |
|------|------|------|----------|------|
| **車流偵測**（Freeway） | YOLO11s | 物件偵測（車輛） | 高公局 CCTV 自動標註（SAHI + yolo11x teacher） | ✅ 訓練完成、已上線 |
| **車禍偵測**（Accident） | MobileNetV2 | 逐幀二分類 | **TAD**（高速公路監控真實事故，影片級切分） | ✅ 訓練完成、已上線 |

> **車禍模型演進**：曾試「軌跡時序（AccidentBench）」與「圖片 CNN（Road Accidents）」兩條,
> 前者訊號太弱、後者疑似資料洩漏(99% 假象);最終採 **TAD**（freeway 監控同域 + 影片級切分），
> 得到誠實的影片級 **ROC-AUC 0.96**。舊兩條已封存於 `_archive/`,評選見 [report.md](report.md) §2.3。

### 1.2 Ray 技術堆疊

| 元件 | 車流（Freeway） | 車禍（Accident，TAD） |
|------|------|------|
| **Ray Data** | 影像切分／前處理（CPU 多節點） | **影片級切分**(防洩漏) + 解碼縮放（CPU 多節點） |
| **Ray Tune** | ASHA 搜偵測超參（ultralytics，GPU） | ASHA 搜 CNN 超參（GPU 分數共享；資料經 `ray.put` 物件存儲共享）|
| **Ray Train** | TorchTrainer 編排 yolo11s 訓練（GPU） | TorchTrainer 微調 MobileNetV2（GPU）|
| **Ray Serve** | 多鏡頭即時偵測儀表板（:8000） | 同 replica：右下角 TAD 測試影片逐幀判定展示 |

> **獨立的 RAY MONITOR**（[scripts/monitor.py](scripts/monitor.py)）以輕量 driver 連上叢集，
> 純觀察各 Ray 元件活動與節點負載，**不依賴 Serve、不佔 GPU**，叢集一啟動即可看，
> 右側 Pipeline 同時顯示**車流／車禍兩條流程**的階段進度。

---

## 2. 兩條流程的設計

### 2.1 車流：偵測 + 密度分級

高公局自身 CCTV 訓練 yolo11s 單類車輛偵測器,即時抓幀 → 偵測 → 依車輛數/佔用面積
分「LOW/MED/HIGH」密度等級,呈現在儀表板。

### 2.2 車禍：同域資料 + 影片級切分（防洩漏）

採 **TAD** 高速公路監控真實事故,**MobileNetV2 逐幀二分類**(事故/正常)。關鍵設計:

- **同域**:TAD 是 freeway 監控視角,與高公局同類,跨域風險最低。
- **影片級切分**:以「影片」為單位切 train/val/test,同一支影片的幀**不跨 split**——
  這是數字可信的核心(隨機切幀會把同片相鄰幀同時放進 train/test → 洩漏灌水)。
- **粗標籤限制**:TAD 無逐幀事故時刻,整片標同一類 → 模型擅長「判斷該片是否事故」
  (影片級 ROC 0.96),但**無法精準定位撞擊瞬間**(已知限制)。

> 詳細評估與資料集評選(軌跡法/圖片 CNN/TAD 三條對照)見 [report.md](report.md)。

---

## 3. 環境需求

### 3.1 硬體

| 項目 | 規格 |
|------|------|
| GPU | NVIDIA（head 持有 1 顆；workers 為 CPU-only） |
| CPU／RAM | 16 核（head 8 + 2×worker 4）／充足記憶體 |

### 3.2 軟體

容器映像（[Dockerfile](Dockerfile)）：CUDA + Python 3.10 + Ray + Ultralytics + PyTorch +
OpenCV（完整見 [requirements.txt](requirements.txt)）。

> 容器無對外網路；torchvision 預訓練權重（MobileNetV2/ResNet18）與 yolo11x 皆於 host
> 預先下載並放 `datasets/weights/`，程式以本地檔載入（`weights=None` + `load_state_dict`）。

### 3.3 資料掛載（[docker-compose.yml](docker-compose.yml)）

| 主機路徑 | 容器路徑 | 權限 | 用途 |
|----------|----------|------|------|
| `F:/dataset` | `/data/detrac` | 唯讀 | UA-DETRAC 車輛偵測原始資料 |
| `F:/DoTA` | `/data/dota` | 唯讀 | DoTA 事故資料（保留）|
| `./datasets` | `/workspace/datasets` | 可寫 | 轉檔資料、權重、**TAD frames**、處理後 npz |
| `./src` | `/workspace/src` | 唯讀 | 原始碼 |
| `./scripts` | `/workspace/scripts` | 唯讀 | 進入點腳本 |
| `./ray_results` | `/workspace/ray_results` | 可寫 | 訓練／搜參輸出 |

> TAD 資料放在 `./datasets/Traffic Anomaly Dataset/TAD/frames`,故沿用 `./datasets` 掛載,
> 不需另開掛載。舊的 `ACCIDENT`／`Road Accidents` 掛載已隨軌跡/圖片集封存而移除。

### 3.4 對外連接埠

| 連接埠 | 服務 |
|--------|------|
| 8265 | Ray Dashboard（原生）|
| 8000 | Ray Serve HTTP（車流相機儀表板）|
| 8501 | RAY MONITOR（叢集監控總覽，兩條流程）|

---

## 4. 快速開始

```powershell
# 1. 啟動 3 節點 Ray 叢集（1 head + 2 worker，共 CPU 16 / GPU 1）
docker compose up -d ray-head ray-worker-1 ray-worker-2

# 2. 開叢集監控總覽（不需 GPU，叢集一起來就能看）
docker compose exec -d ray-head python scripts/monitor.py
#    瀏覽器：http://localhost:8501/

# 3. 車流流程（Freeway）— 已訓練完成的模型在 ray_results/freeway_final/
#    （如需重跑，見 §5）

# 4. 車禍流程（Accident，TAD）— 三階段（TAD frames 置於 datasets/Traffic Anomaly Dataset/）
docker compose exec ray-head python scripts/prepare_accident_tad.py    # ① Ray Data（影片級切分）
docker compose exec ray-head python scripts/eval_accident_tad.py       # 評估（影片級 + 幀級）
#    ② Tune / ③ Train 完整指令見 §11

# 5. 即時儀表板（5 路車流 + 右下角車禍影片 demo，佔 GPU）
docker compose exec -d ray-head python scripts/serve_dashboard.py
#    瀏覽器：http://localhost:8000/

docker compose down   # 關閉叢集
```

> **GPU 配置**：整個叢集只有 1 顆 GPU、由 head 持有。GPU 綁定的工作（yolo 追蹤／偵測訓練／
> serve 推論）只能在 head；CPU 可平行的工作（**車禍 Ray Tune**）才會用到兩個 worker。

---

## 5. 車流偵測流程（Freeway）

以高公局自身 CCTV 訓練的 yolo11s 車輛偵測器，完整走 Ray 四階段。

| 階段 | 腳本 | 說明 |
|------|------|------|
| 前置標註 | [prelabel_freeway.py](scripts/prelabel_freeway.py) | yolo11x **teacher** + **SAHI 切片**自動標註高公局影像（352×240 小圖需 192×192 切片偵測遠處小車）|
| ① Ray Data | [prepare_freeway.py](scripts/prepare_freeway.py) | 鏡頭級切分 train/val/test（整鏡頭隔離避免洩漏），CPU 多節點前處理 |
| ② Ray Tune | [tune_freeway.py](scripts/tune_freeway.py) | ultralytics `model.tune(use_ray=True)` ASHA 搜 lr/scale/mosaic… |
| ③ Ray Train | [train_freeway.py](scripts/train_freeway.py) | `TorchTrainer` 編排 yolo11s 訓練（`optimizer=AdamW` 固定，讓搜到的 lr 生效）|

- 起點權重 yolo11s，單類 Vehicle，imgsz 640。
- 鏡頭級隔離：保留 1 整支鏡頭做 held-out test。
- 成果：**mAP50 0.833 / mAP50-95 0.712**，模型存於 `ray_results/freeway_final/weights/best.pt`。

---

## 6. 車禍偵測流程（Accident）

逐幀二分類 CNN，資料來源為 **TAD**（Traffic Anomaly Dataset，高速公路監控真實事故）。

### 6.1 資料來源與篩選

TAD `frames/` 下 `normal/`（250 片）與 `abnormal/`（260 片，分 7 種異常）。**純車禍偵測**
只取 `abnormal/01_Accident_*`（110 片）為正類、`normal`（250 片）為負類,其餘 5 種異常排除。
畫面 720×300 監控俯視,與高公局同域。

### 6.2 四階段

| 階段 | 腳本／模組 | 說明 |
|------|------|------|
| ① Ray Data | [prepare_accident_tad.py](scripts/prepare_accident_tad.py)<br>[data/accident_tad/](src/data/accident_tad/) | **影片級分層切分**(同片不跨 split) + 每片均勻抽 60 幀 → 解碼縮 224。輸出 `datasets/accident_tad_seq/{train,val,test}.npz`（含 `vid` 供影片級評估）|
| ② Ray Tune | [tune_accident_cnn.py](scripts/tune_accident_cnn.py) | ASHA 搜 backbone/freeze/lr/dropout/aug。資料 `ray.put` 進**物件存儲共享**,多 trial 零拷貝 |
| ③ Ray Train | [train_accident_cnn.py](scripts/train_accident_cnn.py) | `TorchTrainer` 微調 MobileNetV2,選優看 **val F1**,輸出 `accident_tad_final/accident_tad.pt` |
| 評估 | [eval_accident_tad.py](scripts/eval_accident_tad.py) | **影片級（主）** ROC/PR-AUC + 幀級 |

### 6.3 模型與不平衡處理

- **模型**（[modeling/accident_cnn.py](src/modeling/accident_cnn.py)）:MobileNetV2/ResNet18 backbone
  （ImageNet 預訓練,離線從掛載權重載入）+ 單 logit 分類頭。
- **不平衡**:`BCEWithLogitsLoss(pos_weight=neg/pos)`,選優看 **F1**（不看 accuracy）。
- **影片級評估**:每片聚合幀分數（取高分位）→ 判該片是否事故,對齊 TAD 影片級標籤。

### 6.4 結果（test 集）

| 層級 | 指標 | 數值 |
|---|---|---|
| **影片級** | ROC-AUC / PR-AUC / F1 | **0.960 / 0.901 / 0.909**（TP15 FP1 FN2 TN36）|
| 幀級 | PR-AUC / F1 | 0.892 / 0.896 |

> 完整評估、資料集評選與誠實標註見 [report.md](report.md)。

---

## 7. 部署架構

```
   高公局 CCTV（5 路 MJPEG）            TAD 測試影片（展示用）
          │ 抓幀                              │ 逐幀
   ┌──────▼────────┐                  ┌───────▼─────────┐
   │ YOLO 偵測(11s)│                  │ MobileNetV2 CNN │
   │ freeway best  │                  │ accident_tad.pt │
   └──────┬────────┘                  └───────┬─────────┘
   ┌──────▼────────┐                  ┌───────▼─────────┐
   │ 車數／密度分級 │                  │ 事故/正常 + 機率 │
   └──────┬────────┘                  └───────┬─────────┘
          └───────────┬──────────────────────┘
                ┌─────▼──────────┐
                │ Ray Serve      │ → :8000 儀表板（5 路 + 右下車禍面板）
                └────────────────┘
```

- 車流與車禍**共用同一個 Serve replica**（占 1 GPU）:背景輪詢 5 路相機跑 YOLO,
  右下角面板輪播 TAD 測試影片跑 CNN。
- 部署到高公局真實串流時,車禍 CNN 直接抓 frame 逐幀判定(同域,免追蹤)。

---

## 8. 叢集監控（RAY MONITOR）

獨立服務（[scripts/monitor.py](scripts/monitor.py)），唯讀查詢叢集狀態、不佔 GPU。

| 區塊 | 內容 |
|------|------|
| **叢集節點** | active 節點數、CPU/GPU/Object Store 總量、每節點負載與 Ray 任務數 |
| **Ray 元件活動** | Data／Tune／Train／Serve 即時 active/idle 與正在做什麼（含即時 log）|
| **Pipeline（兩條）** | **車流**（已完成可上線）與**車禍**（隨任務即時亮階段）的步驟圖 |

端點：`GET /`、`/cluster.json`、`/components.json`、`/pipeline.json`。

> Pipeline 狀態自動推斷（不寫死）：車流偵測到模型已存在即標完成；車禍以
> `accident_tad_seq/train.npz`、`accident_tad_final/accident_tad.pt` 是否存在 +
> serve replica 是否存活（事故模型部署其中）判定四階段。Tune/Train log 依案別讀對應
> 來源（freeway 走 ultralytics、accident 走 Ray Train），解析 mAP 或 F1/AP。

---

## 9. 專案結構

```
src/
├── core/             叢集連線（init_ray）
├── modeling/         accident_cnn.py：MobileNetV2/ResNet18 事故二分類
├── data/
│   ├── freeway/      高公局抓取(grabber)、SAHI 自動標註(prelabel)、鏡頭切分、Ray Data
│   ├── accident_cnn/ 逐幀事故圖 Ray Data（防洩漏 block 切分，共用基礎設施）
│   └── accident_tad/ TAD 影片級切分 Ray Data（主線）
├── train/
│   ├── freeway/      Ray Train 偵測訓練
│   └── accident_cnn/ trainer.py：CNN 訓練核心 + ray.put 物件存儲共享（Tune/Train 共用）
├── serve/            Ray Serve（app.py 車流推論 + TAD 影片 demo，dashboard.html）
└── monitor/          RAY MONITOR（state.py + overview.html，兩條 pipeline）

scripts/   prepare_/tune_/train_freeway、prelabel_freeway、
           prepare_accident_tad、prepare_/tune_/train_/eval_accident_cnn、eval_accident_tad、
           serve_dashboard、monitor
datasets/  TAD frames、DETRAC、權重、處理後 npz（不納入版控）
ray_results/  訓練與搜參輸出（不納入版控）
_archive/  封存的軌跡法/圖片集程式與舊 run（不納入版控）
```

---

## 10. 效能與現況

| 模型 | Test Set | 主要指標 | 狀態 |
|------|----------|----------|------|
| **Freeway**（yolo11s） | 1 整支鏡頭（鏡頭級隔離）| **mAP50 0.833 / mAP50-95 0.712** | ✅ 完成、已上線 |
| **Accident**（TAD CNN） | **影片級隔離** | **影片級 ROC-AUC 0.96 / F1 0.91**；幀級 PR-AUC 0.89 | ✅ 完成、已上線 |

說明：

- **Freeway** 完成 Ray 四階段並部署,5 路即時偵測,是系統的偵測前端骨幹。
- **Accident** 完成 Ray 四階段並接進儀表板。採 TAD 同域資料 + **影片級切分**,得到
  誠實的影片級 ROC 0.96(17 支事故抓 15、37 正常誤報 1)。
- **誠實標註**:0.96 為同域 in-domain;高公局無標註,跨域未驗證。TAD 粗標籤 → 模型
  判「事故場景」而非「撞擊瞬間」,不做精準時間定位。

> 完整評估與 Demo 見 [report.md](report.md);設計演進、資料集評選與除錯見 [NOTE.md](NOTE.md)。

---

## 11. 指令速查

| 階段 | 指令（前綴 `docker compose exec ray-head`）|
|------|------|
| 啟動 3 節點叢集 | `docker compose up -d ray-head ray-worker-1 ray-worker-2` |
| 叢集監控總覽 | `python scripts/monitor.py`（:8501，不佔 GPU）|
| **車流** ① Ray Data | `python scripts/prepare_freeway.py` |
| **車流** ② Ray Tune | `python scripts/tune_freeway.py` |
| **車流** ③ Ray Train | `python scripts/train_freeway.py` |
| **車禍** ① Ray Data | `python scripts/prepare_accident_tad.py` |
| **車禍** ② Ray Tune | `python scripts/tune_accident_cnn.py --data-dir /workspace/datasets/accident_tad_seq --run-name accident_tad_tune` |
| **車禍** ③ Ray Train | `python scripts/train_accident_cnn.py --data-dir /workspace/datasets/accident_tad_seq --save-path /workspace/ray_results/accident_tad_final/accident_tad.pt --run-name accident_tad_final_raytrain --no-freeze --lr 1e-4 --epochs 15` |
| **車禍** 評估 | `python scripts/eval_accident_tad.py`（影片級 + 幀級）|
| 車流＋車禍儀表板 | `python scripts/serve_dashboard.py`（:8000，佔 GPU）|
| 關閉叢集 | `docker compose down` |

> 透過 Bash/Git Bash 執行含 `/workspace/...` 絕對路徑參數時,前面加 `MSYS_NO_PATHCONV=1`
> 避免路徑被竄改成 `C:/Program Files/Git/...`。

---

## 附錄：開發筆記

設計演進、資料集評選與除錯過程記錄於 [NOTE.md](NOTE.md)。
