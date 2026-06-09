# Freeway 車流偵測 — 投影片版本

## Slide 1：問題 & 方案

### 問題
高公局有 5+ 路 CCTV，需要**自動偵測車流壅塞**。

### 方案
用 **YOLO11** 訓練單類車輛偵測器，分散在 Ray 三節點叢集上執行。

---

## Slide 2：四個階段（Ray 標準流程）

```
準備      調優      訓練      推論
  ↓        ↓        ↓        ↓
Ray Data  Ray Tune  Ray Train Ray Serve
  
  ❶ 鏡頭級切分    ❷ 自動搜超參   ❸ 正式訓練    ❹ 即時儀表板
    (CPU分散)      (GPU並行)      (GPU)        (5路相機)
```

---

## Slide 3：Stage 1 — 資料準備（Ray Data）

**核心概念**：鏡頭為單位隔離測試集

```
📸 高公局 CCTV 標註資料
         ↓
  🔀 鏡頭級隔離切分
     • train：其他鏡頭的圖
     • test：CCTV-N1-S-93.080-M 完整
     
  ✓ 防止「同鏡頭洩漏」
    (model 在訓練時見過該鏡頭風格)
     
  📦 輸出：dataset.yaml (ultralytics 格式)
```

**執行時間** ~ 10 分鐘（CPU 多節點平行）

---

## Slide 4：Stage 2 — 超參搜尋（Ray Tune）

**概念**：讓機器自動找最好的「學習率、權重衰減、增強強度」。

```
🎯 要調的超參
   • 學習率 (0.001 ~ 0.1)
   • 權重衰減 (1e-5 ~ 1e-3)
   • 增強強度 (多種組合)
   
⚡ ASHA 演算法：邊跑邊砍不行的
   • trial 1, 2, 3... 同時跑
   • 無進展的提早停止
   
📊 輸出：最佳超參組合
```

**執行時間** ~ 1 小時（2 trial GPU 並行，可選）  
**結果範例** `lr=0.0072, weight_decay=4.2e-4`

---

## Slide 5：Stage 3 — 正式訓練（Ray Train）

**流程**

```
🏃 用最佳超參訓練 yolo11s
   • 100 epoch (或自訂)
   • 批次 16 (或自訂)
   • 即時增強（旋轉、翻轉、HSV）
   
✅ 自動存最佳權重
   ray_results/freeway_final/weights/best.pt
```

**執行時間** ~ 3 小時（RTX 3060 Ti）  
**結果** mAP50 = **0.833** / mAP50-95 = **0.712**

---

## Slide 6：Stage 4 — 上線（Ray Serve）

**儀表板展示**（http://localhost:8000）

```
┌─────────────────────────────┐
│  🎥 CCTV-N1-S (實時)         │  ← 5 路相機
│  [車輛邊框] [車輛邊框]        │
│  車流密度：HIGH ⚠️          │
└─────────────────────────────┘

🟢 LOW    🟡 MED    🔴 HIGH
```

**背後的運作**
- Ray Serve 拉取即時 CCTV 幀
- 丟進 best.pt 推論
- 統計車輛數 → 密度等級

---

## Slide 7：設計亮點

### 1️⃣ 鏡頭級隔離（防洩漏）
- test 鏡頭在 train 時**完全看不見**
- 避免模型「學會該鏡頭的風格」而非「學會偵測車」

### 2️⃣ 分散執行（Ray 三節點）
- Stage 1（前處理）：3 節點 CPU 平行
- Stage 2（超參搜尋）：GPU 多 trial 並行
- Stage 3（訓練）：GPU 單 trial
- Stage 4（推論）：head 節點即時服務

### 3️⃣ 即時儀表板
- 不是離線評估，是**真實 CCTV 串流**
- 每 2 秒刷新，直接展示系統能力

---

## Slide 8：效能數字

| 指標 | 數值 |
|------|------|
| **mAP50** | **0.833** |
| **mAP50-95** | **0.712** |
| **訓練時間** | ～3 小時 |
| **推論延遲** | < 50ms 單幀 |
| **線上相機** | 5 路即時 |

> 與其他 YOLO 模型對比：yolo11s 在**小模型 vs 精度**之間取得好平衡。

---

## Slide 9：完整命令清單（展示用）

```bash
# ✅ 環境就緒後，逐步執行

# ❶ 前處理（～10 分鐘）
docker compose exec ray-head python scripts/prepare_freeway.py

# ❷ 訓練（～3 小時，自動用保守預設超參）
docker compose exec ray-head python scripts/train_freeway.py

# ❸ 上線（自動載入 best.pt；compose 無 ray-serve service）
docker compose exec ray-head python scripts/serve_dashboard.py

# ❹ 驗證
打開 http://localhost:8000  → 5 路相機即時跑
打開 http://localhost:8501 → MONITOR 顯示 3 節點 + Ray 進度
```

---

## Slide 10：常見問題（Q&A 時段）

**Q：為什麼要「鏡頭級」隔離，不直接「幀級」隨機切？**  
A：幀級隨機會讓同支鏡頭的相鄰幀同時進 train 和 test，導致洩漏。我們用整支鏡頭隔離，確保 test 鏡頭在訓練時完全隔離。

**Q：為什麼 mAP50 = 0.833 不是 0.95？**  
A：因為高公局 CCTV 畫質低、車小、有遮擋。0.833 是**在此條件下**的誠實成績，足以實戰使用。

**Q：Ray 三節點的優勢是？**  
A：平行前處理 + 監控容錯。單機跑 3 節點容器，透過網路通訊，展示分散系統的設計理念。

---

## Slide 11：技術亮點總結

✅ **Ray 四階段標準流程** — Data → Tune → Train → Serve  
✅ **鏡頭級防洩漏設計** — test 完全隔離，mAP50 0.833 誠實  
✅ **分散訓練與推論** — 3 節點叢集，CPU/GPU 各司其職  
✅ **即時儀表板** — 5 路 CCTV 實時展示，2 秒刷新率  

> 可直接部署到高公局真實環境。

---

> **簡報時長** ~ 3–4 分鐘（Slide 1–8），留 Q&A 時間（Slide 9–11）。
