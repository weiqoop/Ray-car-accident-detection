# 整套框架釐清

> 本檔說明系統各部分如何組合。完整操作見 [README.md](README.md)，評估與 Demo 見
> [report.md](report.md)，演進與除錯見 [NOTE.md](NOTE.md)。

## 一、兩個模型、兩條流程

| 流程 | 模型 | 型態 | 資料 | 訓練方式 | 角色 |
|------|------|------|------|----------|------|
| **車流（Freeway）** | YOLO11s | 物件偵測 | 高公局 CCTV（yolo11x+SAHI 自動標）| ultralytics + Ray 四階段 | 即時車流/密度 |
| **車禍（Accident）** | MobileNetV2 | 逐幀二分類 | **TAD**（高速公路監控真實事故）| Ray 四階段 | 事故事件判定 |

兩條流程**各自獨立**完整走 Ray 四階段,在 Serve 階段**共用同一個 replica**呈現:
- 車流：背景輪詢 5 路相機 → YOLO 偵測 → 畫框 + 密度分級
- 車禍：右下角面板輪播 TAD 測試影片 → CNN 逐幀判定

```
高公局 5 路 CCTV ─YOLO11s─► 車數/密度 ┐
                                       ├─► Ray Serve :8000 儀表板
TAD 測試影片 ─MobileNetV2─► 事故/正常 ┘
```

## 二、車禍：同域資料 + 影片級切分（數字可信的核心）

- **同域**：TAD 是 freeway 監控視角(720×300 俯視),與高公局同類 → 跨域風險最低。
- **影片級切分**：以「影片」為單位切 train/val/test,**同一支影片的幀不跨 split**。
  若隨機切「幀」,同片相鄰幀會同時落在 train 與 test → 洩漏灌水(曾在圖片集得 99% 假象)。
- **粗標籤限制**：TAD 整片標同一類(無逐幀事故時刻)→ 模型判「該片是否事故」(影片級
  ROC 0.96)準,但**無法精準定位撞擊瞬間**(已知限制)。

## 三、資料來源（現行）

```
datasets/
├── Traffic Anomaly Dataset/TAD/frames/      ← TAD（在用,serve 也直接讀此做 demo）
│   ├── normal/Normal_*.mp4/                    250 片正常（負類）
│   └── abnormal/01_Accident_*.mp4/             110 片車禍（正類；其餘 5 類異常排除）
├── DETRAC-Images/                            ← 車流偵測訓練源
├── weights/{yolo11x,mobilenet_v2,resnet18}   ← 離線預訓練權重
├── freeway_*                                  ← 高公局車流資料/標註
└── accident_tad_seq/                          ← ① Ray Data 產出：train/val/test.npz(含 vid)

_archive/                                      ← 已封存（不再使用）
├── 軌跡法（AccidentBench）程式 + accident_seq
├── 圖片集 CNN（Road Accidents）資料
└── 各舊 ray_results run
```

> 資料集評選歷程(軌跡時序 → 圖片 CNN → TAD)見 [report.md](report.md) §2.3 與 NOTE.md。

## 四、Ray 四階段對應

| 階段 | 車流（Freeway） | 車禍（Accident，TAD） | 用到哪些節點 |
|------|------|------|------|
| ① Data | 鏡頭級切分、CPU 前處理 | **影片級切分** + 解碼縮放 | **CPU 多節點**（worker 出力）|
| ② Tune | ultralytics ASHA（GPU） | ASHA 搜 CNN 超參（GPU；資料 `ray.put` 共享）| head(GPU) |
| ③ Train | TorchTrainer + yolo11s（GPU） | TorchTrainer 微調 MobileNetV2（GPU）| head(GPU) |
| ④ Serve | 5 路相機儀表板（:8000） | 同 replica：TAD 影片逐幀判定面板 | head(GPU) |

> **節點配置邏輯**：GPU 只有 head 一顆,GPU 綁定工作(偵測/訓練/推論)只在 head;
> 兩個 CPU-only worker 在 **Ray Data 階段**(影像解碼/前處理)吃滿 16 核。這是「三節點」名副其實處。

## 五、物件存儲（Object Store）運用

- **車禍 CNN**：訓練資料 driver 端載一次 → `ray.put` 進 object store →
  `tune.with_parameters` 讓**各 Tune trial 零拷貝共享**,避免每 trial 各自載 2.3GB 爆記憶體。
- **YOLO（車流）**：訓練資料由 ultralytics 自管(從磁碟讀),Ray object store 在訓練被繞過
  (這是 YOLO-on-Ray 的本質,且 YOLO 惰性讀圖本來就無此記憶體痛點);但 **Ray Data 批次推論**
  (prelabel)有走 object store。

## 六、關鍵設計取捨

| 議題 | 決定 | 理由 |
|------|------|------|
| 事故資料 | TAD（freeway 監控真實事故）| 同域 + 影片級可切 + 真實事故,跨域最可信 |
| 切分 | **影片級**(同片不跨 split) | 防同片相鄰幀洩漏(圖片集隨機切幀曾灌出 99% 假象)|
| 模型 | MobileNetV2 解凍微調 | 同域圖,微調最強;ImageNet 權重離線載入 |
| 不平衡 | pos_weight + **F1** 選優 | 事故/正常 ~110:250,不看 accuracy |
| 評估 | **影片級**為主指標 | 對齊 TAD 影片級標籤,回答「能否抓到事故片」|
| 前端偵測 | 訓練 yolo11x / 部署 yolo11s | 訓練要強跨域偵測;部署用高公局自身微調模型 |
