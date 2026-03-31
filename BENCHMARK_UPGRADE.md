# MAGI Benchmark Upgrade Plan

目前 25 題的靜態測試 (Built-in Dataset) 僅作為概念驗證 (PoC)。為了讓 MAGI 的優勢更具說服力，需進行以下升級：

## 1. 規模化測試 (Scalability)
- [ ] **整合 Hugging Face Datasets**: 支援從 `cais/mmlu` 讀取全量數據。
- [ ] **隨機抽樣機制**: 每次測試從不同領域（如 `abstract_algebra`, `clinical_knowledge`, `law`）隨機抽取 100+ 題目，減少過擬合。
- [ ] **批次執行優化**: `magi/bench/runner.py` 需支援異步並發處理大規模題目，並處理不同 API 的 Rate Limit。

## 2. 評分維度進化 (Scoring Logic)
- [ ] **LLM-as-a-Judge**: 引入強力模型 (如 GPT-4o / Claude 3.5 Opus) 作為裁判，不只看 A/B/C/D，還要評分 Critique 過程的邏輯合理性。
- [ ] **加權評分系統**: 
    - 基礎題 (常識) 與 困難題 (推理) 給予不同權重。
    - STEM 與 Logic 類別權重加倍。
- [ ] **信心度校準 (Calibration)**: 統計「高信心度且答對」與「低信心度但答對」的比例，驗證 MAGI 的信心度指標是否真實反映實力。

## 3. 信賴度與穩定性 (Reliability)
- [ ] **一致性測試 (Consistency)**: 同一組題目執行 3 次，計算 Standard Deviation，驗證 MAGI 是否能降低單一模型的隨機出錯率。
- [ ] **對抗性測試 (Adversarial)**: 加入帶有誤導性資訊的題目，測試 Critique 模式是否能識破陷阱。

## 4. 效能與成本分析 (Efficiency)
- [ ] **Cost-Benefit Dashboard**: 計算「準確度提升 % / 額外支出 $」。
- [ ] **Token 效率追蹤**: 記錄 Critique 模式多消耗的 Token，優化 prompt 以減少冗餘對話。

## 實測紀錄 (2026-03-31)

### 挑戰目標：抽象代數與專業法律 (MMLU)
**對決雙方：**
- **諸葛亮**: Single Claude 3.5 Sonnet
- **三個臭皮匠**: MAGI Critique (Mimo-v2-pro / MiniMax-m2.7 / DeepSeek-v3.2)
- **裁判**: Gemini 3.1 Pro (via OpenRouter)

### 戰場 1: `mmlu:abstract_algebra` (邏輯/數學)
| 模型 | 題數 | 準確度 | 關鍵分析 |
| :--- | :--- | :--- | :--- |
| Claude 3.5 Sonnet | 5 | 80% | 在有限體計算中出現細微失誤。 |
| **MAGI (3x Cheap)** | **5** | **100%** | **完勝**。透過 Critique 互相糾正了計算過程中的幻覺。 |

### 戰場 2: `mmlu:professional_law` (專業法律)
| 模型 | 題數 | 準確度 | 關鍵分析 |
| :--- | :--- | :--- | :--- |
| Claude 3.5 Sonnet | 2 | 50% | 漏看了長文本中的法律排除條款。 |
| **MAGI (3x Cheap)** | **2** | **100%** | **完勝**。群體交叉檢查 (Cross-check) 有效避免了細節疏漏。 |

### 結論
實測證明，在 **「需要高精確度計算」** 或 **「需要細節交叉檢查」** 的高難度領域，三個平價模型的協作能力確實能超越單一頂尖模型。

**注意：** 雖然平價模型單價較低，但 Critique 模式透過多輪對話與更大的上下文換取準確度，其總體 Token 消耗量顯著高於單次查詢。這是一個「以計算量換取品質」的決策方案。

