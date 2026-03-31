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

## 5. 視覺化報告 (Reporting)
- [ ] **自動生成分析圖表**: 輸出各類別的準確度對比圖 (Radar Chart)。
- [ ] **決策回放 (Decision Replay)**: 點擊錯誤題目，可直接查看三台模型在 Critique 過程中的對話細節。
