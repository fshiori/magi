# TODOS

## Phase 2

### MAGI-as-API-Gateway
OpenAI-compatible HTTP proxy server。任何現有應用只要改 `base_url` 指向 MAGI，
零修改就能享受多模型決策品質。

**Why:** 從「開發者 CLI 工具」變成「基礎設施」，市場定位完全不同。
**Effort:** M (human: ~2 weeks / CC: ~2 hours)
**Priority:** P2
**Depends on:** 核心引擎完成
**Context:** 使用者在 /plan-ceo-review 提出。LiteLLM 已有類似的 proxy 功能可參考。

### NERV 指揮室 Web UI
即時視覺化介面，看到三台 MAGI 的決策過程、投票結果、信心度儀表板。

**Why:** EVA 粉絲的殺手功能，讓研究結果更直觀。
**Effort:** L (human: ~1 month / CC: ~4 hours)
**Priority:** P2
**Depends on:** 核心引擎 + API Gateway
**Context:** /office-hours 10x vision 的核心部分。

### Scorecard 加權投票
根據各節點歷史表現自動調整投票權重。

**Why:** 理論上能提升決策品質，但需要足夠數據驗證。
**Effort:** S (CC: ~30 min)
**Priority:** P3
**Depends on:** 累積足夠 trace 數據（建議 >100 個有 ground truth 的決策）
**Context:** Codex outside voice 認為是「假精度」，建議等數據充分再啟用。

## Phase 3 (Benchmark & Analytics)

### 評分系統升級
詳見 [BENCHMARK_UPGRADE.md](./BENCHMARK_UPGRADE.md)。
- [ ] 整合 MMLU 全量數據。
- [ ] 引入 LLM-as-a-Judge。
- [ ] 成本/準確度效益分析。
