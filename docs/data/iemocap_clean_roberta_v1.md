# IEMOCAP Clean RoBERTa v1 特征契约

## 目的

MMGCN 官方 PKL 中的 100 维 TextCNN 表示用于旧版复现兼容，但其训练来源与验证边界不足以支撑新增的 Session-Holdout Validation：无法从现有产物证明特征提取阶段没有利用 held-out session 或 emotion label。它因此保留为 `legacy_compatibility_only`，不作为新 Session-Holdout 结论的文本表示。

Clean v1 只替换九项 PKL 的 `videoText`（索引 3）。`videoIDs`、speaker、label、audio、visual、sentence、`trainVid` 和 `testVid` 必须通过逐项审计，audio/visual 的 shape、dtype 和数值必须不变。旧官方 PKL 绝不覆盖。

## 固定协议

- feature set：`iemocap_clean_roberta_base_utterance_mean_v1`
- model：`FacebookAI/roberta-base`
- revision：`c8b8a37ce3afa8b16a98ff5d0016c157a16ef432`
- 输入边界：每个 utterance 独立编码，不拼接历史或未来
- pooling：排除 padding 和全部 special token 后做 mean pooling
- 输出：float32、768 维
- 因果范围：`utterance_only_no_dialogue_context`
- label 使用：`labels_not_used_for_feature_extraction`
- 模型只从命令行给定的本地目录加载，`local_files_only=True`，不微调

## 生成、审计与登记

远程机器准备好本地 RoBERTa 权重后运行构建脚本，并显式传入旧 PKL 的固定 SHA256。脚本输出新 PKL、metadata JSON 和 SHA256 文件；不要把 PKL 或模型权重提交到 Git。

```bash
python scripts/features/build_iemocap_clean_text_features.py \
  --input-pkl third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl \
  --expected-input-sha256 ceb5cc9a45d1792998438e27f86a12642320aa6315546e4425316140ea503fb3 \
  --model-dir <LOCAL_ROBERTA_BASE_DIR> \
  --model-id FacebookAI/roberta-base \
  --model-revision c8b8a37ce3afa8b16a98ff5d0016c157a16ef432 \
  --output-pkl data/processed/iemocap/IEMOCAP_features_clean_roberta_base_c8b8a37_utterance_mean_v1.pkl \
  --pooling mean --batch-size 32 --max-length 512 --device cuda --seed 42 \
  --metadata-output data/processed/iemocap/IEMOCAP_features_clean_roberta_base_c8b8a37_utterance_mean_v1.metadata.json \
  --sha256-output data/processed/iemocap/IEMOCAP_features_clean_roberta_base_c8b8a37_utterance_mean_v1.sha256 \
  --fail-if-output-exists
```

```bash
python scripts/analyze/audit_iemocap_feature_pkl.py \
  --legacy-pkl third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl \
  --candidate-pkl data/processed/iemocap/IEMOCAP_features_clean_roberta_base_c8b8a37_utterance_mean_v1.pkl \
  --expected-legacy-sha256 ceb5cc9a45d1792998438e27f86a12642320aa6315546e4425316140ea503fb3 \
  --expected-text-dim 768 \
  --output-dir <AUDIT_OUTPUT_DIR> \
  --strict
```

随后用 `scripts/analyze/audit_iemocap_feature_pkl.py --strict` 对照旧 PKL。审计通过后：

1. 把生成的 SHA256 同时写入 `configs/data/iemocap_feature_sets.yaml` 的 `clean_roberta_v1.sha256`。
2. 把同一 SHA256 写入后续 clean-feature 正式训练配置的 `dataset.feature_sha256`。
3. 把 registry status 从 `pending_generation` 更新为 `frozen_audited`，并保留审计报告。
4. 先去掉 smoke 配置中的 unpinned 放行，再生成后续正式实验配置；当前仓库不预生成 16-run 配置。

只要 `feature_sha256` 还是 `TO_BE_FILLED_AFTER_GENERATION`，正式训练入口就会拒绝执行。只有明确写有 `allow_unpinned_feature_for_smoke: true` 的 smoke 配置可以绕过 pin 校验，而且真实运行仍要求目标 PKL 已存在。

## 冻结规则

v1 一旦通过审计即冻结到本课题结束。构建脚本默认拒绝覆盖任何既有输出；不得替换、重写或原地修补 v1。任何 tokenizer、revision、pooling、截断、空句策略、上下文边界或实现语义的未来变化都必须生成独立的 v2 路径、feature-set 名称、metadata 和 SHA256，并重新审计。
