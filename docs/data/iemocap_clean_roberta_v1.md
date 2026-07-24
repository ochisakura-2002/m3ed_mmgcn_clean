# IEMOCAP Clean RoBERTa v1 特征契约

## 目的

MMGCN 官方 PKL 中的 100 维 TextCNN 表示用于旧版复现兼容，但其训练来源与验证边界不足以支撑新增的 Session-Holdout Validation：无法从现有产物证明特征提取阶段没有利用 held-out session 或 emotion label。它因此保留为 `legacy_compatibility_only`，不作为新 Session-Holdout 结论的文本表示。

Clean v1 只替换九项 PKL 的 `videoText`（索引 3）。`videoIDs`、speaker、label、audio、visual、sentence、`trainVid` 和 `testVid` 必须通过逐项审计，audio/visual 的 shape、dtype 和数值必须不变。旧官方 PKL 绝不覆盖。

## 固定协议

- feature set：`iemocap_clean_roberta_base_utterance_mean_v1`
- PKL：`data/processed/iemocap/IEMOCAP_features_clean_roberta_base_c8b8a37_utterance_mean_v1.pkl`
- SHA256：`c604c557bc3fbb129ca02b2acd57166b669a89ef76ff0cea1e14f9cb206324bf`
- model：`FacebookAI/roberta-base`
- revision：`c8b8a37ce3afa8b16a98ff5d0016c157a16ef432`
- 输入边界：每个 utterance 独立编码，不拼接历史或未来
- pooling：排除 padding 和全部 special token 后做 mean pooling
- 输出：float32、768 维
- audio/visual 维度：1582 / 342
- 因果范围：`utterance_only_no_dialogue_context`
- label 使用：`labels_not_used_for_feature_extraction`
- 模型只从命令行给定的本地目录加载，`local_files_only=True`，不微调
- 审计状态：严格审计 `PASS`
- 冻结状态：`frozen_for_main_experiments`，immutable v1

## 生成、审计与登记

远程机器准备好本地 RoBERTa 权重后运行构建脚本，并显式传入旧 PKL 的固定 SHA256。脚本输出新 PKL、metadata JSON 和 SHA256 文件；不要把 PKL 或模型权重提交到 Git。

```bash
python scripts/data/build/build_iemocap_clean_text_features.py \
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
python scripts/diagnostics/data/audit_iemocap_feature_pkl.py \
  --legacy-pkl third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl \
  --candidate-pkl data/processed/iemocap/IEMOCAP_features_clean_roberta_base_c8b8a37_utterance_mean_v1.pkl \
  --expected-legacy-sha256 ceb5cc9a45d1792998438e27f86a12642320aa6315546e4425316140ea503fb3 \
  --expected-text-dim 768 \
  --output-dir <AUDIT_OUTPUT_DIR> \
  --strict
```

Clean v1 已使用 `scripts/diagnostics/data/audit_iemocap_feature_pkl.py --strict` 对照旧 PKL，严格审计结果为 `PASS`。审计确认只有 `videoText` 被替换，非文本字段以及 audio/visual 的 shape、dtype 和数值保持不变。

最终产物 SHA256 为：

```text
c604c557bc3fbb129ca02b2acd57166b669a89ef76ff0cea1e14f9cb206324bf
```

该 SHA256 已同时固定到 feature registry 和四个 clean v1 smoke 配置。smoke 与正式训练使用相同的完整文件哈希校验，不再允许 `TO_BE_FILLED_AFTER_GENERATION` 或 `allow_unpinned_feature_for_smoke` 绕过校验。实际 PKL SHA 不匹配时，训练入口必须在模型初始化前失败。当前仓库不预生成正式 16-run 配置。

## Session-holdout 线性探针

对 Clean v1 文本特征执行五个 held-out Session 的线性探针，Weighted-F1（WF1）如下：

| Held-out Session | WF1 |
| --- | ---: |
| Ses01 | 0.415652 |
| Ses02 | 0.507076 |
| Ses03 | 0.440045 |
| Ses04 | 0.462018 |
| Ses05 | 0.453176 |

旧特征在 Ses01–Ses04 出现的异常高分已经消失；Clean v1 的五折结果回到一致且更可信的范围。该探针只用于检查文本表示和 session 边界，不替代四个主模型的正式实验结果。

## 冻结规则

Clean v1 已通过严格审计，自此冻结到本课题结束。构建脚本默认拒绝覆盖任何既有输出；不得替换、重写或原地修补 v1。任何 tokenizer、revision、pooling、截断、空句策略、上下文边界或实现语义的变化都必须创建独立的 v2 路径、feature-set 名称、metadata 和 SHA256，并重新审计。
