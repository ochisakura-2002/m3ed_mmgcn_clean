# B0 类别几何零训练诊断

2026-11-23 · `B0_DIAGNOSTIC = INCONCLUSIVE` · Test HOLD

本地归档包含五个最终确认 F0 的 980 条／seed Dev 预测和 checkpoint SHA，权重文件在归档时排除。本地已用历史 bundle 的文件 SHA 校验五份预测，计算真实标签相对其他类的 logit margin。六个预注册诊断方向如下，误判率和 margin 对五 seed 等权。逐 seed CSV、汇总 CSV 和原始 logits/margins NPZ 位于 `tmp/post_fusion_fastkill_001_local_diagnostics_20261123/b0_archived_dev_logits/`。

| 真实→比较类别 | 每 seed 支持数 | 错误率 %（均值 ± 样本 SD） | 真实类 logit 减比较类 logit |
| --- | ---: | ---: | ---: |
| hap→exc | 60 | 38.00 ± 8.77 | −0.028 |
| exc→hap | 122 | 9.67 ± 6.69 | 3.160 |
| neu→fru | 157 | 27.13 ± 2.94 | 1.240 |
| fru→neu | 319 | 11.41 ± 2.76 | 3.658 |
| ang→fru | 222 | 24.32 ± 4.09 | 1.508 |
| fru→ang | 319 | 9.66 ± 2.06 | 2.879 |

完整诊断在有原始五个 F0 的 Ubuntu 环境中运行 `--b0`。校验原配置、冻结种子、选点日志、历史 Dev evidence、数据及 checkpoint SHA，并严格加载最佳权重。冻结模型在 Train/Dev 各跑一次 eval，捕获第二个 ReLU 输出。两 split 各算六类样本均值 `mu_c`，`D(c,d)=1-cos(mu_c,mu_d)`，`I(c)=mean(1-cos(z_i,mu_c))`，`SeparationRatio=D/mean(I(c),I(d))`。分母为零或类缺失记空值。

以 Train centroid 作为两个 split 的 nearest-centroid 参考。逐样本类间 margin 用 true-class cosine 减最大非真类 cosine，classifier margin 用对应原始 logits。类对 margin 在真类各自的样本上单独按方向计算。导出所有类、30 个方向、pair separation rank、分类器混淆与 separation 的相关性、nearest-centroid 混淆和原始 300D 向量。跨 seed 汇总逐类及逐方向标量的 mean / sample SD，保留独立 seed 的原始 centroid 坐标。B0 机制结论取决于表现空间的实测。 [运行命令](PARALLEL_FASTKILL_RUNBOOK.md)
