# QC 探索、校准与复评

## 三层规则

流水线把质量信息分成三类：

1. 12 组确定性技术契约：OBJ 可读、三角面、有限/非空几何、资产完整、裁剪非空、9 点完整、9 点有限、9 点确为裁剪网格顶点、9 点粗配准可解、模板拓扑和最终几何。它们不可由 QC YAML 关闭；违反时样本 failed 或运行 fail-fast，且不会回退鼻尖平移。
2. 24 组经验性指标：输入 7 组、裁剪 4 组、landmark 6 组、注册 3 组、GPA 1 组、最终几何 1 组、增强稳定性 2 组。阈值完全在外部 QC YAML 中。
3. 4 组遥测：阶段耗时、峰值内存、设备、I/O。用于运行审计，不参与科学失败判断。

增强档的 `crop.stability` 和 `landmark.stability` 只能是 `record_only` 或 `warning`，永远不能造成 failed。

## 第一步：采集

使用 `enhanced + qc_exploratory.yaml` 跑代表性样本。探索配置的 24 组全部是 `record_only`，因此先看数据分布，不预设阈值。

```bash
face-preprocess run ... --qc-level enhanced \
  --qc-config configs/qc_exploratory.yaml
```

可对不同设备、批次或人群分别建立 run，之后合并摘要：

```bash
face-preprocess qc-summarize \
  --run-dir outputs/device_a \
  --run-dir outputs/device_b \
  --output qc/combined_summary
```

## 第二步：生成候选

在 validation plan 中设置实际样本量。`sample_count_target` 可为任意正整数；超过现有样本数时使用全部可用样本。

```bash
face-preprocess qc-calibrate \
  --run-dir outputs/device_a \
  --qc-config configs/qc_production_template.yaml \
  --validation-plan configs/validation_plan.example.yaml \
  --output qc/calibration_a
```

如果已有人工共识标签，可追加 `--labels review_labels.csv`。CSV 使用
`output_key`（或 `sample_id`）以及 `consensus_label`（或 `label`）列；
`acceptable/unacceptable` 会生成逐样本预测表、不可接受样本检出率和可接受
样本误警率，`suspicious/unreviewable` 保留给人工流程但不进入这两个二分类指标。

输出：

- `metric_distributions.csv`：n、min、p01、p05、median、p95、p99、max；
- `candidate_qc.yaml`：输入模板的独立副本；
- `calibration.json`：样本量与产物记录。
- 可选 `labeled_predictions.csv` 与 `labeled_performance.json`：人工标签性能。

工具不会根据分位数自动写 warning 阈值，因为这需要临床/技术语义与误报成本判断。应人工编辑 candidate，修改 `profile_id/profile_version` 并纳入版本管理。

## 规则格式

```yaml
metric_groups:
  crop.retention: warning
rules:
  - rule_id: crop_vertex_fraction
    metric: crop.retention.vertex_fraction
    operator: outside
    threshold: [0.35, 0.90]
    unit: ratio
    severity: warning
    minimum_qc_level: standard
```

运算符只允许 `gt/gte/lt/lte/inside/outside/equals`，不执行字符串表达式或 `eval`。运算符表达“触发条件”；条件满足时触发规则。配置中的 unit 必须和 measurement unit 完全一致。

组模式：

- `disabled`：不评估该经验组的规则；
- `record_only`：记录规则结果但不改变样本状态；
- `warning`：触发时状态为 warning，仍保留 final OBJ。

## 第三步：独立复评

```bash
face-preprocess qc-evaluate \
  --run-dir outputs/device_a \
  --qc-config qc/qc_v2.yaml \
  --qc-level standard \
  --output qc/eval_device_a_v2
```

同一个主程序可以对任意多个数据集、任意多个 QC 版本反复运行。每次指定新的空输出目录；旧阈值、原样本 metadata 和 final OBJ 都不会被覆盖。若规则需要的测量值缺失、计算失败或单位不符，结果标记 `qc_incomplete` 并至少 warning，不会假装通过。

## 建议冻结流程

- 在代表性样本上确定阈值和方向；
- 在独立验证集上评估 warning 命中率、漏检率和人工复核一致性；
- 将冻结 QC YAML 的 SHA256、profile id/version 与验证报告一起归档；
- 新设备或扫描协议使用新的 device profile 和 QC profile，不覆盖旧版本。
