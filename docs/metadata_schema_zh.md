# metadata（pipeline 1.1）中文说明

每个样本的当前记录位于 `metadata/samples/<output_key>.json`，同一次尝试的不可变副本位于 `metadata/attempts/<output_key>/<attempt_id>.json`。final OBJ 和中间文件先写入同一文件系统的 `.staging`，移动完成后才写 metadata，因此存在 `commit_state: complete` 才表示事务已提交。

## 身份与溯源

- `run_id`：每个新运行目录的 UUIDv4。
- `attempt_id`：每次样本尝试的 UUIDv4。
- `output_key`：安全文件名；无冲突时取输入 stem，有冲突时附确定性 UUIDv5。
- `sample_id`：清单或文件名提供的显示标签，不参与采样和身份判断。
- `relative_path`、`source_path`：输入位置。
- `source_file_sha256`：原始字节哈希。
- `geometry_sha256`：按原顶点/面顺序计算的规范几何哈希。
- `subject_group`、`batch_id`、`user_metadata`：清单附加字段。
- `obj_inventory`：输入中的 vt、vn、材质、对象/组、平滑和未知指令计数；v1 不传播纹理。

## 配置与可重复性

- `profiles`：pipeline、Crop model、Landmark model、device 和 QC 的 id 与 version。
- `adapters.crop`、`adapters.landmark`：adapter 名称/版本/源码 SHA256、precision，以及每个具名 asset 的 SHA256 和字节数。
- `fingerprints.input_set`：整个输入集合的路径、文件哈希和几何哈希。
- `fingerprints.scientific`：五份配置哈希、两个 adapter 的名称/版本/源码 SHA256、QC 档位、对称化选择以及完整推理/几何源码树哈希。
- `fingerprints.execution`：工具版本、CPU/CUDA、worker 数、输出模式、Python/PyTorch/CUDA 与关键依赖版本。
- `stage_seeds.crop`、`stage_seeds.landmark`：从几何哈希、阶段槽位和模型 base seed 派生的实际 seed。

所有指纹完全相同，且 `commit_state=complete`，`--resume` 才允许跳过。

## 科学结果

- `landmarks`：全部 9 个 snapped landmark，固定顺序语义为 `bijian, bigen, bixia, wyjzuo, nyjzuo, nyjyou, wyjyou, kouzuo, kouyou`。
- `bijian`：上述 `landmarks.bijian` 的重复便利字段，仅为兼容和查阅方便；不再单独驱动粗配准。
- `coarse_alignment`：9 点 snapped landmark 到固定 9 点参考坐标的等权相似 GPA 记录，包括方法名、固定顺序、是否允许尺度/反射、参考资产 SHA256、4×4 transform、尺度、行列式和 9 点 RMS。RMS 只作审计记录，没有未经校准的自动阈值。
- `registration.rigid_scale`、`registration.trajectory`：MeshMonk 刚性尺度与抽样迭代轨迹。
- `gpa.transform`：4×4 变换矩阵及尺度、行列式、质心指标。
- `symmetry`：未启用时为 null；启用时记录平均/p95/最大位移。
- `enhanced_diagnostics`：standard 为 null；enhanced 记录 D1–D4 seed、旋转和稳定性结果。

## QC

`measurements` 是以完整指标名为键的映射，每项包含：

```json
{
  "value": 1.2,
  "unit": "mm",
  "compute_status": "computed",
  "error": null
}
```

数组可额外包含 `shape` 和 `storage`。常见 `compute_status` 为 `computed`、`not_applicable`、`not_available` 或 `computation_error`。adapter 未提供可选诊断时使用 `not_available`，这与核心输出无效不同。

`qc` 包含：

- `status`：passed/warning/failed；
- `qc_incomplete`：规则所需指标缺失、单位不匹配或计算失败；
- `hard_failures`：不可关闭的技术契约；
- `rules`：逐条规则的 observed、passed、triggered、severity 和原因。

没有总质量分数。经验指标保留原值和逐条判断，避免把异质问题压缩成一个不可解释的分数。

## 输出状态

- `final_obj`：相对运行目录的文件路径；failed 时为 null。
- `status`：passed、warning 或 failed。
- `error`：阶段异常导致 failed 时保存异常类型和信息。
- `output_mode`：compact/full。
- `diagnostic_arrays`：full 模式下指向压缩 NPZ。`landmark_snapped` 和注册数组为核心记录；概率、mask、`continuous` landmark、吸附距离/索引及 heatmap/top-k 等仅在 adapter 提供时存在。compact 为 null。
- `qc_level`：standard/enhanced。
- `committed_at`：UTC ISO-8601 时间。
