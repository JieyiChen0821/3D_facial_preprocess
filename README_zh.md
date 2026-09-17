# 3D 人脸自动预处理流水线

这是一个只用于推理、可审计的 3D 人脸 OBJ 预处理仓库。它把原始三角网格映射到固定的 7,906 点模板，并保留每一步的配置、资产哈希和样本级运行记录。

## 流程

```text
原始 OBJ（毫米）
  -> PointNeXt 自动剪裁
  -> 九点 Landmark 自动标定
  -> 九点相似 GPA 粗配准
  -> MeshMonk mapping
  -> 7,906 点 GPA
  -> 可选对称化
  -> 可选表型 Landmark 映射
```

表型 Landmark 默认关闭。启用对称化时，它读取对称化后的最终网格；未启用时，它读取第二次 GPA 的结果。

## 隐私边界

仓库仅包含推理代码、两个选定的模型权重、固定模板资产、配置、测试和说明文档，不包含原始扫描、训练/验证数据、样本清单、个体预测结果、受试者标签或真实运行输出。

裁剪 checkpoint 已转成最小推理版本，仅保留 `model_state` 和重建网络所需的五个架构参数。训练路径、优化器、学习率调度器、训练指标和运行目录均已移除。发布前可执行：

```bash
python scripts/audit_public_release.py --root . --json
```

## 安装

推荐 Python 3.10。Linux CPU 环境：

```bash
python -m pip install -r environments/requirements-linux-cpu.lock.txt
python -m pip install --no-deps .
```

CUDA 12.1 和 Windows 的锁定依赖文件位于 `environments/`。

## 环境检查

```bash
face-preprocess --json doctor \
  --config configs/pipeline_server.yaml \
  --device-profile configs/device_scanner_1p2mm.yaml \
  --qc-config configs/qc_exploratory.yaml \
  --device cpu
```

## 完整运行

```bash
face-preprocess --json run \
  --input /path/to/raw.obj \
  --output /path/to/run_output \
  --config configs/pipeline_server.yaml \
  --device-profile configs/device_scanner_1p2mm.yaml \
  --qc-config configs/qc_exploratory.yaml \
  --device cpu \
  --workers 1 \
  --qc-level standard \
  --output-mode full \
  --symmetry \
  --phenotype-landmarks
```

`--input` 可接受单个 OBJ 或平铺 OBJ 目录。`--symmetry` 与 `--phenotype-landmarks` 都是可选开关。

## 对已有标准化 OBJ 补算表型 Landmark

```bash
face-preprocess --json phenotype-landmarks \
  --input /path/to/final_obj \
  --output /path/to/phenotype_output \
  --config configs/pipeline_server.yaml
```

默认生成 `<stem>_landmarks.csv` 和 `<stem>_landmarks.pp`；使用 `--no-pp` 可只生成 CSV。输入必须与固定模板的三角面顺序完全一致。

主要输出和科学资产契约见 [输出说明](docs/output_contract.md) 与 [模型资产说明](docs/model_assets.md)。

## 测试

```bash
python -m pytest -q
python scripts/audit_public_release.py --root . --json
```

## 许可

本仓库尚未指定软件许可证；公开可见不等同于自动授予复用权利。
