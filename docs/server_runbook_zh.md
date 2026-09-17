# 通用服务器运行说明

本说明不绑定具体机构、账户或服务器路径。以下命令均应在已克隆仓库的根目录执行。

## CPU 环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r environments/requirements-linux-cpu.lock.txt
python -m pip install --no-deps .
```

## CUDA 12.1 环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r environments/requirements-linux-cuda121.lock.txt
python -m pip install --no-deps .
```

## 严格检查

```bash
face-preprocess --json doctor \
  --config configs/pipeline_server.yaml \
  --device-profile configs/device_scanner_1p2mm.yaml \
  --qc-config configs/qc_exploratory.yaml \
  --device cpu
```

将 `--device cpu` 改成 `--device cuda` 可在已正确分配的 NVIDIA GPU 上运行。CUDA 模式要求 `--workers 1`。

## 单样本验收

```bash
bash scripts/verify_server.sh \
  /path/to/sample.obj \
  /path/to/acceptance_output \
  face-preprocess \
  cpu
```

不要把真实样本、清单或输出目录提交到 Git。
