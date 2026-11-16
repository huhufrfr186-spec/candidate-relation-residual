# Candidate Relation Residual Learning

传输约束候选关系残差学习，用于无人机可见光—红外旋转目标检测。

## 环境

- Python 3.11
- PyTorch 2.5.1、Torchvision 0.20.1
- CUDA 12.1
- NumPy 1.26.4、SciPy 1.14.1

```bash
conda env create -f environment.yml
conda activate crrl
```

CPU 环境：

```bash
python -m venv .venv
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

命令均在项目根目录执行。

## 方法

双路 ResNet18 提取步长为 8 的特征，通道投影后构建局部 RGB—红外二部图。RGB 查询按不重叠的 `2×2` 块组织，每个查询连接半径为 2 的红外候选窗口。

带未匹配节点的熵正则软传输以 `(1,…,1,m)` 和 `(1,…,1,n)` 为行列边际。在固定候选支持上，以 `w=η+(1−η)p` 构建投影度量；每个连通分量固定一个节点自由度，求解双精度加权正规方程，获得候选关系残差。

直接配对路径保留实例证据，关系路径通过 `β⊙tanh(R(z)−R(0))` 提供有界修正。预测采用两种配置：

- **连续输出**：在传输加权读出前修正分类 logits 和六维旋转框编码。
- **概率输出**：对类别条件的四维高斯分布与 π 周期 von Mises 角度分布进行混合；训练计算完整混合分布上的固定区域事件概率，推理对去重候选区域计算完整混合分数并执行旋转 NMS。

监督分配先以全局最小代价单射为每个目标保留唯一查询，再对未保留查询进行中心采样。RGB 辅助头与未匹配头共享参数，红外辅助头在自身模态坐标中监督。

## 数据格式

图像与标注由 JSON 清单配对。类别编号从 0 开始，框为 `[cx, cy, width, height, angle]`，中心和尺寸使用原始图像像素，角度使用弧度并按 π 周期表示。两个模态各自保存标注。

```json
{
  "classes": ["car", "truck", "bus", "van", "freight_car"],
  "samples": [
    {
      "id": "000001",
      "rgb": "images/rgb/000001.png",
      "ir": "images/ir/000001.png",
      "annotations": [{"category_id": 0, "bbox": [320, 256, 42, 18, 0.2]}],
      "ir_annotations": [{"category_id": 0, "bbox": [322, 255, 41, 18, 0.2]}]
    }
  ]
}
```

相对图像路径相对于清单所在目录解析。`scene` 字段可指定场景编号，场景分组划分保证同一场景不会进入不同集合。

### DroneVehicle

使用官方训练、验证和测试集合，类别顺序为 `car, truck, bus, van, freight_car`。原始 `840×712` 图像裁除四周 100 像素白边后为 `640×512`，标注中心同步平移。转换器接受 VOC XML、旋转框 XML 和 DOTA 四点标注。

```bash
python -m crrl prepare --rgb-dir data/raw/DroneVehicle/train/rgb --ir-dir data/raw/DroneVehicle/train/ir --rgb-labels data/raw/DroneVehicle/train/rgb_labels --ir-labels data/raw/DroneVehicle/train/ir_labels --classes car,truck,bus,van,freight_car --crop-border 100 --output data/dronevehicle/train.json
python -m crrl prepare --rgb-dir data/raw/DroneVehicle/val/rgb --ir-dir data/raw/DroneVehicle/val/ir --rgb-labels data/raw/DroneVehicle/val/rgb_labels --ir-labels data/raw/DroneVehicle/val/ir_labels --classes car,truck,bus,van,freight_car --crop-border 100 --output data/dronevehicle/val.json
python -m crrl prepare --rgb-dir data/raw/DroneVehicle/test/rgb --ir-dir data/raw/DroneVehicle/test/ir --rgb-labels data/raw/DroneVehicle/test/rgb_labels --ir-labels data/raw/DroneVehicle/test/ir_labels --classes car,truck,bus,van,freight_car --crop-border 100 --output data/dronevehicle/test.json
```

图像已经裁除白边时使用 `--crop-border 0`。`--ids` 可指定图像编号文本文件。

### LLVIP

```bash
python -m crrl prepare --rgb-dir data/raw/LLVIP/visible/train --ir-dir data/raw/LLVIP/infrared/train --rgb-labels data/raw/LLVIP/Annotations --ir-labels data/raw/LLVIP/Annotations --classes person --output data/llvip/all_train.json
python -m crrl split --manifest data/llvip/all_train.json --output data/llvip --seed 17 --validation-fraction 0.1
python -m crrl prepare --rgb-dir data/raw/LLVIP/visible/test --ir-dir data/raw/LLVIP/infrared/test --rgb-labels data/raw/LLVIP/Annotations --ir-labels data/raw/LLVIP/Annotations --classes person --output data/llvip/test.json
```

### VEDAI

```bash
python -m crrl prepare-vedai --images data/raw/VEDAI/Vehicules512 --annotations data/raw/VEDAI/Annotations512/annotation512.txt --fold-mat data/raw/VEDAI/Annotations512/fold.mat --output data/vedai
python tools/run_folds.py --config configs/vedai.yaml --manifests data/vedai --output runs/vedai
```

采用官方十折图像编号，每次一折测试，下一折验证，其余八折训练。

## 训练

```bash
python -m crrl train --config configs/m3_probabilistic.yaml --output runs/m3_probabilistic/seed_17 --seed 17
python -m crrl train --config configs/m3_continuous.yaml --output runs/m3_continuous/seed_17 --seed 17
```

默认输入为 `640×512`、联合通道数为 32、`η=0.05`。AdamW 训练 24 个 epoch，每次一个图像对、梯度累积 8 次；主干学习率 `1e-4`，其余模块学习率 `1e-3`。完整参数位于 `configs/base.yaml`。

ResNet18 使用 ImageNet-1K 初始化，首次启用预训练主干时由 Torchvision 自动下载权重。

概率头每条边输出 6 个类别 logits 和 50 个框分布参数。关系修正网络为 `32→64→56`，参数量 5,752；双侧图上下文修正网络为 `96→37→56`，参数量 5,717。

图像等比例缩放并填充，旋转框和角度保持几何一致。若目标数超过初始查询容量，训练前统一扩大查询网格；网格随 checkpoint 保存，并在推理时复用。

训练输出 `last.pt`、验证 AP 最佳的 `best.pt`、`config.yaml`、`metrics.jsonl` 和验证预测记录。

继续训练：

```bash
python -m crrl train --config configs/m3_probabilistic.yaml --output runs/m3_probabilistic/seed_17 --resume runs/m3_probabilistic/seed_17/last.pt
```

## 配置

| 配置 | 输出 | 上下文 |
|---|---|---|
| `m1_probabilistic.yaml` | 概率混合 | 直接路径 |
| `m2_probabilistic.yaml` | 概率混合 | 双侧图聚合 |
| `m3_probabilistic.yaml` | 概率混合 | 传输加权关系残差 |
| `m1_continuous.yaml` | 连续框 | 直接路径 |
| `m2_continuous.yaml` | 连续框 | 双侧图聚合 |
| `m3_continuous.yaml` | 连续框 | 传输加权关系残差 |
| `m3_refinement.yaml` | 概率混合与区域内优化 | 传输加权关系残差 |
| `demeaning.yaml` | 概率混合 | 双侧去均值 |
| `equal_weight.yaml` | 概率混合 | 等权投影 |
| `stop_transport_gradient.yaml` | 概率混合 | 投影权重停止传输梯度 |

## 推理与评估

```bash
python -m crrl predict --checkpoint runs/m3_probabilistic/seed_17/best.pt --manifest data/dronevehicle/test.json --output runs/m3_probabilistic/seed_17/predictions.json
python -m crrl evaluate --checkpoint runs/m3_probabilistic/seed_17/best.pt --manifest data/dronevehicle/test.json --output runs/m3_probabilistic/seed_17/test
```

概率输出默认使用中心 2 像素、对数尺度 0.05、角度 2° 的固定区域。非角度维度使用均值所在区间及两侧相邻区间；集中角度采用相邻三个圆周区间，近均匀角度枚举完整周期。所有有效分量与未匹配分量共同产生候选并计算区域分数，区域内优化保持原区域质量作为分数。

评估基于旋转多边形 IoU、101 个召回点插值和 0.50–0.95 的十个 IoU 阈值，输出总体及各类别 AP50、AP75、AP50:95。

导出四点旋转框结果：

```bash
python -m crrl export --predictions runs/m3_probabilistic/seed_17/test/predictions.json --classes car,truck,bus,van,freight_car --output runs/m3_probabilistic/seed_17/dota
```

## 五次重复实验

```bash
python tools/run_seeds.py --config configs/m3_probabilistic.yaml --output runs/m3_probabilistic --seeds 17 29 43 61 83
python tools/run_seeds.py --config configs/m3_continuous.yaml --output runs/m3_continuous --seeds 17 29 43 61 83
```

统计命令接收各次运行的 `metrics.json`，使用样本标准差及配对 t 区间：

```bash
python -m crrl summarize --metrics runs/m3_probabilistic/seed_17/test/metrics.json runs/m3_probabilistic/seed_29/test/metrics.json runs/m3_probabilistic/seed_43/test/metrics.json runs/m3_probabilistic/seed_61/test/metrics.json runs/m3_probabilistic/seed_83/test/metrics.json --output runs/m3_probabilistic/summary.json
```

`--paired-metrics` 按相同种子顺序接收对照结果，输出逐次配对差值的均值、样本标准差和 95% t 区间。

## 残差干预与诊断

```bash
python -m crrl evaluate --checkpoint runs/m3_continuous/seed_17/best.pt --manifest data/dronevehicle/test.json --intervention zero --output runs/intervention_zero
python -m crrl evaluate --checkpoint runs/m3_continuous/seed_17/best.pt --manifest data/dronevehicle/test.json --intervention shuffle --seed 17 --output runs/intervention_shuffle
python -m crrl diagnostics --manifest data/dronevehicle/test.json --predictions runs/m3_continuous/seed_17/test/predictions.json --threshold 0.01 --output runs/diagnostics.json
```

干预仅作用于图内残差，保持训练权重、直接路径和传输质量固定。诊断记录几何邻域竞争分组、目标面积，以及固定阈值下的 TP、FP、FN、Precision、Recall 和 F1。

## 计时

```bash
python -m crrl benchmark --checkpoint runs/m3_probabilistic/seed_17/best.pt --manifest data/dronevehicle/test.json --warmup 100 --iterations 500 --output runs/timing
```

计时包含模型、候选区域解码和旋转 NMS，不包含文件读取；CUDA 计时显式同步。输出逐次记录、均值、样本标准差和吞吐率。

## 快速运行

```bash
python tools/make_demo_data.py
python -m crrl train --config configs/smoke.yaml --output runs/smoke
python -m crrl predict --checkpoint runs/smoke/last.pt --manifest data/smoke/train.json --output runs/smoke/predictions.json --device cpu
python -m unittest discover -s tests -v
```

## 代码结构

```text
crrl/
  graph.py          局部图、带未匹配节点的传输、加权投影
  model.py          双路主干、配对编码和有界关系修正
  assignment.py     唯一查询保留与中心采样
  probability.py    稳定区间概率与类别条件分布
  losses.py         连续输出与区域混合监督
  decode.py         完整混合评分、区域内优化和解码
  geometry.py       旋转框几何、IoU 和 NMS
  data.py           成对图像与双模态坐标变换
  prepare.py        标注转换、场景划分和 VEDAI 十折
  engine.py         训练、checkpoint、评估和计时
  metrics.py        旋转框 AP
  diagnostics.py    几何竞争和固定工作点统计
  cli.py            命令行入口
configs/            训练与消融配置
tools/              多种子、十折和快速运行脚本
tests/              算子、梯度、分配和模型测试
```

## 数据来源

- [DroneVehicle](https://github.com/VisDrone/DroneVehicle)
- [LLVIP](https://github.com/bupt-ai-cz/LLVIP)
- [VEDAI](https://downloads.greyc.fr/vedai/)
