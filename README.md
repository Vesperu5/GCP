# GCP 协同感知代码

基于 [GCP](https://github.com/yihangtao/GCP) 和 [CoPerception](https://github.com/coperception/coperception) 的最终代码整理，非论文作者官方仓库。保留攻击、防御、模型、配置与检测数据加载代码，使用说明以当前实现为准。

## 环境

建议Linux x86_64、Python 3.7.16、NVIDIA GPU。归档中的 mapping.cpython-37m-x86_64-linux-gnu.so 是运行扩展，不能用于Windows或其他Python ABI，压缩包没有其构建源码。

```bash
conda create -n gcp python=3.7.16
conda activate gcp
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install numpy opencv-python nuscenes-devkit==1.0.9 pyquaternion numba matplotlib terminaltables shapely seaborn scipy filterpy tqdm
python -m pip install "mmcv<2"
python -m pip install -e ./coperception --no-deps
```

原归档没有完整依赖锁定文件，上述安装步骤不保证所有新环境均兼容。应安装本地代码而非用同名PyPI包替代。

## 运行输入

从 [V2X-Sim官方入口](https://ai4ce.github.io/V2X-Sim/index.html) 获取数据。需要准备预处理的检测数据、与融合方法相符的检测检查点、LSTM-AE检查点，以及BH模式使用的正常样本校准文件。以上资源须另行准备，仓库不分发模型或数据。

检测检查点加载器会读取 model_state_dict、optimizer_state_dict、scheduler_state_dict，不能直接传入裸state_dict。LSTM使用8维框坐标、隐藏维度32、2层LSTM；其序列长度须与运行配置兼容。校准文件为含非空 scores 数组的 .npz，分数应来自真实正常样本且与当前评分实现一致，不要用随机分数替代。归档没有完整校准采样流水线。

原始数据预处理入口为 coperception/tools/det/create_data_det.py，主要参数为 --root、--split、--scene_begin、--scene_end、--savepath、--from_agent、--to_agent；场景与数据划分应按实际数据设置。--data 指向预处理检测数据，而非原始点云。

## 运行GCP

从仓库根目录执行，替换所有绝对路径。以下为使用示例，不代表已在新环境重新跑通或承诺特定指标。

```bash
cd coperception/tools/det
python gcp.py \
  --data /absolute/path/preprocessed/test \
  --resume /absolute/path/meanfusion/epoch_49.pth \
  --lstm_model /absolute/path/lstm/best_model.pth \
  --calibration_path /absolute/path/calibration.npz \
  --log --logpath /absolute/path/outputs/gcp_pgd \
  --com mean --bound lowerbound --gcp gcp \
  --scene_id 8 --ego_agent 1 --num_agent 6 --batch 1 \
  --history_length 5 --max_interpolation 3 --box_matching_thresh 0.3 \
  --number_of_attackers 2 --attack_ratio 0.6 \
  --adv_method pgd --eps 0.5 --adv_iter 10 \
  --attack_mode Random --use_bh --alpha_bh 0.36 --seed 42
```

场景与agent数量必须与本地数据匹配。部分源文件默认路径指向旧机器，请显式指定输入与输出路径。

| 参数 | 说明 |
|---|---|
| --adv_method | pgd、bim、cw-l2、bac等 |
| --attack_mode | Random、Poission、SI；Poission为保留的命令拼写 |
| --gcp | gcp、made、robosac_mAP、gated_late_fusion、no_defense、upperbound、lowerbound等 |
| --spatial_only / --time_only | 单分支选择，勿同时设置 |
| --sample_start / --sample_end | 包含端点的帧区间 |
| --colluding | 交替块坐标攻击优化 |

BAC将攻击参数改为 --adv_method bac --bac_eps 0.5 --frame_rate_hz 10。帧率应填写数据的真实值，代码据此计算0.5 Hz掩码更新间隔；--eps控制输入预算，--bac_eps控制BAC输出偏差。

BH方式需有效校准文件；非BH方式需按代码要求配置 --fixed_decision_threshold。

## 文件与输出

- coperception/coperception/：模型、配置、数据加载与运行工具；为保持包内导入关系保留了相关模块。
- coperception/tools/det/gcp.py：攻击、防御与检测评估。
- coperception/tools/det/bh_decision.py：p值与BH判决。
- coperception/tools/det/box_matching.py：框匹配。
- coperception/coperception/utils/bac_attack.py：BAC实现。
- coperception/tools/det/train_codet.py、train_lstm_ae.py：训练入口。LSTM训练读取含 bev_flows 的序列文件，参数见 --data_path、--save_path、--seq_length、--epochs。

终端输出各agent和平均AP@0.5/AP@0.7，启用 --log 后在 --logpath 下保存日志。ego指标应读取对应agent行；脚本中部分FPS是按固定时间系数估算，不能视作当前设备实测速度。

## 来源与许可

保留上游LICENSE与源码版权注释。本仓库不包含复现报告、实验结果、日志、模型权重、数据集或校准数据。代码实现与论文的完全一致性不作额外声明。
