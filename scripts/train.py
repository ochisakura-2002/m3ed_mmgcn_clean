"""
M3ED + MMGCN 训练入口脚本。

当前版本是最小入口版本，只负责：
1. 读取 YAML 配置
2. 设置随机种子
3. 创建本次实验的 run 目录
4. 保存 experiment_config.yaml
5. 打印当前配置摘要

当前版本不做：
1. 不读取 M3ED 数据
2. 不构建 MMGCN 模型
3. 不训练模型
4. 不保存 checkpoint

这个文件后续会逐步扩展成完整 train.py。
"""

from pathlib import Path
import argparse
import sys


# 当前文件路径：
#   m3ed_mmgcn_clean/scripts/train.py
# parents[1] 是项目根目录：
#   m3ed_mmgcn_clean/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 保证从任意目录运行该脚本时，都能 import 项目内模块。
sys.path.append(str(PROJECT_ROOT))

from utils.io import load_yaml, prepare_run_environment  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    当前只支持一个参数：
        --config: YAML 配置文件路径。
    """
    parser = argparse.ArgumentParser(
        description="Minimal training entry for M3ED + MMGCN."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_mmgcn_m3ed.yaml",
        help="Path to YAML config file.",
    )

    return parser.parse_args()


def print_config_summary(config: dict, config_path: str, run_info: dict) -> None:
    """
    打印当前实验配置摘要。

    这里只打印最关键的信息，方便启动脚本后快速确认：
    1. 是否读到了正确配置
    2. 是否创建了正确 run 目录
    3. 当前实验名、模型、数据集是否正确
    """
    print("=" * 60)
    print("M3ED + MMGCN minimal train entry")
    print("=" * 60)

    print("Project root:", PROJECT_ROOT)
    print("Config path:", config_path)

    print("\nProject:")
    print("  name:", config["project"]["name"])
    print("  experiment_name:", config["project"]["experiment_name"])

    print("\nSystem:")
    print("  seed:", config["system"]["seed"])
    print("  device:", config["system"]["device"])
    print("  output_dir:", config["system"]["output_dir"])

    print("\nDataset:")
    print("  name:", config["dataset"]["name"])
    print("  raw_dir:", config["dataset"]["raw_dir"])
    print("  processed_dir:", config["dataset"]["processed_dir"])
    print("  metadata_dir:", config["dataset"]["metadata_dir"])
    print("  num_classes:", config["dataset"]["num_classes"])

    print("\nModel:")
    print("  name:", config["model"]["name"])
    print("  hidden_dim:", config["model"]["hidden_dim"])
    print("  dropout:", config["model"]["dropout"])

    print("\nTrain:")
    print("  batch_size:", config["train"]["batch_size"])
    print("  learning_rate:", config["train"]["learning_rate"])
    print("  weight_decay:", config["train"]["weight_decay"])
    print("  max_epochs:", config["train"]["max_epochs"])

    print("\nRun directories:")
    print("  run_id:", run_info["run_id"])
    print("  run_dir:", run_info["run_dir"])
    print("  logs_dir:", run_info["logs_dir"])
    print("  checkpoints_dir:", run_info["checkpoints_dir"])
    print("  figures_dir:", run_info["figures_dir"])

    print("\nCurrent status:")
    print("  This is a minimal train entry script.")
    print("  Next step: inspect MMGCN code and M3ED data format.")
    print("=" * 60)


def main() -> None:
    """
    主函数。

    当前流程：
    1. 读取命令行参数
    2. 加载 YAML 配置
    3. 设置随机种子
    4. 创建 run 目录
    5. 打印配置摘要
    """
    args = parse_args()

    config = load_yaml(args.config)

    seed = int(config["system"]["seed"])
    set_seed(seed)

    run_info = prepare_run_environment(config)

    print_config_summary(
        config=config,
        config_path=args.config,
        run_info=run_info,
    )


if __name__ == "__main__":
    main()