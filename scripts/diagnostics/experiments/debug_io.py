"""
测试 utils/io.py 的功能。

这个脚本用于确认：
1. YAML 配置能否读取
2. 项目路径能否正确解析
3. run 目录能否创建
4. experiment_config.yaml 能否保存
5. latest_run.txt 能否写入

它不训练模型，也不读取数据。
"""

from pathlib import Path
import sys


# 当前文件路径是：
#   m3ed_mmgcn_clean/scripts/debug_io.py
# parents[1] 是项目根目录：
#   m3ed_mmgcn_clean/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 把项目根目录加入 Python 搜索路径。
# 这样即使从 scripts/ 目录运行，也能 import utils.io。
sys.path.append(str(PROJECT_ROOT))

from utils.io import (  # noqa: E402
    get_project_root,
    load_yaml,
    resolve_project_path,
    prepare_run_environment,
)


def main() -> None:
    """
    运行 IO 工具测试。
    """
    print("=" * 60)
    print("Debug utils/io.py")
    print("=" * 60)

    print("Project root:", get_project_root())

    config_path = "configs/mmgcn/unified/m3ed/full_context/m3ed_features/skeleton.yaml"
    config = load_yaml(config_path)

    print("\nConfig loaded successfully.")
    print("Project name:", config["project"]["name"])
    print("Experiment name:", config["project"]["experiment_name"])
    print("Dataset:", config["dataset"]["name"])
    print("Model:", config["model"]["name"])

    raw_dir = resolve_project_path(config["dataset"]["raw_dir"])
    output_dir = resolve_project_path(config["system"]["output_dir"])

    print("\nResolved paths:")
    print("Raw data dir:", raw_dir)
    print("Output dir:", output_dir)

    # 为了避免和正式实验混淆，这里临时改成 debug 实验名。
    # 注意：这里只是修改内存里的 config，不会改原始 YAML 文件。
    config["project"]["experiment_name"] = "debug_io_test"

    run_info = prepare_run_environment(config)

    print("\nRun directories created:")
    for key, value in run_info.items():
        print(f"{key}: {value}")

    print("\nCheck files:")
    print("experiment_config.yaml:", run_info["logs_dir"] / "experiment_config.yaml")
    print("latest_run.txt:", run_info["latest_run_path"])

    print("\nDebug IO finished successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
