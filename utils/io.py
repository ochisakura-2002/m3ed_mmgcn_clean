"""
项目输入输出工具。

这个文件负责和“路径、配置文件、实验输出目录”有关的公共操作。

它不负责：
1. 训练模型
2. 读取数据集
3. 计算指标
4. 分析实验结果

后续 train.py / evaluate_best_model.py / analyze_run.py 都会调用这里的函数。
"""

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import re

import yaml

from utils.output_paths import (
    configured_output_root,
    create_unique_run_dir,
    resolve_experiment_date,
    resolve_output_category,
)
from utils.run_metadata import write_run_metadata


# 当前文件路径是：
#   m3ed_mmgcn_clean/utils/io.py
# 所以 parents[1] 是项目根目录：
#   m3ed_mmgcn_clean/
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_project_root() -> Path:
    """
    返回项目根目录。

    后续所有相对路径都应该基于这个根目录解析，
    不要依赖当前终端所在目录。
    """
    return PROJECT_ROOT


def resolve_project_path(path_str: Optional[str]) -> Optional[Path]:
    """
    把路径解析成绝对路径。

    规则：
    1. None 返回 None
    2. 绝对路径直接返回
    3. 相对路径默认相对于项目根目录

    示例：
        data/raw/M3ED
        -> /home/zhiyuan/research/m3ed_mmgcn_clean/data/raw/M3ED
    """
    if path_str is None:
        return None

    path = Path(path_str)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def ensure_dir(path: Path) -> None:
    """
    如果目录不存在，就创建目录。
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def load_yaml(config_path: str) -> Dict[str, Any]:
    """
    读取 YAML 配置文件。

    参数：
        config_path:
            YAML 文件路径。
            可以是相对项目根目录的路径，也可以是绝对路径。

    返回：
        配置字典。
    """
    path = resolve_project_path(config_path)

    if path is None:
        raise ValueError("config_path cannot be None")

    if not path.exists():
        raise FileNotFoundError(f"YAML config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"YAML config file is empty: {path}")

    return config


def save_yaml(data: Dict[str, Any], save_path: Path) -> None:
    """
    保存字典为 YAML 文件。

    参数：
        data:
            要保存的配置字典。
        save_path:
            保存路径。
    """
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
        )


def sanitize_name(name: str) -> str:
    """
    清理实验名，避免 run_id 里出现空格、斜杠等不适合做路径的字符。

    示例：
        "MMGCN baseline debug"
        -> "MMGCN_baseline_debug"
    """
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

    if len(name) == 0:
        name = "experiment"

    return name


def make_run_id(experiment_name: str) -> str:
    """
    根据当前时间和实验名生成 run_id。

    示例：
        20260518_171530_mmgcn_m3ed_baseline_debug
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = sanitize_name(experiment_name)

    return f"{timestamp}_{safe_name}"


def create_run_dirs(
    output_dir: str,
    experiment_name: str,
    experiment_date: Optional[str] = None,
) -> Dict[str, Path]:
    """
    创建一次实验的输出目录。

    目录结构：
        outputs/<YYYYMMDD>/runs/<run_id>/
            logs/
            checkpoints/
            figures/

    参数：
        output_dir:
            输出根目录，通常来自 YAML 的 system.output_dir。
        experiment_name:
            实验名，通常来自 YAML 的 project.experiment_name。

    返回：
        run 相关路径字典。
    """
    output_root = resolve_project_path(output_dir)

    if output_root is None:
        raise ValueError("output_dir cannot be None")

    frozen_date = resolve_experiment_date(cli_date=experiment_date)
    run_dir = create_unique_run_dir(
        experiment_name=experiment_name,
        experiment_date=frozen_date,
        output_root=output_root,
    )
    run_id = run_dir.name
    logs_dir = run_dir / "logs"
    checkpoints_dir = run_dir / "checkpoints"
    figures_dir = run_dir / "figures"
    manifest_dir = resolve_output_category(
        "manifests", frozen_date, output_root
    ) / run_id

    ensure_dir(logs_dir)
    ensure_dir(checkpoints_dir)
    ensure_dir(figures_dir)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "logs_dir": logs_dir,
        "checkpoints_dir": checkpoints_dir,
        "figures_dir": figures_dir,
        "manifest_dir": manifest_dir,
        "experiment_date": frozen_date,
        "output_root": output_root,
        "day_output_root": output_root / frozen_date,
    }


def write_latest_run(run_id: str, run_dir: Path, manifest_dir: Path) -> Path:
    """
    写入当前 run 自己的 manifest 目录，避免跨 pipeline 共享状态。

    这个文件用于记录最近一次实验目录。
    后续 evaluate_best_model.py 可以默认读取它。
    """
    latest_path = Path(manifest_dir) / "latest_run.txt"
    ensure_dir(latest_path.parent)

    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(f"run_id={run_id}\n")
        f.write(f"run_dir={run_dir}\n")
    return latest_path


def prepare_run_environment(
    config: Dict[str, Any], experiment_date: Optional[str] = None
) -> Dict[str, Path]:
    """
    根据配置文件创建实验输出目录，并保存本次实验配置。

    这个函数后续会在 train.py 开头调用。

    它做三件事：
    1. 创建 outputs/<YYYYMMDD>/runs/<run_id>/
    2. 保存 logs/experiment_config.yaml
    3. 写当前 run manifest 目录内的 latest_run.txt

    参数：
        config:
            从 YAML 读取到的配置字典。

    返回：
        run 相关路径字典。
    """
    experiment_name = config["project"]["experiment_name"]
    output_dir = configured_output_root(config)
    frozen_date = resolve_experiment_date(
        cli_date=experiment_date,
        config=config,
    )

    run_info = create_run_dirs(
        output_dir=str(output_dir),
        experiment_name=experiment_name,
        experiment_date=frozen_date,
    )

    config.setdefault("output", {})
    config["output"].update(
        {
            "root": str(output_dir),
            "experiment_date": frozen_date,
            "output_root": str(run_info["output_root"]),
            "day_output_root": str(run_info["day_output_root"]),
            "run_dir": str(run_info["run_dir"]),
            "log_dir": str(run_info["logs_dir"]),
            "analysis_dir": str(
                resolve_output_category("analysis", frozen_date, run_info["output_root"])
            ),
            "manifest_dir": str(run_info["manifest_dir"]),
        }
    )

    config_save_path = run_info["logs_dir"] / "experiment_config.yaml"
    save_yaml(config, config_save_path)
    write_run_metadata(
        config=config,
        output_path=run_info["run_dir"] / "run_metadata.json",
        project_root=PROJECT_ROOT,
    )

    run_info["latest_run_path"] = write_latest_run(
        run_id=run_info["run_id"],
        run_dir=run_info["run_dir"],
        manifest_dir=run_info["manifest_dir"],
    )

    return run_info
