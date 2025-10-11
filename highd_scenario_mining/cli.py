"""Command line interface for the HighD scenario mining pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import DEFAULT_CONFIG, PipelineConfig, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HighD 场景识别与覆盖度评估工具")
    parser.add_argument("--prefix", type=str, required=True, help="HighD 文件名前缀，例如 18")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.cwd(),
        help="HighD 原始 CSV 文件所在目录 (默认: 当前目录)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出结果目录 (默认: 与数据目录相同)",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="包含配置覆盖项的 JSON 文件 (可选)",
    )
    return parser


def parse_config(config_json: Path | None) -> PipelineConfig:
    if not config_json:
        return DEFAULT_CONFIG
    import json

    data = json.loads(Path(config_json).read_text(encoding="utf-8"))
    base = DEFAULT_CONFIG.__dict__.copy()
    base.update(data)
    return PipelineConfig(**base)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = parse_config(args.config_json)
    run(prefix=args.prefix, data_dir=args.data_dir, output_dir=args.output_dir, cfg=cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
