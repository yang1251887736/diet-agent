"""Token 消耗统计（跨模块累加，run_cli / evaluate 结束时打印）。"""

STATS = {"calls": 0, "input_tokens": 0, "output_tokens": 0}


def acc(resp) -> None:
    """从 langchain 响应里累加 token 用量（字段缺失时静默跳过）。"""
    STATS["calls"] += 1
    u = getattr(resp, "usage_metadata", None) or {}
    STATS["input_tokens"] += u.get("input_tokens", 0) or 0
    STATS["output_tokens"] += u.get("output_tokens", 0) or 0


def reset() -> None:
    STATS.update(calls=0, input_tokens=0, output_tokens=0)
