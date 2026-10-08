"""计划导出：Markdown（人读）+ JSON（程序读），固定文件名覆盖写。"""
import json
from datetime import datetime
from pathlib import Path

from config import RESULTS_DIR


def plan_to_markdown(state: dict) -> str:
    pool = {c["dish_id"]: c for c in state.get("candidate_pool", [])}
    zh = state.get("plan_zh") or {}
    v = state.get("validation", {})
    lines = [
        f"# 膳食计划（{datetime.now():%Y-%m-%d %H:%M}）",
        f"需求：{state.get('user_request', '')}",
        f"偏好：{', '.join(state.get('diet_prefs') or []) or '无'}"
        f"　喜好：{', '.join(state.get('likes') or []) or '无'}"
        f"　忌口：{', '.join(state.get('dislikes') or []) or '无'}",
        f"语料覆盖等级：{state.get('coverage', '?')}（候选 {len(pool)} 道菜）",
        "",
    ]
    for di, day in enumerate(state.get("plan", {}).get("days", []), 1):
        lines.append(f"## Day {di}")
        for m in day.get("meals", []):
            did = m.get("dish_id", "")
            zh_name = zh.get(did, {}).get("name")
            en_name = pool.get(did, {}).get("dish", did or "?")
            name = zh_name or en_name
            line = f"- **{m.get('slot', '?')}**：{name}（`{did}`）"
            if zh_name:
                line += f" — {en_name}"     # 中文为主，英文原名备查
            lines.append(line)
            s = (zh.get(did, {}).get("summary") or "").strip()
            if s:
                lines.append(f"  {s}")
    if state.get("rationale_zh"):
        lines += ["", "## 为什么这么安排", state["rationale_zh"]]
    if v.get("issues"):
        lines += ["", "> ⚠️ 未完全满足约束："]
        lines += [f"> - {i}" for i in v["issues"]]
    return "\n".join(lines)


def export_plan(state: dict) -> Path:
    """写 results/plan.md + plan.json（固定文件名覆盖写）。

    早期版本用时间戳命名（plan_20260928_140200.md），每次运行都新增
    两个文件，测试期攒了几十份垃圾。改为固定名覆盖：时间戳在 md 内容
    里，历史在 SQLite 里（edit <run_id> 可调回任意旧计划），磁盘上
    永远只有最新一份。json 供 edit 命令复用。"""
    RESULTS_DIR.mkdir(exist_ok=True)
    md_path = RESULTS_DIR / "plan.md"
    json_path = RESULTS_DIR / "plan.json"
    md_path.write_text(plan_to_markdown(state), encoding="utf-8")
    json_path.write_text(json.dumps(state.get("plan", {}), ensure_ascii=False, indent=2),
                         encoding="utf-8")
    return md_path
