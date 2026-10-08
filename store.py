"""运行历史（SQLite 单文件）：多轮对话的记忆来源，长期偏好从这里实时推导。"""
import json
import sqlite3
from datetime import datetime

from config import RESULTS_DIR

_DB = RESULTS_DIR / "history.db"


def _conn() -> sqlite3.Connection:
    RESULTS_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS runs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, request TEXT, prefs TEXT, dislikes TEXT,
        coverage TEXT, plan_json TEXT, ok INTEGER)""")
    # 老库迁移：加 likes 列（"历史即记忆"也要记住用户喜好）
    try:
        conn.execute("ALTER TABLE runs ADD COLUMN likes TEXT")
    except sqlite3.OperationalError:
        pass   # 列已存在
    return conn


def save_run(state: dict) -> int:
    v = state.get("validation", {})
    ok = 1 if (state.get("plan", {}).get("days")
               and not v.get("violations") and not v.get("invalid_ids")) else 0
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO runs(ts, request, prefs, dislikes, coverage, plan_json, ok, likes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"),
         state.get("user_request", ""),
         ",".join(state.get("diet_prefs") or []),
         ",".join(state.get("dislikes") or []),
         state.get("coverage", "?"),
         json.dumps(state.get("plan", {}), ensure_ascii=False),
         ok,
         ",".join(state.get("likes") or [])))
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


def list_runs(limit: int = 10) -> list:
    conn = _conn()
    rows = conn.execute(
        "SELECT id, ts, request, ok, coverage FROM runs ORDER BY id DESC LIMIT ?",
        (limit,)).fetchall()
    conn.close()
    return rows


def get_plan(run_id: int):
    conn = _conn()
    row = conn.execute("SELECT plan_json FROM runs WHERE id = ?", (run_id,)).fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def latest_run():
    """最近一次运行（意图识别的'编辑最近计划'与 replan 偏好/天数合并依赖它）。"""
    conn = _conn()
    row = conn.execute(
        "SELECT id, request, plan_json, prefs, dislikes, likes FROM runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "request": row[1], "plan": json.loads(row[2]),
            "prefs": [p for p in (row[3] or "").split(",") if p],
            "dislikes": [d for d in (row[4] or "").split(",") if d],
            "likes": [x for x in ((row[5] or "") if len(row) > 5 else "").split(",") if x]}


def all_runs(limit: int = 100) -> list:
    """历史运行（request, prefs_str, dislikes_str），长期记忆从这里推导。"""
    conn = _conn()
    rows = conn.execute(
        "SELECT request, prefs, dislikes FROM runs ORDER BY id DESC LIMIT ?",
        (limit,)).fetchall()
    conn.close()
    return rows


def delete_run(run_id: int) -> bool:
    """删除指定历史记录。注意：长期记忆由 all_runs 实时推导，
    删除记录会同步影响偏好画像——这是"历史即记忆"设计的自然结果。"""
    conn = _conn()
    cur = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted
