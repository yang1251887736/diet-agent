"""CLI 入口：多轮对话（规划 / 编辑 / 重规划 / 语料问答 / 历史 / 记忆）。

  venv\\Scripts\\python run_cli.py
直接输入任意话，理解节点自动识别意图并抽取约束：
  "给我三天的菜单" / "3天素食不吃辣"   → 新计划（一句话含全部约束）
  "忘了说了我不吃辣"                   → 合并上次偏好重新规划
  "把第2天晚餐换成鱼"                  → 编辑最近计划
  "咖喱是什么菜"                       → 语料问答（中文回答）
命令：history / memory / edit <run_id|plan.json> <指令> / del <id> / q
"""
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import store  # noqa: E402
from exporter import export_plan  # noqa: E402
from graph import TOKEN_STATS, build_graph  # noqa: E402
from user_profile import build_hint  # noqa: E402


def _base_state(user: str) -> dict:
    # 偏好/忌口/喜好留空：由理解节点从话里抽取（含中文归一化），历史自动延续
    return {"user_request": user, "diet_prefs": [], "dislikes": [], "likes": [],
            "intent": "", "memory_hint": "", "answer": "", "rationale_zh": "",
            "candidate_pool": [], "coverage": "?",
            "plan": {}, "plan_zh": {}, "validation": {}, "retries": 0,
            "prev_plan": {}, "edit_instruction": ""}


INTENT_ZH = {"new_plan": "规划新菜单", "edit": "修改菜单", "replan": "按新要求重排",
             "qa": "菜品问答", "plan_qa": "菜单问答", "chat": "闲聊"}
COV_ZH = {"A": "语料匹配充足", "B": "语料匹配有限", "C": "语料没找到相关菜品"}


def _show(result: dict, export_path=None) -> None:
    intent = result.get("intent", "?")
    cov = result.get("coverage", "?")
    print(f"\n［识别］{INTENT_ZH.get(intent, intent)} · {COV_ZH.get(cov, cov)}"
          f"（候选 {len(result.get('candidate_pool', []))} 道菜）")
    if result.get("memory_hint"):
        print("［记忆］结合了之前记住的偏好：", result["memory_hint"])
    print("\n=== 候选池 ===")
    for c in result.get("candidate_pool", []):
        print(f"  {c['dish_id']}  {c['dish']}")
    print("\n=== 计划 ===")
    zh = result.get("plan_zh") or {}
    pool = {c["dish_id"]: c for c in result.get("candidate_pool", [])}
    days = result.get("plan", {}).get("days", [])
    for di, day in enumerate(days, 1):
        meals = ", ".join(
            f"{m.get('slot')}:{(zh.get(m.get('dish_id'), {}) or {}).get('name') or pool.get(m.get('dish_id'), {}).get('dish', m.get('dish_id'))}"
            for m in day.get("meals", []))
        print(f"  Day {di}: {meals}")
    v = result.get("validation", {})
    print(f"\n=== 校验（retries={result.get('retries')}）===")
    for i in v.get("issues", []):
        print("  -", i)
    if not v.get("issues"):
        print("  全部通过")
    if result.get("rationale_zh"):
        print("\n=== 为什么这么安排 ===")
        print(f"  {result['rationale_zh']}")
    s = TOKEN_STATS
    print(f"\n[Token] {s['calls']} 次调用 / 输入 {s['input_tokens']} / 输出 {s['output_tokens']}")
    if export_path:
        print(f"[导出] {export_path}")


def _resolve_plan(target: str):
    if target.endswith(".json"):
        try:
            return json.loads(Path(target).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"读取失败：{e}")
            return None
    if target.isdigit():
        plan = store.get_plan(int(target))
        if plan is None:
            print(f"找不到 run #{target}")
        return plan
    print("目标须是 run_id 或 plan_*.json 路径")
    return None


def _print_history() -> None:
    rows = store.list_runs()
    if not rows:
        print("  还没有历史记录。")
        return
    print("  编号  时间               状态  内容")
    for rid, ts, req, ok, cov in rows:
        print(f"  #{rid:<3} [{ts}]  {'✓' if ok else '✗'}  {req}")


def main() -> None:
    app = build_graph()
    thread_id = f"cli-{int(time.time())}"   # 同一会话共享一个 thread（Checkpointer 断点）
    print("╔══════════════════════════════════════════════════╗")
    print("║          智膳 Agent —— 地中海膳食规划助手          ║")
    print("╚══════════════════════════════════════════════════╝")
    print("\n【怎么用】不用学任何格式，像聊天一样说话就行。例如：")
    print('  "给我三天的素食菜单，不吃辣"      → 生成菜单（一句话说完所有要求）')
    print('  "我喜欢吃鸡肉"                  → 我会记住你的喜好并安排进去')
    print('  "忘了说了我不吃辣"               → 自动合并新要求，重新排一份')
    print('  "晚餐太重复了，换换花样"          → 在现有菜单上修改')
    print('  "把第2天晚餐换成海鲜"            → 精确修改某一餐')
    print('  "这份菜单里哪些有鸡肉"            → 针对当前菜单提问')
    print('  "希腊酸奶是什么"                 → 问我菜品/食材知识')
    print("  每份菜单都会附上「为什么这么安排」的说明。")
    print("\n【命令】（打完整词回车）")
    print("  history                查看最近的菜单记录（每条有编号）")
    print("  memory                 查看我记住的你的偏好")
    print("  edit <编号> <修改要求>   修改某次菜单，如：edit 3 晚餐换成鱼")
    print("  del <编号>              删除某条记录，如：del 3")
    print("  q                      退出")
    cfg = {"configurable": {"thread_id": thread_id}}
    while True:
        user = input("\n你> ").strip()
        if not user:
            continue
        if user.lower() in {"q", "quit", "exit"}:
            break
        if user.lower() in ("history", "hs"):
            _print_history()
            continue
        if user.lower() == "memory":
            hint = build_hint()
            print("我记住的偏好（由历史推导，下次规划会自动用上）：" if hint
                  else "还没记住什么偏好——多聊几次我就懂你了。")
            if hint:
                print(" ", hint)
            continue
        if user.lower() == "del":
            print("先看编号，再输 del <编号>，例如 del 3：")
            _print_history()
            continue
        if user.lower().startswith("del "):
            target = user.split(maxsplit=1)[1].strip()
            if not target.isdigit():
                print("编号应是数字。当前记录：")
                _print_history()
                continue
            ok = store.delete_run(int(target))
            print(f"已删除 run #{target}" if ok else f"找不到 run #{target}。当前记录：")
            if not ok:
                _print_history()
            continue
        if user.lower() in ("edit", "edit "):
            print("先看编号，再输 edit <编号> <修改要求>，例如 edit 3 晚餐换成鱼：")
            _print_history()
            continue
        if user.lower().startswith("edit "):
            parts = user.split(maxsplit=2)
            if len(parts) < 3:
                print("少了修改要求。用法：edit <编号> <修改要求>，例如 edit 3 晚餐换成鱼。当前记录：")
                _print_history()
                continue
            prev = _resolve_plan(parts[1])
            if prev is None:
                continue
            latest = store.latest_run() or {}
            st = _base_state(parts[2])
            st.update({"prev_plan": prev, "edit_instruction": parts[2], "intent": "edit",
                       "diet_prefs": latest.get("prefs", []),
                       "dislikes": latest.get("dislikes", []),
                       "likes": latest.get("likes", [])})
            result = app.invoke(st, config=cfg)
            _show(result, export_plan(result))
            store.save_run(result)
            continue

        result = app.invoke(_base_state(user), config=cfg)
        intent = result.get("intent")
        if intent in ("qa", "plan_qa"):
            print(f"\n{result.get('answer', '')}")
            continue
        if intent == "chat":
            print("我是地中海膳食规划助手。可以直接说需求，比如："
                  "“3 天素食地中海计划，不吃坚果”；也可以问菜，比如：“moussaka 是什么”。")
            continue
        _show(result, export_plan(result))
        store.save_run(result)


if __name__ == "__main__":
    main()
