"""冒烟测试：规划 → 闲聊 → 编辑（自动意图）一次跑完。花少量 API 费用。

运行：venv\\Scripts\\python smoke_test.py
"""
import os
import time

from dotenv import load_dotenv

load_dotenv()

import retriever  # noqa: E402,F401  必须先于 langchain 导入（规避 transformers 懒加载冲突）
import store  # noqa: E402
from graph import TOKEN_STATS, build_graph  # noqa: E402

app = build_graph()
cfg = {"configurable": {"thread_id": f"smoke-{int(time.time())}"}}


def run(user, prefs=None, dislikes=None):
    st = {"user_request": user, "diet_prefs": prefs or [], "dislikes": dislikes or [],
          "intent": "", "memory_hint": "", "candidate_pool": [], "coverage": "?",
          "plan": {}, "validation": {}, "retries": 0,
          "prev_plan": {}, "edit_instruction": ""}
    r = app.invoke(st, config=cfg)
    print(f"\n=== 输入: {user} ===")
    print(f"intent={r.get('intent')}  coverage={r.get('coverage')}  retries={r.get('retries')}")
    if r.get("memory_hint"):
        print(f"memory: {r['memory_hint'][:120]}...")
    for di, d in enumerate(r.get("plan", {}).get("days", []), 1):
        print(f"  Day {di}:", [(m.get("slot"), m.get("dish_id")) for m in d.get("meals", [])])
    issues = r.get("validation", {}).get("issues", [])
    print("issues:", issues if issues else "无")
    return r


r1 = run("Plan 2 days of Mediterranean meals with seafood")
store.save_run(r1)                                   # 存历史 → 下一步 edit 才有基底
run("hello there!")                                  # 期待 intent=chat
r3 = run("Change day 1 dinner to a vegetarian dish")  # 期待 intent=edit（自动取最近计划）

print("\n=== 历史记录 ===")
for row in store.list_runs(3):
    print(" ", row)
print("checkpoints.sqlite 存在:", os.path.exists("results/checkpoints.sqlite"))
s = TOKEN_STATS
print(f"Token: {s['calls']} 次调用 / 输入 {s['input_tokens']} / 输出 {s['output_tokens']}")
