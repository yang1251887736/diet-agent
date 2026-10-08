"""工具层：菜品查询（防幻觉的回填数据源）+ 约束校验（规则版）。

设计说明（面试要能讲）：
- lookup_dish 是"真实数据源"：模型生成的任何菜品描述都不采信，
  展示/校验一律按 dish_id 从语料回填 —— 这是对抗 LLM 幻觉的兜底。
- 工具报错不抛异常，而是返回 {"error": ...}：错误信息要回到模型/调用方手里，
  让上层决定重试或换方案，而不是让整个 Agent 崩掉。
"""
import csv
import re

from langchain_core.tools import tool

from config import CHUNKS_CSV
from data.constraints import DIET_RULES, normalize_prefs, normalize_dislikes


def _hit(word: str, text: str) -> bool:
    """词边界匹配而非子串：子串会把 eggplant 误判为 egg、goat 误判为 oat。
    兼容复数（clams/eggs/nuts...）——曾漏放 Spaghetti with Clams，就是栽在复数上。"""
    return re.search(rf"\b{re.escape(word)}(?:s|es)?\b", text) is not None


@tool
def lookup_dish(dish_id: str) -> dict:
    """Look up a dish's real information from the corpus by dish_id
    (name, cuisine, country, course, ingredient summary).
    Call this when you need to confirm a dish exists or show its details."""
    with open(CHUNKS_CSV, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["dish_id"] == dish_id]
    if not rows:
        return {"error": f"{dish_id} not found in corpus"}
    first = rows[0]
    return {
        "dish_id": dish_id,
        "dish": first["dish"],
        "cuisine": first["cuisine"],
        "country": first["country"],
        "course": first["course"],
        "summary": " ".join(r["chunk_text"] for r in rows[:2])[:600],
    }


@tool
def check_constraints(dish_text: str, diet_prefs: list[str], dislikes: list[str]) -> dict:
    """Check whether a dish description violates dietary preferences or dislikes.
    diet_prefs: any of vegan / vegetarian / pescatarian / halal. dislikes: free-form words."""
    text = dish_text.lower()
    # 归一化：用户输"素食/不吃辣"这类自然语言，先转成规则表认识的词
    # （曾因没归一化导致中文偏好被静默忽略——候选池全是海鲜也判"合规"）
    prefs = normalize_prefs(diet_prefs)
    dislikes = normalize_dislikes(dislikes)
    violations = []
    for pref in prefs:
        for word in DIET_RULES.get(pref, []):
            if _hit(word, text):
                violations.append({"rule": pref, "hit": word})
    for d in dislikes:
        if _hit(d, text):
            violations.append({"rule": "dislike", "hit": d})
    return {"ok": not violations, "violations": violations}


# ── 自测：5 个用例全过说明工具层正常（不花 API 钱） ────────────────
if __name__ == "__main__":
    d = lookup_dish.invoke({"dish_id": "dish_0001"})
    assert d["dish"].lower() == "adobo" and "cuisine" in d, d
    assert "error" in lookup_dish.invoke({"dish_id": "dish_9999"})

    r = check_constraints.invoke({"dish_text": "lamb chops with yogurt sauce",
                                  "diet_prefs": ["vegetarian"], "dislikes": []})
    assert r["ok"] is False and any(v["hit"] == "lamb" for v in r["violations"]), r

    r2 = check_constraints.invoke({"dish_text": "tomato and olive salad",
                                   "diet_prefs": ["vegan"], "dislikes": ["olive"]})
    assert r2["ok"] is False and any(v["rule"] == "dislike" for v in r2["violations"]), r2

    r3 = check_constraints.invoke({"dish_text": "grilled vegetables with olive oil and herbs",
                                   "diet_prefs": ["vegan"], "dislikes": []})
    assert r3["ok"] is True, r3

    # 词边界回归用例：子串匹配会把 eggplant 误判为 egg、goat 误判为 oat
    r4 = check_constraints.invoke({"dish_text": "roasted eggplant with tahini",
                                   "diet_prefs": ["vegan"], "dislikes": []})
    assert r4["ok"] is True, r4
    r5 = check_constraints.invoke({"dish_text": "goat cheese and spinach tart",
                                   "diet_prefs": ["vegan"], "dislikes": []})
    assert r5["ok"] is False and any(v["hit"] == "cheese" for v in r5["violations"]), r5

    # 复数回归用例：词边界不认复数曾漏放 Spaghetti with Clams
    r6 = check_constraints.invoke({"dish_text": "Spaghetti with Clams in white wine",
                                   "diet_prefs": ["vegetarian"], "dislikes": []})
    assert r6["ok"] is False and any(v["hit"] == "clam" for v in r6["violations"]), r6
    # 归一化用例：中文偏好/忌口
    r7 = check_constraints.invoke({"dish_text": "spicy lamb stew with chili",
                                   "diet_prefs": ["素食"], "dislikes": ["不吃辣"]})
    assert r7["ok"] is False and len(r7["violations"]) >= 2, r7
    print("工具层自测 9/9 通过")
