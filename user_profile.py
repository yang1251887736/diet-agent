"""长期记忆：从历史运行中推导用户偏好（"越用越懂你"）。

设计决策（面试考点）：偏好不单独存表，而是从 runs 历史实时推导——
历史即记忆，单一事实来源，不存在"档案与历史不一致"的问题；
代价是每次推导要扫一遍 runs（个人使用量级下可忽略）。
"""
from collections import Counter

from data.constraints import normalize_dislikes, normalize_prefs
from store import all_runs


def build_hint(min_runs: int = 2) -> str:
    """生成注入 planner prompt 的偏好提示。历史不足 2 次时返回空串。

    统计前先做中英归一化：否则"素食"和 vegetarian 会被当成两个
    偏好分开计数，提示里出现重复条目（真实跑出来的毛病）。"""
    runs = all_runs()
    if len(runs) < min_runs:
        return ""
    pref_c, dis_c = Counter(), Counter()
    for _req, prefs, dislikes in runs:
        for p in normalize_prefs((prefs or "").split(",")):
            pref_c[p] += 1
        for d in normalize_dislikes((dislikes or "").split(",")):
            dis_c[d] += 1
    parts = [f"prefers {p} ({c}/{len(runs)} previous requests)"
             for p, c in pref_c.most_common(3)]
    parts += [f"dislikes {d} ({c}/{len(runs)})" for d, c in dis_c.most_common(3)]
    if not parts:
        return ""
    return ("User preference history (long-term memory): " + "; ".join(parts)
            + ". Default to these unless the user says otherwise.")
