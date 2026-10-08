"""评测脚本（简历数字来源）。

指标一（检索三模式对比，检索优化链的量化验证）：
    baseline / +Query Rewrite / +Rewrite & Rerank 各跑 210 题，
    出 Top1 / MRR 对比表 —— 优化带来的提升就是简历上的核心数字。
    口径：答案文本中出现的菜品名是否被召回（Top1 = 第 1 位即命中；
    MRR = 首个命中排名的倒数均值）。口径在报告"局限"里注明。

指标二（幻觉率）：planner 引用的 dish_id 不在候选池的占比。
    首轮（max_retries=0，量机制的必要性）vs 回流后（完整图，量机制的有效性）。

指标三（约束满足率）：首轮违规率 vs 回流收敛后满足率（固定 6 个请求）。

指标四（Agent 行为）：意图分类准确率（19 条标注语句 × 六路意图 × 有/无历史）
    + 工具调用成功率（lookup_dish 调用统计，随规划评测自动收集）。

指标五（端到端场景）：3 个多轮对话场景 × 逐检查点断言（意图/计划结构/
    约束/喜好落实/编辑精确性/诚实回答），整场景全过为"任务完成"。
    评测用独立历史库（results/e2e_history.db），不污染 CLI 真实历史。
    这是单点指标之外的"整机测试"，最贴 Agent 的真实质量。

指标六（RAG 生成质量，RAGAS 风格三指标，LLM-as-judge）：
    忠实度 faithfulness（答案论断是否被上下文支撑）
    答案相关性 answer_relevancy（从答案反推问题，与原问题算 embedding 余弦，
    RAGAS 官方口径，embedding 用本地 MiniLM）
    上下文相关性 context_relevance（召回 chunk 中含回答所需信息的比例）。
    默认不跑（每题 4 次 LLM 调用），python evaluate.py --qa 30 开跑，--qa all 全量。

运行：venv\\Scripts\\python evaluate.py            # 指标一二三四五
     venv\\Scripts\\python evaluate.py --qa 30     # 追加指标六（+约 120 次调用）
费用：检索 210 题 × rewrite ≈ 210 次 LLM 调用（DeepSeek 几分钱）；
     规划 6 请求 × 2 图；意图 19 次；端到端 3 场景约 40 次。
"""
import argparse
import json
import re
from datetime import datetime

from config import (QA_DIR, RESULTS_DIR, USE_REWRITE, USE_RERANK,
                    DEEPSEEK_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL)
from graph import (TOKEN_STATS, build_graph,
                   _to_english_query, _llm_invoke_retry, acc)
from rag import retrieve_v2

# 固定请求集：覆盖 无约束 / 四种偏好 / 忌口，全部温度 0 可复现
EVAL_REQUESTS = [
    {"user_request": "Plan 3 days of Mediterranean meals", "diet_prefs": [], "dislikes": []},
    {"user_request": "Plan 2 days of vegetarian Mediterranean meals", "diet_prefs": ["vegetarian"], "dislikes": []},
    {"user_request": "Plan 2 days of vegan Mediterranean meals", "diet_prefs": ["vegan"], "dislikes": []},
    {"user_request": "Plan 2 days of Mediterranean meals with seafood focus", "diet_prefs": ["pescatarian"], "dislikes": []},
    {"user_request": "Plan 2 days of Mediterranean meals without nuts", "diet_prefs": [], "dislikes": ["nuts"]},
    {"user_request": "Plan 3 days of halal Mediterranean meals", "diet_prefs": ["halal"], "dislikes": []},
]


# ── 指标一：检索三模式 ──────────────────────────────────
def load_questions() -> list:
    qs = []
    for f in QA_DIR.glob("*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        for src in data.get("sources", []):
            for q in src.get("questions", []):
                qs.append({"source": f.name, **q})
    return qs


def eval_retrieval(questions: list, use_rewrite: bool, use_rerank: bool) -> dict:
    hits1, rr, n = 0, 0.0, 0
    for q in questions:
        answer = (q.get("answer") or "").lower()
        if not answer:
            continue
        ranked = retrieve_v2(q["question"], k=10,
                             use_rewrite=use_rewrite, use_rerank=use_rerank)
        ranks = [i + 1 for i, r in enumerate(ranked) if r["dish"].lower() in answer]
        n += 1
        if ranks:
            if ranks[0] == 1:
                hits1 += 1
            rr += 1.0 / ranks[0]
    return {"n": n, "top1": round(hits1 / n, 4), "mrr": round(rr / n, 4)}


# ── 指标四：RAG 生成质量（RAGAS 风格三指标，LLM-as-judge） ────
def _judge() -> "ChatOpenAI":
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY,
                      base_url=DEEPSEEK_BASE_URL, temperature=0)


def _rag_answer(question: str) -> tuple:
    """qa 专家同款 RAG 管线：译英 → 检索 → 只准引用语料。
    返回 (英文答案, 召回的 chunks)——答案用英文是因为 judge 打分
    与参考答案/上下文同语言才公平。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    q_en = _to_english_query(question)
    hits = retrieve_v2(q_en, k=6, use_rewrite=USE_REWRITE, use_rerank=USE_RERANK)
    if not hits:
        return "", []
    ctx = "\n\n".join(f"[{h['dish']}] {h['text'][:500]}" for h in hits)
    resp = _llm_invoke_retry(_judge(), [
        SystemMessage(content=(
            "Answer the user's question about Mediterranean cuisine in English, "
            "using ONLY the provided context. If the context does not contain "
            "the answer, reply exactly: The corpus does not cover this. "
            "Keep it under 80 words.")),
        HumanMessage(content=f"Context:\n{ctx}\n\nQuestion: {question}"),
    ])
    acc(resp)
    return (resp.content or "").strip(), hits


def _faithfulness(context: str, answer: str) -> float:
    """忠实度：答案中的每个论断是否都能被上下文支撑（幻觉检测）。
    1-5 分 → 归一化到 0~1。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    resp = _llm_invoke_retry(_judge(), [
        SystemMessage(content=(
            "You are a strict evaluator. Given CONTEXT and ANSWER, score how "
            "faithful the ANSWER is to the CONTEXT: 5 = every claim is directly "
            "supported; 3 = mostly supported with minor unsupported details; "
            "1 = contains fabricated or contradictory information. "
            "Reply with ONLY one integer 1-5.")),
        HumanMessage(content=f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"),
    ])
    acc(resp)
    try:
        return (int(re.search(r"\d+", resp.content).group()) - 1) / 4
    except Exception:
        return 0.0


def _answer_relevancy(question: str, answer: str) -> float:
    """答案相关性（RAGAS 官方口径）：从答案反向生成 3 个它可能在回答的问题，
    与原问题算 embedding 余弦相似度取均值——答案跑题则生成的问题会偏离原问题。
    embedding 用本地 MiniLM，不花钱。"""
    from langchain_core.messages import HumanMessage, SystemMessage
    from retriever import _load

    resp = _llm_invoke_retry(_judge(), [
        SystemMessage(content=(
            "Generate 3 distinct questions that the given answer could be "
            "answering. Reply with ONLY the 3 questions, one per line, "
            "no numbering.")),
        HumanMessage(content=answer),
    ])
    acc(resp)
    gen_qs = [q.strip() for q in (resp.content or "").splitlines() if q.strip()][:3]
    if not gen_qs:
        return 0.0
    model = _load()[0]
    vecs = model.encode([question] + gen_qs, normalize_embeddings=True)
    qv = vecs[0]
    return float(sum(float(vecs[i] @ qv) for i in range(1, len(vecs))) / (len(vecs) - 1))


def _context_relevance(question: str, hits: list) -> float:
    """上下文相关性：召回的 chunk 中有多少真正包含回答所需信息（LLM 判定）。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    numbered = "\n\n".join(f"Chunk {i+1} [{h['dish']}]: {h['text'][:300]}"
                           for i, h in enumerate(hits))
    resp = _llm_invoke_retry(_judge(), [
        SystemMessage(content=(
            "Given a QUESTION and retrieved text CHUNKS, list the chunk numbers "
            "that contain information relevant to answering the question. "
            "Reply with ONLY the numbers separated by commas (e.g. 1,3); "
            "reply NONE if none are relevant.")),
        HumanMessage(content=f"QUESTION: {question}\n\n{numbered}"),
    ])
    acc(resp)
    text = (resp.content or "").strip()
    if not text or text.upper() == "NONE":
        return 0.0
    idxs = {int(n) for n in re.findall(r"\d+", text) if 1 <= int(n) <= len(hits)}
    return len(idxs) / len(hits)


def eval_generation(questions: list, n) -> dict:
    sample = questions if n == "all" else questions[:int(n)]
    faith = rel = ctx = 0.0
    cnt = 0
    for i, q in enumerate(sample, 1):
        ref = (q.get("answer") or "").strip()
        if not ref:
            continue
        answer, hits = _rag_answer(q["question"])
        if not hits:
            continue
        cnt += 1
        context = "\n\n".join(f"[{h['dish']}] {h['text'][:500]}" for h in hits)
        faith += _faithfulness(context, answer)
        rel += _answer_relevancy(q["question"], answer)
        ctx += _context_relevance(q["question"], hits)
        print(f"      qa {i}/{len(sample)} done", end="\r")
    print()
    if not cnt:
        return {"n": 0}
    return {"n": cnt,
            "faithfulness": round(faith / cnt, 4),
            "answer_relevancy": round(rel / cnt, 4),
            "context_relevance": round(ctx / cnt, 4)}


# ── 指标五：Agent 行为（意图分类准确率 + 工具调用成功率） ─────
# 覆盖六路意图 × 有/无历史两种上下文（edit/replan/plan_qa 只在有历史时出现）
INTENT_CASES = [
    ("plan 3 days of Mediterranean meals", False, "new_plan"),
    ("给我三天的素食菜单", False, "new_plan"),
    ("两天纯素，不吃坚果", False, "new_plan"),
    ("Plan a weekend of halal meals", True, "new_plan"),
    ("把第2天晚餐换成鱼", True, "edit"),
    ("change day 1 lunch to something else", True, "edit"),
    ("换一下昨晚那个计划，晚餐不要鸡肉", True, "edit"),
    ("忘了说了我不吃辣", True, "replan"),
    ("改成纯素", True, "replan"),
    ("另外我对海鲜过敏", True, "replan"),
    ("what is moussaka", False, "qa"),
    ("咖喱是什么菜", False, "qa"),
    ("披萨是怎么做的", False, "qa"),
    ("你给我的哪些有鸡肉", True, "plan_qa"),
    ("计划里哪天有鱼", True, "plan_qa"),
    ("第二天的早餐是什么", True, "plan_qa"),
    ("hi there", False, "chat"),
    ("今天天气怎么样", False, "chat"),
    ("谢谢", False, "chat"),
]


def eval_intents() -> dict:
    """意图分类准确率：固定 19 条标注语句，直接调路由层的理解函数，
    1 条 = 1 次 LLM 调用（不检索、不规划，便宜且可复现）。"""
    from graph import _understand

    ok = 0
    mistakes = []
    for utt, has_hist, expected in INTENT_CASES:
        got = _understand(utt, has_history=has_hist).get("intent", "new_plan")
        if got == expected:
            ok += 1
        else:
            mistakes.append({"utterance": utt, "expected": expected, "got": got})
    return {"n": len(INTENT_CASES), "accuracy": round(ok / len(INTENT_CASES), 4),
            "mistakes": mistakes}


class ToolStats:
    """工具调用统计：用同名计数工具替换 graph 命名空间里的 lookup_dish。

    ⚠️ 必须保持 LangChain 工具接口——节点用 lookup_dish.invoke({...}) 调用、
    planner 的 bind_tools 需要它的 name/args schema。第一版用普通函数包一层
    直接把规划链打炸（'function' object has no attribute 'invoke'），
    所以这里重建一个 StructuredTool 而不是包函数。"""

    calls = 0
    ok = 0

    @classmethod
    def install(cls):
        from langchain_core.tools import StructuredTool
        import graph as gmod
        orig = gmod.lookup_dish
        orig_func = orig.func   # StructuredTool 底层的同步函数

        def counting(dish_id: str) -> dict:
            cls.calls += 1
            r = orig_func(dish_id)
            if not r.get("error"):
                cls.ok += 1
            return r

        gmod.lookup_dish = StructuredTool.from_function(
            func=counting, name=orig.name, description=orig.description)

    @classmethod
    def summary(cls) -> dict:
        return {"calls": cls.calls,
                "success_rate": round(cls.ok / cls.calls, 4) if cls.calls else None}


# ── 指标六：端到端场景评测（多轮对话脚本 → 逐检查点断言） ──────
# 和单点指标的区别：单点测"零件"，端到端测"整机"——一句话进，整张图
# （理解→检索→规划/问答→校验→回答）走通且结果对才算过，最贴 Agent 的真实质量。
# 历史隔离：评测把 store._DB 指到独立库，不污染 CLI 的真实历史与记忆画像。
def _meal_map(plan: dict) -> dict:
    out = {}
    for di, day in enumerate((plan or {}).get("days", [])):
        for m in day.get("meals", []):
            out[(di, m.get("slot", ""))] = m.get("dish_id", "")
    return out


def _plan_dish_names(plan: dict) -> list:
    from tools import lookup_dish
    names = []
    for did in set(_meal_map(plan).values()):
        info = lookup_dish.invoke({"dish_id": did})
        if "error" not in info:
            names.append(info["dish"].lower())
    return names


def _run_check(check: dict, state: dict, prev_state: dict) -> tuple:
    """单个检查点 → (是否通过, 说明)。check = {"type": ..., "expect": ...}"""
    t, exp = check["type"], check.get("expect")
    if t == "intent":
        # replan 会在理解节点里与历史偏好合并后转为 new_plan（设计如此），
        # 所以"忘了说了我不吃辣"的最终意图就是 new_plan
        return state.get("intent") == exp, f"intent={state.get('intent')}"
    if t == "plan_days":
        got = len(state.get("plan", {}).get("days", []))
        return got == exp, f"days={got}"
    if t == "meals_per_day":
        bad = [di for di, d in enumerate(state.get("plan", {}).get("days", []), 1)
               if {m.get("slot") for m in d.get("meals", [])} != {"breakfast", "lunch", "dinner"}]
        return not bad, f"缺餐次的天={bad or '无'}"
    if t == "validation_clean":
        v = state.get("validation", {})
        return not v.get("violations") and not v.get("invalid_ids"), \
            f"violations={len(v.get('violations', []))}, invalid={len(v.get('invalid_ids', []))}"
    if t in ("prefs_contain", "dislikes_contain", "likes_contain"):
        key = {"prefs_contain": "diet_prefs", "dislikes_contain": "dislikes",
               "likes_contain": "likes"}[t]
        got = {w.lower() for w in state.get(key) or []}
        need = {w.lower() for w in exp}
        return need <= got, f"{key}={sorted(got)}"
    if t == "dislikes_any":
        got = {w.lower() for w in state.get("dislikes") or []}
        return bool({w.lower() for w in exp} & got), f"dislikes={sorted(got)}"
    if t == "answer_contains":
        ans = (state.get("answer") or "").lower()
        hit = [w for w in exp if w.lower() in ans]
        return bool(hit), f"命中={hit or '无'}"
    if t == "answer_not_empty":
        return bool((state.get("answer") or "").strip()), ""
    if t == "dish_name_contains":
        # 喜好落实检查：计划里菜名含关键词的数量 >= expect
        kw = check["keyword"].lower()
        n = sum(1 for name in _plan_dish_names(state.get("plan", {})) if kw in name)
        return n >= exp, f"含'{check['keyword']}'的菜 {n} 道"
    if t == "changed_slots_max":
        # 编辑精确性：只允许动有限个餐位（planner 会重建整份计划，
        # 阈值放宽到 3：允许多米诺式的少量连带调整，不允许大面积重排）
        a, b = _meal_map(prev_state.get("plan", {})), _meal_map(state.get("plan", {}))
        changed = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
        return len(changed) <= exp, f"改动了 {len(changed)} 个餐位: {changed}"
    return False, f"未知检查类型 {t}"


E2E_SCENARIOS = [
    {"name": "多轮记忆：点菜→补忌口→诚实问答",
     "turns": [
         {"input": "给我两天的地中海菜单",
          "save": True,
          "checks": [{"type": "intent", "expect": "new_plan"},
                     {"type": "plan_days", "expect": 2},
                     {"type": "meals_per_day"},
                     {"type": "validation_clean"}]},
         {"input": "忘了说了我不吃辣",
          "save": True,
          "checks": [{"type": "intent", "expect": "new_plan"},   # replan 合并后即 new_plan
                     {"type": "dislikes_any", "expect": ["chili", "chilli", "spicy"]},
                     {"type": "plan_days", "expect": 2},
                     {"type": "validation_clean"}]},
         {"input": "咖喱是什么菜",
          "save": False,
          "checks": [{"type": "intent", "expect": "qa"},
                     {"type": "answer_contains", "expect": ["没有"]}]},  # 语料无咖喱，必须诚实
     ]},
    {"name": "喜好全链路：编排→菜单问答→编辑精确",
     "turns": [
         {"input": "三天的菜单，我喜欢吃鸡肉",
          "save": True,
          "checks": [{"type": "plan_days", "expect": 3},
                     {"type": "dish_name_contains", "keyword": "chicken", "expect": 3},
                     {"type": "validation_clean"}]},
         {"input": "你给我的哪些有鸡肉",
          "save": False,
          "checks": [{"type": "intent", "expect": "plan_qa"},
                     {"type": "answer_contains", "expect": ["chicken", "鸡"]}]},
         {"input": "把第一天的晚餐换成另一种鸡的做法",
          "save": True,
          "checks": [{"type": "changed_slots_max", "expect": 3},
                     {"type": "dish_name_contains", "keyword": "chicken", "expect": 3}]},
         {"input": "计划里哪天有鱼",
          "save": False,
          "checks": [{"type": "intent", "expect": "plan_qa"},
                     {"type": "answer_not_empty"}]},
     ]},
    {"name": "中文约束 + 跨语言问答",
     "turns": [
         {"input": "两天纯素菜单，不要坚果",
          "save": True,
          "checks": [{"type": "prefs_contain", "expect": ["vegan"]},
                     {"type": "plan_days", "expect": 2},
                     {"type": "meals_per_day"},
                     {"type": "validation_clean"}]},
         {"input": "moussaka 是什么",
          "save": False,
          "checks": [{"type": "intent", "expect": "qa"},
                     {"type": "answer_contains", "expect": ["moussaka"]}]},
     ]},
]


def eval_e2e() -> dict:
    import store
    store._DB = RESULTS_DIR / "e2e_history.db"   # 历史隔离：不污染真实 CLI 库
    if store._DB.exists():
        store._DB.unlink()   # 清掉上一次评测的残留历史，保证 has_history 判断干净
    g = build_graph(with_intent=True, translate=False)   # 端到端 = 完整图，含理解节点
    scenario_rows = []
    total_checks = passed_checks = 0
    for si, sc in enumerate(E2E_SCENARIOS):
        cfg = {"configurable": {"thread_id": f"e2e-s{si}"}}
        n_ok = n_all = 0
        details = []
        prev_state = {}
        prev_plan_state = {}   # 最近一个"带计划"的轮次——编辑精确性的对比基底
        for ti, turn in enumerate(sc["turns"], 1):
            state = dict(user_request=turn["input"], diet_prefs=[], dislikes=[], likes=[],
                         intent="", memory_hint="", answer="", rationale_zh="",
                         candidate_pool=[], coverage="?", plan={}, plan_zh={},
                         validation={}, retries=0, prev_plan={}, edit_instruction="")
            result = g.invoke(state, config=cfg)
            for check in turn["checks"]:
                # ⚠️ changed_slots 的基底必须是"最近带计划的轮次"：
                # plan_qa/qa 轮的 state 里没有 plan（空字典），拿它对比会把
                # 所有餐位误判成"全改了"（第一版评测脚本踩的坑）
                base = prev_plan_state if check["type"] == "changed_slots_max" else prev_state
                ok, detail = _run_check(check, result, base)
                n_all += 1
                n_ok += ok
                if not ok:
                    details.append(f"T{ti} '{turn['input'][:18]}' {check['type']}: {detail}")
            if turn.get("save"):
                store.save_run(result)   # 等价于 CLI 的行为，后续轮次才有"记忆"可依赖
            if result.get("plan", {}).get("days"):
                prev_plan_state = result
            prev_state = result
        total_checks += n_all
        passed_checks += n_ok
        scenario_rows.append({"name": sc["name"], "turns": len(sc["turns"]),
                              "ok": n_ok, "all": n_all,
                              "pass": n_ok == n_all, "fail_detail": details})
        mark = "✓" if n_ok == n_all else "✗"
        print(f"      {mark} {sc['name']}（{n_ok}/{n_all} 检查点）")
    return {"n_scenarios": len(E2E_SCENARIOS),
            "completion_rate": round(sum(1 for r in scenario_rows if r["pass"])
                                     / len(E2E_SCENARIOS), 4),
            "check_pass_rate": round(passed_checks / total_checks, 4) if total_checks else 0,
            "rows": scenario_rows}


# ── 指标二 & 三：规划 ────────────────────────────────────
def _dish_refs(state: dict) -> list:
    return [m.get("dish_id", "")
            for d in state.get("plan", {}).get("days", [])
            for m in d.get("meals", [])]


def eval_planning() -> list:
    # 评测固定走规划链（with_intent=False）：不花意图分类的钱，指标更稳定；
    # translate=False：翻译是展示层节点，评测不需要
    ToolStats.install()   # 顺手收集工具调用成功率（指标四的一部分）
    g_single = build_graph(max_retries=0, with_intent=False, translate=False)
    g_full = build_graph(with_intent=False, translate=False)
    rows = []
    for i, req in enumerate(EVAL_REQUESTS):
        base = {**req, "intent": "", "memory_hint": "", "candidate_pool": [], "coverage": "?",
                "plan": {}, "validation": {}, "retries": 0,
                "prev_plan": {}, "edit_instruction": ""}
        s0 = g_single.invoke(base, config={"configurable": {"thread_id": f"eval-s{i}"}})
        pool_ids0 = {c["dish_id"] for c in s0["candidate_pool"]}
        refs0 = _dish_refs(s0)
        v0 = s0["validation"]

        sf = g_full.invoke(base, config={"configurable": {"thread_id": f"eval-f{i}"}})
        # ⚠️ 回流后的幻觉必须对照【本次运行自己的】候选池：两次运行各自检索，
        # rewrite 的 LLM 输出可能存在微小非确定性导致池子不同——拿 s0 的池核对
        # sf 的输出会把合法 ID 误判为幻觉（第一版评测脚本踩过的真实 bug）。
        pool_ids_f = {c["dish_id"] for c in sf["candidate_pool"]}
        refs_f = _dish_refs(sf)
        vf = sf["validation"]
        rows.append({
            "request": req["user_request"],
            "pool_size": len(pool_ids0),
            "first_halluc": sum(1 for r in refs0 if r not in pool_ids0),
            "final_halluc": sum(1 for r in refs_f if r not in pool_ids_f),
            "total_refs": len(refs_f) or len(refs0),
            "first_violations": len(v0.get("violations", [])) + len(v0.get("invalid_ids", [])),
            "final_ok": bool(refs_f) and not vf.get("violations") and not vf.get("invalid_ids"),
        })
    return rows


def _summarize(rows: list) -> dict:
    total_refs = sum(r["total_refs"] for r in rows)
    return {
        "n_req": len(rows),
        "first_halluc_rate": round(sum(r["first_halluc"] for r in rows) / total_refs, 4) if total_refs else 0,
        "final_halluc_rate": round(sum(r["final_halluc"] for r in rows) / total_refs, 4) if total_refs else 0,
        "first_violation_rate": round(sum(1 for r in rows if r["first_violations"] > 0) / len(rows), 4),
        "final_ok_rate": round(sum(1 for r in rows if r["final_ok"]) / len(rows), 4),
    }


def main():
    ap = argparse.ArgumentParser(description="智膳 Agent 评测")
    ap.add_argument("--qa", default="0",
                    help="追加 RAG 生成质量评测的题数（如 30）或 'all'（全量 210 题），默认不跑")
    ap.add_argument("--no-e2e", action="store_true",
                    help="跳过端到端场景评测（默认跑，3 场景约 40 次调用）")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    print("[1/6] 检索 baseline …")
    qs = load_questions()
    ret_base = eval_retrieval(qs, use_rewrite=False, use_rerank=False)
    print(f"      Top1={ret_base['top1']} MRR={ret_base['mrr']}")

    print("[2/6] 检索 + Query Rewrite / + Rerank …（210 次 LLM 调用）")
    ret_rw = eval_retrieval(qs, use_rewrite=True, use_rerank=False)
    ret_rr = eval_retrieval(qs, use_rewrite=True, use_rerank=True)
    print(f"      rewrite Top1={ret_rw['top1']} | rewrite+rerank Top1={ret_rr['top1']}")

    print("[3/6] 规划评测（6 请求 × 2 图）…")
    rows = eval_planning()
    summary = _summarize(rows)

    print("[4/6] 意图分类准确率（19 条标注语句）…")
    intents = eval_intents()
    print(f"      准确率 {intents['accuracy']}（{intents['n'] - len(intents['mistakes'])}/{intents['n']}）")
    tools_stat = ToolStats.summary()

    e2e = None
    if not args.no_e2e:
        print("[5/6] 端到端场景评测（3 场景 × 多轮对话）…")
        e2e = eval_e2e()

    gen = None
    if args.qa not in ("0", "", "no"):
        print(f"[6/6] RAG 生成质量评测（n={args.qa}，RAGAS 风格三指标）…")
        gen = eval_generation(qs, args.qa)
        print(f"      忠实度={gen['faithfulness']} 答案相关性={gen['answer_relevancy']} "
              f"上下文相关性={gen['context_relevance']}")

    s = TOKEN_STATS
    lines = [
        "# 智膳 Agent 评测报告",
        f"\n生成时间：{datetime.now():%Y-%m-%d %H:%M}　模型：deepseek-chat（temperature=0）"
        f"　Token：{s['calls']} 次调用 / 输入 {s['input_tokens']} / 输出 {s['output_tokens']}",
        "\n## 1. 检索三模式对比（n=%d）" % ret_base["n"],
        "\n| 模式 | Top1 命中率 | MRR |",
        "|---|---|---|",
        f"| baseline（纯向量召回） | {ret_base['top1']} | {ret_base['mrr']} |",
        f"| + Query Rewrite | {ret_rw['top1']} | {ret_rw['mrr']} |",
        f"| + Rewrite & Cross-Encoder Rerank | {ret_rr['top1']} | {ret_rr['mrr']} |",
        "\n口径：答案文本中出现的菜品名是否被召回。",
        "\n## 2. 规划（固定 6 个请求，规划链开启 rewrite+rerank）",
        "\n| 请求 | 候选池 | 首轮幻觉 | 回流后幻觉 | 首轮违规 | 最终通过 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['request']} | {r['pool_size']} | {r['first_halluc']} | "
                     f"{r['final_halluc']} | {r['first_violations']} | {'是' if r['final_ok'] else '否'} |")
    lines += [
        f"\n| 指标 | 数值 |\n|---|---|",
        f"| 首轮幻觉率（dish_id 不在候选池） | {summary['first_halluc_rate']} |",
        f"| 回流后幻觉率 | {summary['final_halluc_rate']} |",
        f"| 首轮约束违规率（按请求） | {summary['first_violation_rate']} |",
        f"| 最终约束满足率 | {summary['final_ok_rate']} |",
        "\n## 3. Agent 行为",
        f"\n| 指标 | 数值 |\n|---|---|",
        f"| 意图分类准确率（六路意图 × 有/无历史，n={intents['n']}） | {intents['accuracy']} |",
        f"| 工具调用（lookup_dish）次数 | {tools_stat['calls']} |",
        f"| 工具调用成功率 | {tools_stat['success_rate'] if tools_stat['success_rate'] is not None else '未触发'} |",
        "\n意图误判明细：" + (json.dumps(intents["mistakes"], ensure_ascii=False)
                            if intents["mistakes"] else "无"),
    ]
    if e2e:
        lines += [
            "\n## 4. 端到端场景评测（多轮对话脚本，逐检查点断言）",
            f"\n**端到端任务完成率：{e2e['completion_rate']}**（整场景全过为完成）"
            f"　检查点通过率：{e2e['check_pass_rate']}",
            "\n| 场景 | 轮数 | 检查点 | 通过 |",
            "|---|---|---|---|",
        ]
        for r in e2e["rows"]:
            lines.append(f"| {r['name']} | {r['turns']} | {r['ok']}/{r['all']} | "
                         f"{'✓' if r['pass'] else '✗'} |")
        if any(r["fail_detail"] for r in e2e["rows"]):
            lines.append("\n失败明细：")
            for r in e2e["rows"]:
                for d in r["fail_detail"]:
                    lines.append(f"- [{r['name']}] {d}")
    if gen:
        lines += [
            f"\n## 5. RAG 生成质量（RAGAS 风格，n={gen['n']}，LLM-as-judge：deepseek-chat）",
            "\n| 指标 | 含义 | 数值 |",
            "|---|---|---|",
            f"| Faithfulness 忠实度 | 答案论断被上下文支撑的程度（1=大量编造，5=全部可溯源，归一化 0~1） | {gen['faithfulness']} |",
            f"| Answer Relevancy 答案相关性 | 从答案反推问题与原问题的 embedding 余弦（RAGAS 官方口径，MiniLM） | {gen['answer_relevancy']} |",
            f"| Context Relevance 上下文相关性 | 召回 chunk 中含回答所需信息的比例 | {gen['context_relevance']} |",
            "\n> 注：三指标均为 LLM-as-judge / embedding 余弦，judge 与被评模型同源（deepseek-chat）"
            "存在自评偏好，数字用于横向对比与回归监测，不宜绝对化解读。",
        ]
    lines += [
        "\n## 局限（诚实记录，面试可讲）",
        "- 检索命中口径为菜品名字面匹配，未做语义相关性人工抽检。",
        "- 约束校验为词表词边界匹配，仍有语义级漏检（描述不含任何食材词时无法拦截）。",
        "- 规划评测仅 6 个固定请求、端到端仅 3 个场景，样本小；temperature=0 下"
        "LLM 仍有微小非确定性（尤其编辑重建整份计划），单次结果以复现为准。",
        "- RAGAS 风格三指标依赖 LLM-as-judge，judge 与生成模型同源（deepseek-chat），存在自评偏差。",
    ]
    report = RESULTS_DIR / "eval_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已生成 {report}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
