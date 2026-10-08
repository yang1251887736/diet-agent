"""LangGraph 状态图：意图识别 → 候选池检索 → Planner(DeepSeek, ID 约束) → 校验 → 条件回流。

六个设计决策（面试必考，逐条能讲）：
1. 意图识别入口：LLM 把输入分类为 new_plan / edit / chat，条件边自动路由；
   edit 自动取最近一次历史计划作为修改基底（长期记忆的第一种用法）。
2. ID 约束防幻觉：Planner 的 prompt 注入候选池并声明"只准用池内 dish_id"，
   validate_node 再用代码复核 —— prompt 引导 + 代码兜底，双层防线。
3. Planner 绑 lookup_dish 工具：对某道菜不确定时自己决定查语料（tool calling），
   查到的是语料真实数据而非模型记忆；所有展示数据都从语料回填。
4. 回流带反馈：违规信息拼进下一轮 prompt，模型改得有的放矢；
   MAX_RETRIES 防死循环（状态机必须有出口）。
5. 长期记忆：偏好从 runs 历史推导后注入 prompt（profile.build_hint），
   另配 SqliteSaver Checkpointer 做线程级断点持久化。
6. 编辑模式：prev_plan + edit_instruction 非空时，Planner 改为
   "按指令修改既有计划"——同一张图服务两个入口。
"""
import json
import re
import sqlite3
import time
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from config import (DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
                    TOP_K, MAX_RETRIES, SUMMARY_CHARS,
                    USE_REWRITE, USE_RERANK, MIN_POOL, RESULTS_DIR)
from rag import retrieve_v2
from tools import lookup_dish, check_constraints
from data.constraints import normalize_prefs, normalize_dislikes
from usage import acc, STATS
from user_profile import build_hint
import store


class DietState(TypedDict):
    user_request: str
    diet_prefs: list
    dislikes: list
    likes: list           # 用户喜好（"喜欢吃鸡肉"→["chicken"]），检索/规划都要用
    intent: str           # new_plan / edit / replan / qa / plan_qa / chat
    memory_hint: str      # 长期记忆注入的偏好提示
    candidate_pool: list
    coverage: str          # 语料覆盖分级 A/B/C（A 充足 / B 有限 / C 无匹配）
    plan: dict
    plan_zh: dict          # 中文展示层（翻译节点输出，只进展示不进校验链）
    rationale_zh: str      # 安排理由的中文版（planner 产出 rationale，翻译节点转译）
    validation: dict
    answer: str            # qa / plan_qa 意图时问答专家的中文回答
    retries: int
    prev_plan: dict        # 编辑模式：既有计划
    edit_instruction: str  # 编辑模式：自然语言修改指令
    plan_days: int         # replan 时从历史继承的天数（当前话没提天数才用）


TOKEN_STATS = STATS     # 供 run_cli / evaluate 读取


def understand_node(state: DietState) -> dict:
    """理解节点（多轮对话唯一入口）：一次 LLM 调用同时完成意图分类 + 约束抽取。

    六路意图（路由到四个专家节点）：
    - new_plan → 规划专家（retrieve→planner→validate→translate）
    - edit     → 编辑专家（同一张规划链，prev_plan 作为修改基底）
    - replan   → 规划专家，但约束先与上次 run 合并
                 （"忘了说了我不吃辣" = 上次偏好 ∪ 辣，重新规划）
    - qa       → 语料问答专家（RAG 回答菜品/食材问题）
    - plan_qa  → 菜单问答专家（问"你给我的哪些有鸡肉"——答案必须来自
                 最近一次计划，而不是整个语料库；这是真实翻车教训）
    - chat     → 直接结束（CLI 给固定应答）

    约束抽取让用户一句话说完所有需求（"3天素食不吃辣"），不再分两行问。
    likes（喜好）与 prefs（限制）是两个方向：前者要"多给"，后者要"拦截"。
    """
    if state.get("edit_instruction") and state.get("prev_plan"):
        return {"intent": "edit"}   # CLI 强制编辑入口，跳过理解
    latest = store.latest_run()
    u = _understand(state["user_request"], has_history=latest is not None)
    intent = u.get("intent", "new_plan")
    prefs = normalize_prefs(u.get("diet_prefs") or [])
    dislikes = normalize_dislikes(u.get("dislikes") or [])
    likes = [w for w in (u.get("likes") or []) if w]
    out = {"intent": intent,
           "diet_prefs": prefs or state.get("diet_prefs") or [],
           "dislikes": dislikes or state.get("dislikes") or [],
           "likes": likes}
    if intent in ("edit", "replan") and latest:
        # 喜好跨轮延续：编辑/重规划时把历史喜好带上（"别老吃柠檬鸡"暗含
        # "还是爱吃鸡"，不能因为这句话没提鸡就把喜好丢了）
        out["likes"] = list(dict.fromkeys(
            (latest.get("likes") or []) + likes))
    if intent == "edit":
        if latest:
            out["prev_plan"] = latest["plan"]
            out["edit_instruction"] = state["user_request"]
        else:
            out["intent"] = "new_plan"
    elif intent == "replan":
        # 约束合并：上次偏好 ∪ 本次新增，走全新规划链
        merged_p = normalize_prefs(list(dict.fromkeys(
            (latest["prefs"] if latest else []) + prefs)))
        merged_d = normalize_dislikes(list(dict.fromkeys(
            (latest["dislikes"] if latest else []) + dislikes)))
        out.update({"intent": "new_plan", "diet_prefs": merged_p, "dislikes": merged_d})
        # 天数延续（端到端评测抓出的真 bug）："忘了说了我不吃辣"没提天数，
        # 直接落到默认 3 天会推翻用户之前"两天"的要求——从历史请求继承
        if _extract_days(state["user_request"], default=None) is None and latest:
            prev_days = _extract_days(latest.get("request") or "", default=None)
            if prev_days:
                out["plan_days"] = prev_days
    hint = build_hint()
    if hint:
        out["memory_hint"] = hint
    return out


def _understand(request: str, has_history: bool) -> dict:
    try:
        llm = ChatOpenAI(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY,
                         base_url=DEEPSEEK_BASE_URL, temperature=0)
        resp = llm.invoke(
            "You are the intent router of a Mediterranean meal-planning assistant.\n"
            "Classify the user utterance AND extract dietary constraints.\n"
            "Intents:\n"
            '- new_plan: asks for a meal plan (e.g. "3天素食菜单", "plan 3 days")\n'
            '- edit: wants to MODIFY the previous plan structure (e.g. "把第2天晚餐换成鱼")\n'
            '- replan: adds or changes dietary constraints without referencing plan '
            'structure (e.g. "忘了说了我不吃辣", "改成纯素")\n'
            '- qa: asks about a dish / ingredient / cuisine in general '
            '(e.g. "咖喱是什么菜", "what is moussaka")\n'
            '- plan_qa: asks about the dishes IN the current/latest plan '
            '(e.g. "你给我的哪些有鸡肉", "计划里哪天有鱼")\n'
            '- chat: greeting / small talk / not related\n'
            + ("A previous plan exists in history.\n" if has_history
               else "No history exists: edit, replan and plan_qa are impossible.\n")
            + "Extract diet_prefs from: vegan, vegetarian, pescatarian, halal, "
              "素食, 纯素, 全素, 清真, 鱼素. dislikes: free words, keep original language.\n"
            "Extract likes: foods the user explicitly ENJOYS (e.g. 喜欢吃鸡肉 → "
            '["chicken"], 爱吃海鲜 → ["seafood"]). Only explicit enjoyment, '
            "not mere mention. English words.\n"
            'Return STRICT JSON only:\n'
            '{"intent": "...", "diet_prefs": [], "dislikes": [], "likes": []}\n\n'
            "User: " + request
        )
        acc(resp)
        return _parse_json(resp.content) or {}
    except Exception:
        return {"intent": "new_plan"}   # 理解节点挂了不能挡住主流程


# 偏好激活时补充的英文语义查询（中文原句的向量信号弱，实测只能捞到 2 道合规菜）
PREF_QUERY_TPL = {
    "vegetarian": "vegetarian Mediterranean dishes, soups, salads, pasta, legumes",
    "vegan": "vegan Mediterranean dishes, vegetable salads, legumes, tomato sauces",
    "pescatarian": "Mediterranean fish and seafood dishes",
    "halal": "halal-friendly Mediterranean dishes without pork and alcohol",
}
POOL_MAX = 15          # 过滤后候选池上限（太多会撑爆 planner prompt）
POOL_RAW_K = 40        # 过滤前的召回量（过滤会砍掉一部分，得留余量）


def retrieve_node(state: DietState) -> dict:
    """多路检索 → 按 dish 去重 → 约束过滤 → 候选池 + 覆盖分级（池子全程稳定）。

    两个教训（都来自真实翻车）：
    - 约束过滤必须在检索层：曾出现"素食请求、候选池全是海鲜"——
      检索只用原句，偏好信息根本没参与召回，planner 无菜可选。
    - 中文原句直接做向量检索信号很弱："给我三天的菜单 vegetarian"
      只能召回 2 道合规菜 → 偏好激活时补一路英文语义查询（k=40）再合并。
    - 过滤与校验必须同文本源：过滤原本用"被召回的那 1 个 chunk"，校验用
      lookup_dish 的"前 2 个 chunk 拼接"——某菜没被召回的 chunk 里有 wine
      时，会出现"过滤放行→校验必挂→打回也换不出"的收敛死局（halal 评测
      final_ok 0.8333 的真凶）。修复：过滤统一走 lookup_dish 同一文本源。
    """
    prefs = normalize_prefs(state.get("diet_prefs"))
    dislikes = normalize_dislikes(state.get("dislikes"))
    likes = state.get("likes") or []
    # 查询顺序有讲究：喜好/偏好的语义查询必须在原句【前面】——池上限 15，
    # 原句（中文弱信号）先跑就会用随机结果占满池子，轮到喜好查询时已无席
    # （实测：柠檬鸡独占鸡菜席，Chicken Picasso 根本进不来）。
    queries = [PREF_QUERY_TPL[p] for p in prefs if p in PREF_QUERY_TPL]
    queries += [f"{lk} dishes, Mediterranean cuisine" for lk in likes]
    queries.append(state["user_request"])
    raw_k = POOL_RAW_K if (prefs or dislikes or likes) else TOP_K
    pool, seen, seen_names = [], set(), set()
    for q in queries:
        hits = retrieve_v2(q, k=raw_k, use_rewrite=USE_REWRITE, use_rerank=USE_RERANK)
        for h in hits:
            did = h["chunk_id"].rsplit("_chunk_", 1)[0]
            if did in seen or len(pool) >= POOL_MAX:
                continue
            if h["dish"].lower() in seen_names:
                continue   # 同名不同 ID 的重名菜只留一个（Lemon chicken 双席位教训）
            seen.add(did)
            info = lookup_dish.invoke({"dish_id": did})   # 与 validate_node 同一文本源
            if "error" in info:
                continue
            r = check_constraints.invoke({
                "dish_text": f"{info['dish']} {info['summary']}",
                "diet_prefs": prefs, "dislikes": dislikes,
            })
            if not r["ok"]:
                continue     # 违规候选在源头剔除
            seen_names.add(info["dish"].lower())
            pool.append({
                "dish_id": did,
                "dish": info["dish"],
                "summary": info["summary"][:SUMMARY_CHARS],
            })
    if state.get("edit_instruction") and state.get("prev_plan"):
        pool = _merge_prev_plan_pool(pool, seen, state["prev_plan"])
    coverage = "A" if len(pool) >= MIN_POOL else ("B" if pool else "C")
    return {"candidate_pool": pool, "coverage": coverage}


def _merge_prev_plan_pool(pool: list, seen: set, prev_plan: dict) -> list:
    """编辑模式：把既有计划里的菜并入候选池。

    否则重检索产生的新池不包含原计划的菜，validate 会把原菜判为
    "池外 ID"，导致编辑永远无法收敛（冒烟测试抓到的真实 bug）。
    """
    for day in prev_plan.get("days", []):
        for meal in day.get("meals", []):
            did = meal.get("dish_id", "")
            if not did or did in seen:
                continue
            info = lookup_dish.invoke({"dish_id": did})
            if "error" in info:      # 旧计划里的菜已不在语料（理论不会发生），跳过
                continue
            seen.add(did)
            pool.append({"dish_id": did, "dish": info["dish"],
                         "summary": info["summary"][:SUMMARY_CHARS]})
    return pool


def _llm_invoke_retry(llm, messages, attempts: int = 3):
    """LLM 调用带指数退避重试（1s/2s/4s）。

    评测时单次要打 200+ 个 LLM 调用，长跑中会遇到瞬时连接抖动
    （OpenAIConnectionError）——不重试的话整个评测中途报废。
    """
    last = None
    for i in range(attempts):
        try:
            return llm.invoke(messages)
        except Exception as e:          # noqa: BLE001 网络类异常统一重试
            last = e
            if i < attempts - 1:
                time.sleep(2 ** i)
    raise last


def planner_node(state: DietState) -> dict:
    llm = _planner_llm()
    messages = _planner_messages(state)
    resp = _llm_invoke_retry(llm, messages)
    acc(resp)

    # mini tool-calling 循环：模型要查菜品就让它查，最多 4 轮
    for _ in range(4):
        if not resp.tool_calls:
            break
        messages.append(resp)
        for tc in resp.tool_calls:
            result = lookup_dish.invoke(tc["args"])
            messages.append(ToolMessage(content=json.dumps(result, ensure_ascii=False),
                                        tool_call_id=tc["id"]))
        resp = _llm_invoke_retry(llm, messages)
        acc(resp)

    plan = _parse_json(resp.content)
    if plan.get("days") and state.get("edit_instruction") and state.get("prev_plan"):
        plan = _merge_edit(plan, state["prev_plan"], state["edit_instruction"])
    return {"plan": plan, "retries": state.get("retries", 0) + 1}


def validate_node(state: DietState) -> dict:
    """纯代码校验：① dish_id 在池内 ② 每道菜过约束校验 ③ 计划结构完整。

    ③是补出来的：曾出现"Day 2/Day 3 整个为空也判全部通过"——校验只看
    已存在的菜、不查缺失，不完整的计划必须回流重排。
    """
    pool_by_id = {c["dish_id"]: c for c in state["candidate_pool"]}
    invalid_ids, violations = [], []
    issues = []

    days = state.get("plan", {}).get("days", [])
    if not days:
        return {"validation": {"invalid_ids": [], "violations": [],
                               "issues": ["Plan is empty or unparseable. Output STRICT JSON."]}}

    # ③ 结构完整性：天数够不够、每天三餐齐不齐
    required_days = _extract_days(state["user_request"])
    required_slots = {"breakfast", "lunch", "dinner"}
    if len(days) < required_days:
        issues.append(f"Plan has only {len(days)} day(s); {required_days} required. Add the missing days.")
    for di, day in enumerate(days, 1):
        got_slots = {m.get("slot", "") for m in day.get("meals", [])}
        missing = required_slots - got_slots
        if missing:
            issues.append(f"Day {di} is missing meals: {', '.join(sorted(missing))}. Add them.")

    # ④ 跨天重复（条件启用）：planner 曾一边喊着"不重复"一边把柠檬鸡排了两天。
    # 只在池子足够大（>= 总餐位数）时检查——小池子本来就得重复，检查只会
    # 白烧 3 轮回流。满足条件下重复才算 issue，打回重排。
    total_slots = len(days) * len(required_slots)
    if len(state["candidate_pool"]) >= total_slots and len(days) >= 2:
        usage = {}
        for di, day in enumerate(days, 1):
            for meal in day.get("meals", []):
                usage.setdefault(meal.get("dish_id", ""), []).append(di)
        for did, ds in usage.items():
            if len(ds) > 1:
                issues.append(f"Dish {did} is used on days {ds}. Do not repeat the same "
                              "dish across days — pick a different one from the pool.")

    # ① ② 逐菜校验
    for di, day in enumerate(days, 1):
        for meal in day.get("meals", []):
            did = meal.get("dish_id", "")
            if did not in pool_by_id:
                invalid_ids.append({"day": di, "dish_id": did})
                continue
            info = lookup_dish.invoke({"dish_id": did})   # 语料回填，不信模型
            r = check_constraints.invoke({
                "dish_text": f"{info.get('dish', '')} {info.get('summary', '')}",
                "diet_prefs": state["diet_prefs"],
                "dislikes": state["dislikes"],
            })
            if not r["ok"]:
                violations.append({"day": di, "dish_id": did, "detail": r["violations"]})

    if invalid_ids:
        issues.append(f"Invalid dish_id not in pool: {invalid_ids}. Use ONLY pool ids.")
    for v in violations:
        issues.append(f"Day {v['day']} dish {v['dish_id']} violates {v['detail']}. Replace it.")
    return {"validation": {"invalid_ids": invalid_ids, "violations": violations, "issues": issues}}


def route_after_validate(state: DietState, max_retries: int) -> str:
    v = state.get("validation", {})
    failed = bool(v.get("invalid_ids") or v.get("violations") or v.get("issues"))
    if failed and state.get("retries", 0) < max_retries:
        return "planner"
    return "translate"   # 合格→翻译输出；步数用尽仍不合格→带问题清单照常输出


def translate_node(state: DietState) -> dict:
    """输出层中文化：把计划里的菜名/简介/安排理由译成中文，写入 plan_zh / rationale_zh。

    设计要点（面试可讲）：翻译只进【展示层】，不进校验链——
    - 校验永远基于英文原文（词表匹配的是语料原文）；
    - 翻译错误最多影响观感，不可能污染约束判定；
    - 用独立节点而非绑定 planner 的工具：翻译必须 100% 发生，
      工具是模型"按需调用"的，保证不了这一点。
    """
    plan = state.get("plan", {})
    ids = sorted({m.get("dish_id", "") for d in plan.get("days", [])
                  for m in d.get("meals", []) if m.get("dish_id")})
    if not ids:
        return {"plan_zh": {}, "rationale_zh": ""}
    payload = []
    for did in ids:
        info = lookup_dish.invoke({"dish_id": did})
        if "error" not in info:
            payload.append({"dish_id": did, "dish": info["dish"],
                            "summary": info["summary"][:300]})
    if not payload:
        return {"plan_zh": {}, "rationale_zh": ""}
    rationale = (plan.get("rationale") or "").strip()
    llm = ChatOpenAI(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY,
                     base_url=DEEPSEEK_BASE_URL, temperature=0)
    resp = _llm_invoke_retry(llm, [
        SystemMessage(content=(
            "Translate dish names and summaries into Simplified Chinese. "
            "Keep culinary terms natural (e.g. use the accepted Chinese name). "
            'Return STRICT JSON only: {"dishes": {"<dish_id>": {"name": "...", '
            '"summary": "..."}}, "rationale": "<the rationale in Chinese>"}. '
            "One dishes entry per dish_id, each summary under 60 Chinese characters. "
            "If no rationale is provided, set rationale to an empty string.")),
        HumanMessage(content=json.dumps(
            {"dishes": payload, "rationale": rationale}, ensure_ascii=False)),
    ])
    acc(resp)
    out = _parse_json(resp.content) or {}
    return {"plan_zh": out.get("dishes") or {},
            "rationale_zh": (out.get("rationale") or "").strip()}


def plan_qa_node(state: DietState) -> dict:
    """菜单问答专家：回答"关于当前/最近一次计划"的问题（真实翻车教训：

    "你给我的哪些有鸡肉"曾被当成普通语料问答，列了一堆计划外的菜。
    这里把最近计划的每道菜（含日期/餐次/语料信息）作为唯一上下文，
    答案必须限定在计划内——问计划答计划，不串台到语料库。"""
    latest = store.latest_run()
    if not latest or not (latest.get("plan") or {}).get("days"):
        return {"intent": "plan_qa",
                "answer": "我还没有给你排过菜单。先说一句需求（比如\"给我三天的素食菜单\"），"
                          "排完之后就可以问我关于这份菜单的问题了。"}
    lines = []
    for di, day in enumerate(latest["plan"].get("days", []), 1):
        for m in day.get("meals", []):
            did = m.get("dish_id", "")
            info = lookup_dish.invoke({"dish_id": did})
            desc = "" if "error" in info else f" | {info['dish']} | {info['summary'][:200]}"
            lines.append(f"Day {di} {m.get('slot', '?')}: {did}{desc}")
    llm = ChatOpenAI(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY,
                     base_url=DEEPSEEK_BASE_URL, temperature=0)
    resp = _llm_invoke_retry(llm, [
        SystemMessage(content=(
            "Answer questions about the user's CURRENT meal plan using ONLY the "
            "listed plan dishes. Reference days/meals explicitly (e.g. 第2天晚餐). "
            "Answer in Simplified Chinese; dish names as 中文名（English original）. "
            "If the answer is not derivable from the plan, say so. "
            "Keep it under 150 Chinese characters.")),
        HumanMessage(content=f"Current plan:\n" + "\n".join(lines)
                     + f"\n\nQuestion: {state['user_request']}"),
    ])
    acc(resp)
    return {"intent": "plan_qa", "answer": (resp.content or "").strip()}


def build_graph(max_retries: int | None = None,
                with_intent: bool = True, use_checkpointer: bool = True,
                translate: bool = True):
    """with_intent=False 用于评测（固定走规划链，不花意图分类的钱）；
    use_checkpointer=True 时挂 SqliteSaver，invoke 需带 thread_id；
    translate=False 跳过中文化节点（评测不需要展示层翻译）。"""
    g = StateGraph(DietState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("planner", planner_node)
    g.add_node("validate", validate_node)
    if translate:
        g.add_node("translate", translate_node)
    if with_intent:
        g.add_node("qa", qa_node)
        g.add_node("plan_qa", plan_qa_node)
        g.add_node("intent", understand_node)
        g.set_entry_point("intent")
        g.add_conditional_edges("intent", route_after_intent,
                                {"retrieve": "retrieve", "qa": "qa",
                                 "plan_qa": "plan_qa", END: END})
    else:
        g.set_entry_point("retrieve")
    g.add_edge("retrieve", "planner")
    g.add_edge("planner", "validate")
    after_validate = {"planner": "planner",
                      "translate": "translate" if translate else END}
    g.add_conditional_edges(
        "validate",
        lambda s: route_after_validate(s, MAX_RETRIES if max_retries is None else max_retries),
        after_validate,
    )
    if translate:
        g.add_edge("translate", END)
    return g.compile(checkpointer=_checkpointer() if use_checkpointer else None)


# ── 内部辅助 ────────────────────────────────────────────
def route_after_intent(state: DietState) -> str:
    special = {"qa": "qa", "plan_qa": "plan_qa"}
    return special.get(state.get("intent"),
                       END if state.get("intent") == "chat" else "retrieve")


def _to_english_query(question: str) -> str:
    """把（可能是中文的）问题译成英文用于检索；失败退回原文。

    embedding 模型是纯英文的（all-MiniLM-L6-v2），中文查询召回质量差。
    翻译只影响检索键，不影响用户看到的问题原文。"""
    if question.isascii():
        return question
    try:
        llm = ChatOpenAI(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY,
                         base_url=DEEPSEEK_BASE_URL, temperature=0)
        resp = _llm_invoke_retry(llm, [
            SystemMessage(content="Translate the user's question into English for "
                                  "retrieval. Output ONLY the English translation."),
            HumanMessage(content=question),
        ])
        acc(resp)
        out = (resp.content or "").strip()
        return out if out else question
    except Exception:
        return question


def qa_node(state: DietState) -> dict:
    """语料问答专家：RAG 检索 → 中文回答，答案必须来自语料。

    和规划链共用同一套检索基座（retrieve_v2），但出口完全不同：
    规划输出结构化计划（受 ID 约束），问答输出自然语言（受"只准引用语料"约束）。
    这就是路由 + 专家节点的 multi-agent 形态：入口统一理解，专家各管各的。

    跨语言检索修复（真实 bug）：MiniLM 是纯英文 embedding，中文问题直接
    检索等于语义盲查——"披萨是什么做的"捞回 Menemen/Döner，菜单里刚排过
    的 Pizza 反而查不到。修复：检索前先把问题译成英文（失败退回原查询，
    不能因翻译挂掉而挡住问答主流程）。"""
    q_en = _to_english_query(state["user_request"])
    hits = retrieve_v2(q_en, k=6,
                       use_rewrite=USE_REWRITE, use_rerank=USE_RERANK)
    if not hits:
        return {"intent": "qa",
                "answer": "语料库里没有找到与这个问题相关的内容。这是个地中海饮食语料库，"
                          "试试问某道地中海菜品或食材。"}
    ctx = "\n\n".join(f"[{h['dish']}] {h['text'][:500]}" for h in hits)
    llm = ChatOpenAI(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY,
                     base_url=DEEPSEEK_BASE_URL, temperature=0)
    resp = _llm_invoke_retry(llm, [
        SystemMessage(content=(
            "Answer the user's question about Mediterranean cuisine using ONLY the "
            "provided context. Answer in Simplified Chinese; dish names write as "
            "中文名（English original）. If the context does not contain the answer, "
            "say so honestly instead of guessing (e.g. 语料中没有关于这道菜的记载). "
            "Keep it under 150 Chinese characters.")),
        HumanMessage(content=f"Context:\n{ctx}\n\nQuestion: {state['user_request']}"),
    ])
    acc(resp)
    return {"intent": "qa", "answer": (resp.content or "").strip()}


def _checkpointer():
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        RESULTS_DIR.mkdir(exist_ok=True)
        conn = sqlite3.connect(str(RESULTS_DIR / "checkpoints.sqlite"),
                               check_same_thread=False)
        return SqliteSaver(conn)
    except Exception as e:
        print(f"[checkpointer] 不可用，降级为无持久化：{e}")
        return None


def _planner_llm():
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("请先复制 .env.example 为 .env 并填入 DEEPSEEK_API_KEY")
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0,   # 可复现，评测的前提
    ).bind_tools([lookup_dish])


def _planner_messages(state: DietState) -> list:
    pool_lines = "\n".join(
        f"- {c['dish_id']} | {c['dish']} | {c['summary'][:200]}"
        for c in state["candidate_pool"]
    )
    prefs = ", ".join(normalize_prefs(state["diet_prefs"])) or "none"
    dislikes = ", ".join(normalize_dislikes(state["dislikes"])) or "none"
    likes = ", ".join(state.get("likes") or []) or "none"
    days = state.get("plan_days") or _extract_days(state["user_request"])

    if state.get("edit_instruction") and state.get("prev_plan"):
        task = ("You will UPDATE an existing meal plan according to an edit instruction. "
                "Keep the overall structure (days/meals) unless the edit says otherwise. "
                "The user's likes still apply: replace a disliked/repetitive dish with a "
                "DIFFERENT dish they like from the pool, not with something unrelated.")
    else:
        task = (f"Plan meals for exactly {days} day(s), "
                "and EVERY day must have all 3 meals (breakfast, lunch, dinner). "
                "Never leave a day or a meal empty.")

    system = (
        "You are a Mediterranean diet meal planner.\n"
        f"{task}\n"
        "STRICT RULES:\n"
        "1. Use ONLY dish_id values from the candidate pool. Any other id is INVALID.\n"
        "2. Respect the dietary preferences and dislikes strictly.\n"
        "3. The user's LIKES are a positive requirement: actively include dishes "
        "matching them, spread across days, and VARY the preparations — never repeat "
        "the same dish on more than one day.\n"
        "4. If unsure about a dish, call the lookup_dish tool to check the corpus.\n"
        '5. Final answer = STRICT JSON only, no markdown, exact schema:\n'
        '{"days": [{"meals": [{"dish_id": "...", "slot": "breakfast|lunch|dinner"}]}], '
        '"rationale": "2-4 sentences: how the arrangement serves the user\'s request, '
        'likes, constraints and variety. Justify the key choices."}\n'
    )
    user = (f"User request: {state['user_request']}\n"
            f"Dietary preferences (hard restrictions): {prefs}\n"
            f"Dislikes (hard restrictions): {dislikes}\n"
            f"Likes (try to include): {likes}\n\n"
            f"Candidate pool:\n{pool_lines}")
    if state.get("memory_hint"):
        user += (f"\n\n{state['memory_hint']}\n"
                 "IMPORTANT: the CURRENT user request always overrides this history hint.")
    if state.get("edit_instruction") and state.get("prev_plan"):
        user += ("\n\nCurrent plan JSON:\n"
                 + json.dumps(state["prev_plan"], ensure_ascii=False)
                 + f"\n\nEdit instruction: {state['edit_instruction']}")
    if state.get("coverage") == "B":
        user += "\n\nNote: corpus coverage for this request is limited; choose the best matches from the pool."
    if not state["candidate_pool"]:
        user += '\n\nThe candidate pool is EMPTY. Respond with {"days": []} and nothing else.'
    if state.get("validation", {}).get("issues"):
        user += ("\n\nYour previous plan was REJECTED. Fix these issues and resubmit:\n- "
                 + "\n- ".join(state["validation"]["issues"]))
    return [SystemMessage(content=system), HumanMessage(content=user)]


def _extract_days(request: str, default: int | None = 3) -> int | None:
    """天数提取 v3：数字/英文数词/中文数词/周表达/workdays/weekend。

    顺序有讲究：workdays 必须先于 week（"a week of workdays" 是 5 天不是 7 天），
    weekend 先于 week（否则 \\bweek\\b 会先命中 weekend 的前缀）。
    无法识别时返回 default（默认 3 天，不阻塞主流程；传 None 可区分
    "用户没提天数"——replan 靠它决定是否从历史继承）。
    """
    t = re.sub(r"\s+", " ", request.lower())
    m = re.search(r"(\d+)\s*(?:天|days?\b|day\b)", t)
    if m:
        return min(max(int(m.group(1)), 1), 7)
    if re.search(r"\b(?:work ?days?|week ?days?|working week)\b", t):
        return 5
    if re.search(r"周末|weekend", t):
        return 2
    if re.search(r"一周|(?:a|one|this|next)\s+week\b|\bweek\b", t):
        return 7
    cn = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7}
    m = re.search(r"([一二两三四五六七])\s*天", t)
    if m:
        return cn[m.group(1)]
    for w, n in (("one", 1), ("two", 2), ("three", 3), ("four", 4),
                 ("five", 5), ("six", 6), ("seven", 7)):
        if re.search(rf"\b{w}\s+(?:天|days?\b)", t):
            return n
    return default


if __name__ == "__main__":
    pass   # 自测在文件末尾（依赖本文件全部函数定义）


def _parse_json(text: str) -> dict:
    """容错解析：截取首个 { 到最后一个 }；解析失败返回空 dict（交给 validate 驳回）。"""
    try:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return {}
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


# ── 编辑餐位保护（端到端评测抓出的真 bug 的代码级修复） ─────────
def _parse_edit_targets(instruction: str) -> set:
    """解析编辑指令针对的餐位 {(day_idx|None, slot|None)}。

    "把第一天的晚餐换成鱼" → {(0, 'dinner')}；
    "晚餐换花样"（没提天）→ {(None, 'dinner')} = 所有天的晚餐；
    "换个花样"（天和餐次都定位不了）→ 空集 = 宽泛编辑，不套保护。"""
    t = instruction.lower()
    slot_words = [("breakfast", ("breakfast", "早餐")),
                  ("lunch", ("lunch", "午餐")),
                  ("dinner", ("dinner", "supper", "晚餐", "晚饭"))]
    slots = [s for s, ws in slot_words if any(w in t for w in ws)]
    days = []
    cn = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7}
    for m in re.finditer(r"第\s*([一二两三四五六七\d]+)\s*天", instruction):
        tok = m.group(1)
        days.append((int(tok) if tok.isdigit() else cn.get(tok, 0)) - 1)
    for m in re.finditer(r"\bday\s*(\d+)", t):
        days.append(int(m.group(1)) - 1)
    for m in re.finditer(r"\b(\d)(?:st|nd|rd|th)?\s*day\b", t):
        days.append(int(m.group(1)) - 1)
    if not days and not slots:
        return set()
    days = [d for d in days if 0 <= d <= 6] or [None]
    return {(d, s) for d in days for s in (slots or [None])}


def _slot_targeted(di: int, slot: str, targets: set) -> bool:
    return any((d is None or d == di) and (s is None or s == slot)
               for d, s in targets)


def _merge_edit(plan: dict, prev_plan: dict, instruction: str) -> dict:
    """编辑精确性保证：以原计划为基底，只采纳指令针对餐位的新菜，其余强制保留。

    背景：planner 是"重建整份计划"而不是打补丁——端到端评测实测"换第一天的
    晚餐"后 9 个餐位全变。prompt 层约束（"keep the structure"）不可靠，
    与防幻觉同一哲学：prompt 约束 + 代码校验双层。指令定位不到目标餐位时
    不套保护，避免误伤"换个花样"这类宽泛编辑。"""
    targets = _parse_edit_targets(instruction)
    if not targets or not (prev_plan or {}).get("days") or not (plan or {}).get("days"):
        return plan
    merged = json.loads(json.dumps(prev_plan))   # 深拷贝：基底 = 原计划
    new_days = plan.get("days", [])
    for di, day in enumerate(merged["days"]):
        for m in day.get("meals", []):
            if not _slot_targeted(di, m.get("slot", ""), targets):
                continue
            new_meal = next((mm for mm in (new_days[di].get("meals", [])
                                           if di < len(new_days) else [])
                             if mm.get("slot") == m.get("slot")), None)
            if new_meal and new_meal.get("dish_id"):
                m["dish_id"] = new_meal["dish_id"]
    return merged


if __name__ == "__main__":
    # 离线自测（不花 API 钱）：放文件末尾，依赖本文件全部函数定义
    cases = [
        ("Plan 3 days of meals", 3), ("Plan 2 days of vegan meals", 2),
        ("规划 5 天", 5), ("three days of Mediterranean food", 3),
        ("a week of workdays lunches", 5), ("weekend breakfast ideas", 2),
        ("a week of dinners", 7), ("something to eat", 3),
        ("给我三天的菜单", 3), ("来两天素食计划", 2), ("周末吃什么", 2), ("安排一周", 7),
    ]
    for req, want in cases:
        got = _extract_days(req)
        assert got == want, f"{req!r}: want {want}, got {got}"
    assert _extract_days("忘了说了我不吃辣", default=None) is None   # 没提天数
    assert _extract_days("改两天的", default=None) == 2
    print(f"天数提取自测 {len(cases) + 2}/{len(cases) + 2} 通过")

    # 编辑餐位保护自测（端到端评测抓出的 bug 的回归用例）
    prev = {"days": [{"meals": [{"slot": "breakfast", "dish_id": "a1"},
                                {"slot": "lunch", "dish_id": "a2"},
                                {"slot": "dinner", "dish_id": "a3"}]} for _ in range(2)]}
    new = {"days": [{"meals": [{"slot": "breakfast", "dish_id": "b1"},
                               {"slot": "lunch", "dish_id": "b2"},
                               {"slot": "dinner", "dish_id": "b3"}]} for _ in range(2)]}
    m = _merge_edit(new, prev, "把第一天的晚餐换成鱼")
    assert m["days"][0]["meals"][2]["dish_id"] == "b3"    # 目标餐位采纳新菜
    assert m["days"][0]["meals"][0]["dish_id"] == "a1"    # 其余保留原菜
    assert m["days"][1]["meals"][2]["dish_id"] == "a3"    # 第二天晚餐也不动
    m2 = _merge_edit(new, prev, "晚餐换花样")             # 所有天的晚餐都改
    assert all(d["meals"][2]["dish_id"] == "b3" for d in m2["days"])
    assert all(d["meals"][0]["dish_id"] == "a1" for d in m2["days"])
    m3 = _merge_edit(new, prev, "换个花样")               # 定位不了 → 放开全改
    assert all(mm["dish_id"].startswith("b") for d in m3["days"] for mm in d["meals"])
    print("编辑餐位保护自测 3/3 通过")
