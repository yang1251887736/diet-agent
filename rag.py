"""两阶段检索优化：Query Rewrite + Cross-Encoder Rerank。

三个模式（evaluate.py 逐个对比出表，这是简历数字的主要来源）：
- baseline       ：直接向量召回
- rewrite        ：LLM 改写查询后再召回（任何异常自动退回原查询——优化不能比基线脆弱）
- rewrite_rerank ：改写召回 20 条 → 本地 Cross-Encoder 精排取 top-k

⚠️ Cross-Encoder 首次运行会下载 BAAI/bge-reranker-base（约 1.1GB），
   下载失败自动降级为原排序，不会中断流程。
"""
import time

from config import (DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
                    RERANK_MODEL, RERANK_CANDIDATES, CACHE_TTL, TOP_K)
from usage import acc

_cache: dict = {}   # rerank 结果缓存 {key: (timestamp, result)}，替代 Redis 的进程内 TTL 缓存
_reranker = None    # 惰性加载（加载要几秒，别放 import 时）


def rewrite_query(query: str) -> str:
    """LLM 改写：只做消歧（代词/模糊指代→具体词），不扩写。

    ⚠️ 小样本实测教训：让 LLM"补全成具体食材词"会在 QA 式问题上负优化
    （20 题 Top1 15→12）——扩写出来的新词把 embedding 语义带偏。
    改为保守指令（保留全部原词、只消歧）后回到 15/20，与 baseline 持平。
    失败时退回原查询（优化不能比基线脆弱）。
    """
    if not DEEPSEEK_API_KEY:
        return query
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY,
                         base_url=DEEPSEEK_BASE_URL, temperature=0)
        resp = llm.invoke(
            "Rewrite the query for retrieval. STRICT RULES: keep ALL original content "
            "words; only replace pronouns and vague references with concrete terms; "
            "do NOT add ingredients, attributes, or new terms; "
            "output ONLY the rewritten query, nothing else.\n\nQuery: " + query
        )
        acc(resp)
        out = (resp.content or "").strip()
        return out if out else query
    except Exception:
        return query


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def rerank(query: str, chunks: list, top_k: int) -> list:
    """Cross-Encoder 对 (query, chunk) 成对打分重排。带 TTL 缓存。"""
    if not chunks:
        return chunks
    key = (query, tuple(c["chunk_id"] for c in chunks), top_k)
    ts, cached = _cache.get(key, (0.0, None))
    if cached is not None and time.time() - ts < CACHE_TTL:
        return cached
    try:
        scores = _get_reranker().predict([(query, c["text"]) for c in chunks])
    except Exception as e:
        print(f"[rerank] 降级为原排序：{e}")
        return chunks[:top_k]
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    result = [chunks[i] for i in order[:top_k]]
    _cache[key] = (time.time(), result)
    return result


def retrieve_v2(query: str, k: int = TOP_K,
                use_rewrite: bool = False, use_rerank: bool = False) -> list:
    """统一检索入口：召回用改写后的查询，精排用用户原始查询（更忠实意图）。"""
    q = rewrite_query(query) if use_rewrite else query
    hits = _raw_retrieve(q, k=RERANK_CANDIDATES if use_rerank else k)
    if use_rerank:
        hits = rerank(query, hits, k)
    return hits


def _raw_retrieve(query: str, k: int) -> list:
    from retriever import retrieve
    return retrieve(query, k=k)
