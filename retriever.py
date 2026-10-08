"""检索封装（样板，已写好，可直接用）。

职责：加载你现有的 FAISS 索引 + chunks.csv，提供 retrieve(query, k)。
自测：直接运行本文件，应打印出与查询相关的菜谱 chunk。

⚠️ 如果报「索引维度不匹配」或「Invalid index type」：
   把 config.py 的 FAISS_INDEX 换成 faiss_cosine.index 再跑。
"""
import csv

# ⚠️ config 必须先于 sentence_transformers 导入：config 里的
# HF_HUB_OFFLINE 离线开关要在 transformers/huggingface_hub 加载前生效。
from config import CHUNKS_CSV, FAISS_INDEX, EMBED_MODEL, TOP_K

import faiss
from sentence_transformers import SentenceTransformer

_cache: dict = {}


def _load():
    """进程内只加载一次（模型 ~90MB，索引 ~2MB）。"""
    if not _cache:
        model = SentenceTransformer(EMBED_MODEL)
        index = faiss.read_index(str(FAISS_INDEX))
        with open(CHUNKS_CSV, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if index.ntotal != len(rows):
            raise RuntimeError(
                f"索引向量数 {index.ntotal} != chunks 行数 {len(rows)}，索引和 csv 不配套"
            )
        _cache.update(model=model, index=index, rows=rows)
    return _cache["model"], _cache["index"], _cache["rows"]


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """返回 top-k 相关 chunk：[{chunk_id, dish, text, score}]，score 越大越相关。"""
    model, index, rows = _load()
    vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(vec, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        row = rows[int(idx)]
        results.append({
            "chunk_id": row["chunk_id"],
            "dish": row["dish"],
            "text": row["chunk_text"],
            "score": float(score),
        })
    return results


if __name__ == "__main__":
    print(f"索引已加载：{len(_load()[2])} chunks\n")
    for q in [
        "Spanish dish with paprika and garlic",
        "healthy breakfast with yogurt",
        "lentil soup recipe",
    ]:
        print(f"Q: {q}")
        for r in retrieve(q, k=3):
            print(f"  {r['score']:.3f}  {r['dish']:<20} {r['text'][:60]}...")
        print()
