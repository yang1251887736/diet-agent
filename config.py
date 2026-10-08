"""全局配置：资产路径 + 模型参数 + 检索/Agent 开关。"""
from pathlib import Path
import os
from dotenv import load_dotenv

# 两个模型（MiniLM 在 HF 缓存、bge-reranker 在 models/ 本地目录）都已在本地，
# 强制离线模式：跳过 HuggingFace 连网检查——本机网络环境下该检查会
# WinError 10060 卡住一两分钟甚至 502。若日后要下载新模型，临时去掉这两行。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

load_dotenv()

# ── RAG 资产（已并入本仓库，clone 即可运行，无外部依赖） ─────────────
# 来源：本人的地中海饮食 RAG 问答系统（616 篇语料 → 1650 chunks，
# all-MiniLM-L6-v2 向量化 + FAISS）。构建索引的完整流水线（3 种切分 ×
# 3 种向量化 × 4 路检索 × 3 种 prompt 对比实验）见原工程：
# E:\Study\Text_mining\Agent and RAG project\Mediterranean_RAG_Deliverables
BASE_DIR = Path(__file__).parent
CHUNKS_CSV = BASE_DIR / "data" / "index" / "chunks.csv"        # 1650 chunks
FAISS_INDEX = BASE_DIR / "data" / "index" / "faiss_st.index"   # 与 MiniLM 配套的扁平索引
QA_DIR = BASE_DIR / "qa"                                       # 210 组评测题（3 个来源）

# ── 模型 ────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"        # 与你旧索引一致，不能换
# 精排模型指向本地目录（scripts/download_reranker.py 下载）。
# 若目录不存在，rerank 会自动降级为原排序并提示。
RERANK_MODEL = str(Path(__file__).parent / "models" / "bge-reranker-base")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ── 检索优化开关（evaluate 会按三模式对比） ─────────────────────
USE_REWRITE = True    # Query Rewrite（LLM 改写，需 API Key）
USE_RERANK = True     # Cross-Encoder 精排（本地，首次下载模型）
RERANK_CANDIDATES = 20  # 精排前的召回条数
CACHE_TTL = 3600        # rerank 结果缓存秒数（进程内 TTL，替代 Redis）

# ── Agent 参数 ──────────────────────────────────────────
TOP_K = 12                # 最终进入候选池的 chunk 数
MIN_POOL = 4              # 候选池覆盖分级阈值：>=4 为 A，1~3 为 B，0 为 C
MAX_RETRIES = 3           # validate → planner 回流步数上限（防死循环）
SUMMARY_CHARS = 400       # 候选池里每道菜的描述截断长度
RESULTS_DIR = Path(__file__).parent / "results"
