"""手动下载 bge-reranker-base 到 models/ 目录（绕过 huggingface_hub 下载器）。

背景：本机沙箱代理环境下 huggingface_hub 会写出 0 字节文件
（curl / requests 流式下载均正常），故改用 requests 逐文件下载，
config.RERANK_MODEL 直接指向本地路径，CrossEncoder 支持离线加载。
"""
import sys
from pathlib import Path

import requests

REPO = "BAAI/bge-reranker-base"
DEST = Path(__file__).resolve().parent.parent / "models" / "bge-reranker-base"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
# 模型权重优先 safetensors，404 则退回 pytorch_model.bin；
# xlm-roberta 分词器需要 sentencepiece.bpe.model
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "sentencepiece.bpe.model"]
WEIGHT_CANDIDATES = ["model.safetensors", "pytorch_model.bin"]


def fetch(name: str) -> None:
    url = f"{BASE}/{name}"
    dest = DEST / name
    head = requests.head(url, allow_redirects=True, timeout=30)
    if head.status_code == 404:
        print(f"[skip] {name} 仓库中不存在")
        return
    head.raise_for_status()
    total = int(head.headers.get("content-length", 0))
    if dest.exists() and total and dest.stat().st_size == total:
        print(f"[skip] {name} 已存在（{dest.stat().st_size:,} 字节）")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[down] {name}  ({total / 1e6:.1f} MB)")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
    if total and done != total:
        raise RuntimeError(f"{name}: 下载不完整 {done}/{total}")
    tmp.replace(dest)
    print(f"[done] {name}  {done:,} 字节")


def main() -> None:
    ok = True
    for name in FILES:
        try:
            fetch(name)
        except Exception as e:
            ok = False
            print(f"[fail] {name}: {e}")
    weights_ok = False
    for name in WEIGHT_CANDIDATES:
        try:
            fetch(name)
            weights_ok = True
            break
        except Exception as e:
            print(f"[fail] {name}: {e}")
    ok = ok and weights_ok
    if not ok:
        sys.exit(1)
    print("全部就绪 →", DEST)


if __name__ == "__main__":
    main()
