# 智膳 Agent（Mediterranean Diet Planning Agent）

> 输入饮食需求 / 偏好 / 忌口 → LangGraph 状态图编排：两阶段检索（Query Rewrite + Cross-Encoder Rerank）生成候选池 → Planner 生成膳食计划（只准用池内菜品 ID，可自主调用查菜工具）→ 约束校验 → 违规带反馈回流 → 结构化计划 + Markdown/JSON 导出 + SQLite 历史 + 量化评测。
> 基座：在本人自建的地中海饮食 RAG 问答系统之上构建（见下节），索引与评测集已并入本仓库，clone 即可运行。
> LLM：DeepSeek（deepseek-chat，OpenAI 兼容接口，支持 tool calling）。

## RAG 基座（本人的前置项目，资产已并入 `data/` 与 `qa/`）

本项目的检索基座来自我此前独立构建的**地中海饮食领域 RAG 问答系统**：
从 Wikipedia / Wikibooks / 美食博客三个异构来源爬取 **616 篇**菜品食谱文档，
在流水线每个环节做三种策略横向对比（3 种切分 × 3 种向量化 × 4 路检索 × 3 种 prompt），
用 **210 组 QA** 基准集完成自动评测（关键词重合率 / BLEU-1 / ROUGE-L）。

- 最佳配置：sentence-window 切分 → `all-MiniLM-L6-v2` 稠密检索 → FAISS 余弦 → instructed prompt
- 关键结论：稠密检索 Top-1 关键词重合率 0.569，比 TF-IDF 基线（0.337）高约 69%；prompt 设计的影响（ROUGE-L +0.076~0.099）不亚于换检索器（+0.033~0.057）
- 本项目沿用其最佳配置产物：`data/index/`（1650 chunks + FAISS 索引）与 `qa/`（210 组评测题）

即：**同一个语料基座，从"RAG 问答系统"进化为"带约束校验与多轮记忆的 Agent"**——
这正是本项目的核心叙事。原工程（含完整对比实验与爬虫）：
`Mediterranean_RAG_Deliverables`。

## 核心特性

- **防幻觉双层防线**：Planner 只准输出候选池内的 dish_id（prompt 约束）+ validate_node 代码校验兜底（ID/约束/结构完整性），违规菜在检索层就被过滤出池
- **六路意图路由 + 专家节点**（LangGraph multi-agent 形态）：new_plan / edit / replan / qa（语料问答）/ plan_qa（菜单问答，只答计划内的菜）/ chat，一次 LLM 调用完成意图分类 + 约束抽取
- **多轮对话 + 长期记忆**：偏好/忌口/喜好跨轮延续（"忘了说了我不吃辣"自动合并重排），偏好画像从历史实时推导注入 prompt
- **喜好建模与安排理由**："喜欢吃鸡肉"→ 检索补路 + planner 主动编排（同菜不跨天）；每份计划附"为什么这么安排"（planner 同轮输出 rationale，零额外成本）
- **中文全程体验**：中文偏好/忌口归一化（"素食/不吃辣"直接可用）+ 输出层翻译节点（只进展示不进校验链）+ 跨语言检索（中文问题先译英再查英文语料）
- **编辑精确性**：自然语言改菜单（"把第一天的晚餐换成鱼"），代码层保证只动目标餐位，其余强制保留
- 两阶段检索（Query Rewrite + Cross-Encoder Rerank，均带失败降级回退）+ 覆盖分级 A/B/C + Markdown/JSON 导出 + SQLite 历史 + Token 统计

## 设计取舍

> 本项目定位为评测驱动的 Agent 管线，不含 Web 前端与容器化部署；
> Redis 缓存以进程内 TTL 缓存等价替代（rerank 结果命中率场景一致）。

## 架构

```
用户任意输入（一句话，中英皆可）
        │
        ▼
┌────────────────── LangGraph StateGraph（路由 + 专家节点）───────────────────┐
│ understand（一次 LLM 调用：意图分类 + 约束抽取）                            │
│   ├─ new_plan/replan ─► retrieve ─► planner ─► validate ─┬─► translate ─►  │
│   │   （replan 先合并历史偏好）  多路检索   (DeepSeek      │    中文输出      │
│   │                            +约束过滤    ID约束+工具)  │(不合格且未超步数)│
│   │                                ▲  └────── 带违规反馈 ◄┘                │
│   ├─ edit ────────────► 同上规划链（以最近计划为修改基底）                   │
│   ├─ qa ─────────────► 语料问答专家（RAG 检索，中文回答，只准引用语料）      │
│   ├─ plan_qa ────────► 菜单问答专家（只答当前/最近计划内的菜，不串台语料库）│
│   └─ chat ───────────► 结束（固定应答）                                     │
└────────────────────────────────────────────────────────────────────────────┘
```

**多轮对话**：偏好/忌口/喜好跨轮次延续——"忘了说了我不吃辣"会被识别为 replan，
与上次运行偏好合并后重新规划；"咖喱是什么菜"路由给语料问答专家（语料没有就明说）；
"你给我的哪些有鸡肉"路由给菜单问答专家（只从计划里答，引用具体天次）。
理解节点一次 LLM 调用完成意图分类 + 约束/喜好抽取，不用分两行输入。
**喜好与限制分开建模**：限制（vegan/忌口）决定"不给什么"（检索层过滤），
喜好（"喜欢吃鸡肉"）决定"多给什么"（检索补路 + planner 主动编排 + 同菜不跨天）。
每份计划附**安排理由**：planner 在同一轮输出 rationale（零额外调用），随计划翻译展示。

**防幻觉机制（核心卖点）**：Planner 只准输出候选池内的 dish_id（prompt 约束）；
validate_node 逐个用代码复核 ID、约束与**计划结构完整性**（天数、每日三餐，
空 Day 会被打回重排）——多层防线；候选池本身在检索层就经过约束过滤，
planner 拿到的池子天然合规。所有菜品数据一律由语料回填，不采信模型生成描述。

## 模块

| 文件 | 内容 |
|---|---|
| `retriever.py` | FAISS 向量召回（基线） |
| `rag.py` | Query Rewrite + Cross-Encoder Rerank + TTL 缓存 |
| `tools.py` | `lookup_dish`（语料回填）/ `check_constraints`（词表校验，词边界+复数匹配，中文偏好/忌口归一化），含 9 例自测 |
| `graph.py` | StateGraph：intent → retrieve(多路+约束过滤) → planner(tool calling) → validate(ID/约束/完整性) → 条件回流 → translate(中文输出) |
| `exporter.py` / `store.py` | Markdown+JSON 导出 / SQLite 历史 |
| `evaluate.py` | 检索三模式对比 + 幻觉率 + 约束满足率 + Agent 行为（意图准确率 / 工具调用成功率）+ **端到端场景评测**（3 场景 × 9 轮多轮对话，逐检查点断言）+ RAGAS 风格生成质量三指标 → `results/eval_report.md` |

## 快速开始

```bash
cd E:\Study\Text_mining\LangChain\diet-agent
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env   # 填入 DEEPSEEK_API_KEY（platform.deepseek.com 申请）
venv\Scripts\python tools.py                      # ① 工具层自测（不花钱）
venv\Scripts\python retriever.py                  # ② 检索自测（不花钱）
venv\Scripts\python scripts\download_reranker.py  # ③ 下载精排模型 ~1.1GB（一次性）
venv\Scripts\python run_cli.py                    # ④ 交互规划/编辑/历史/记忆
venv\Scripts\python evaluate.py                   # ⑤ 评测出报告（约 290 次 LLM 调用，含端到端）
venv\Scripts\python evaluate.py --qa 30           # ⑤' 追加 RAG 生成质量评测（每题 +4 次调用）
```

> 索引与评测集已随仓库附带（`data/index/`、`qa/`），无需从外部项目拷贝任何文件。

## 评测结果（deepseek-chat，temperature=0，2026-09-28 定稿实跑，683 次调用）

| 指标 | 数值 |
|---|---|
| 检索 baseline Top1 / MRR（n=210） | **0.6714 / 0.6967** |
| + Query Rewrite Top1 / MRR | 0.6667 / 0.6956 |
| + Rewrite & Rerank Top1 / MRR | 0.6571 / 0.6873 |
| 首轮幻觉率 → 回流后幻觉率 | 0.0 → 0.0 |
| 首轮约束违规率 → 最终约束满足率 | 0.0 → **1.0** |
| 意图分类准确率（六路意图 × 有/无历史，n=19） | **1.0** |
| 工具调用成功率（n=435 次调用） | **1.0** |
| **端到端任务完成率**（3 场景 × 9 轮多轮对话，25 个检查点） | **1.0** |
| RAG 忠实度 Faithfulness（n=30，LLM-as-judge） | 0.867 |
| RAG 答案相关性 Answer Relevancy（n=30，RAGAS 口径） | 0.635 |
| RAG 上下文相关性 Context Relevance（n=30） | 0.433 |

**怎么读这张表**：

1. **检索优化是负结果，但负得有信息量**。两阶段检索在本评测集上没有提升（±1.4pp ≈ 3 题，噪声范围内）。原因：本评测的 210 题是 QA 式问句（本身已具体），LLM 改写的收益空间为零——第一版改写词还曾把 Top1 从 15/20 拉到 12/20（扩写新词带偏向量语义），改为"只消歧、不扩写"的保守指令后才回到持平。保留该机制的原因：**规划链的真实入口是模糊口语需求**（"something easy for dinner"），改写 + 降级回退（rewrite 失败退回原查询、rerank 失败退回原排序）保证优化层任何异常都不伤基线。
2. **约束防线前移到检索层**：早期版本首轮违规率 0.5（素食/清真请求的候选池里就有违规菜），检索层"多路召回 + 约束源头过滤 + 统一文本源"改造后首轮违规率降为 0——**最好的校验是让违规菜根本进不了候选池**；validate 回流闭环保留作为兜底（端到端评测曾抓到过滤/校验文本源不一致导致"能进池必违规、打回换不出"的死局，修复后 halal 连跑 3 次全收敛）。
3. **幻觉率 0.0 是"双层防线"的预防性结果**：prompt 层 ID 约束让模型第一轮就不出池，代码层校验作为安全网全程零触发——防幻觉机制的价值在"保证下限"，不在"修复存量"。
4. **端到端评测抓出两个单点指标测不出的真 bug**（修完才跑出 1.0）：① replan 丢天数——"给我两天的菜单"后说"忘了说了我不吃辣"，重排默认成 3 天（当前话没提天数应从历史继承）；② 编辑不精确——"换第一天的晚餐"被 planner 重建了全部 9 个餐位（修复：解析指令定位目标餐位，代码层强制其余餐位保留原菜，与防幻觉同一哲学：prompt 约束 + 代码兜底）。回归用例固化在 `python graph.py` 离线自测。

完整报告见 `results/eval_report.md`（含逐请求明细与局限声明）。
