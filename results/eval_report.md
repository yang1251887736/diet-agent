# 智膳 Agent 评测报告

生成时间：2026-09-28 16:56　模型：deepseek-chat（temperature=0）　Token：683 次调用 / 输入 217305 / 输出 24389

## 1. 检索三模式对比（n=210）

| 模式 | Top1 命中率 | MRR |
|---|---|---|
| baseline（纯向量召回） | 0.6714 | 0.6967 |
| + Query Rewrite | 0.6667 | 0.6956 |
| + Rewrite & Cross-Encoder Rerank | 0.6571 | 0.6873 |

口径：答案文本中出现的菜品名是否被召回。

## 2. 规划（固定 6 个请求，规划链开启 rewrite+rerank）

| 请求 | 候选池 | 首轮幻觉 | 回流后幻觉 | 首轮违规 | 最终通过 |
|---|---|---|---|---|---|
| Plan 3 days of Mediterranean meals | 11 | 0 | 0 | 0 | 是 |
| Plan 2 days of vegetarian Mediterranean meals | 9 | 0 | 0 | 0 | 是 |
| Plan 2 days of vegan Mediterranean meals | 6 | 0 | 0 | 0 | 是 |
| Plan 2 days of Mediterranean meals with seafood focus | 12 | 0 | 0 | 0 | 是 |
| Plan 2 days of Mediterranean meals without nuts | 13 | 0 | 0 | 0 | 是 |
| Plan 3 days of halal Mediterranean meals | 15 | 0 | 0 | 0 | 是 |

| 指标 | 数值 |
|---|---|
| 首轮幻觉率（dish_id 不在候选池） | 0.0 |
| 回流后幻觉率 | 0.0 |
| 首轮约束违规率（按请求） | 0.0 |
| 最终约束满足率 | 1.0 |

## 3. Agent 行为

| 指标 | 数值 |
|---|---|
| 意图分类准确率（六路意图 × 有/无历史，n=19） | 1.0 |
| 工具调用（lookup_dish）次数 | 435 |
| 工具调用成功率 | 1.0 |

意图误判明细：无

## 4. 端到端场景评测（多轮对话脚本，逐检查点断言）

**端到端任务完成率：1.0**（整场景全过为完成）　检查点通过率：1.0

| 场景 | 轮数 | 检查点 | 通过 |
|---|---|---|---|
| 多轮记忆：点菜→补忌口→诚实问答 | 3 | 10/10 | ✓ |
| 喜好全链路：编排→菜单问答→编辑精确 | 4 | 9/9 | ✓ |
| 中文约束 + 跨语言问答 | 2 | 6/6 | ✓ |

## 5. RAG 生成质量（RAGAS 风格，n=30，LLM-as-judge：deepseek-chat）

| 指标 | 含义 | 数值 |
|---|---|---|
| Faithfulness 忠实度 | 答案论断被上下文支撑的程度（1=大量编造，5=全部可溯源，归一化 0~1） | 0.8667 |
| Answer Relevancy 答案相关性 | 从答案反推问题与原问题的 embedding 余弦（RAGAS 官方口径，MiniLM） | 0.6346 |
| Context Relevance 上下文相关性 | 召回 chunk 中含回答所需信息的比例 | 0.4333 |

> 注：三指标均为 LLM-as-judge / embedding 余弦，judge 与被评模型同源（deepseek-chat）存在自评偏好，数字用于横向对比与回归监测，不宜绝对化解读。

## 局限
- 检索命中口径为菜品名字面匹配，未做语义相关性人工抽检。
- 约束校验为词表词边界匹配，仍有语义级漏检（描述不含任何食材词时无法拦截）。
- 规划评测仅 6 个固定请求、端到端仅 3 个场景，样本小；temperature=0 下LLM 仍有微小非确定性（尤其编辑重建整份计划），单次结果以复现为准。
- RAGAS 风格三指标依赖 LLM-as-judge，judge 与生成模型同源（deepseek-chat），存在自评偏差。