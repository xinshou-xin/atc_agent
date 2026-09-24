---
name: atc-mapping
description: Map a drug to WHO ATC code candidates using official evidence, including ambiguous names, multiple indications, and multiple routes of administration.
---

# ATC 映射工作流

1. 先调用 translate_drug_name 标准化，优先使用返回的 INN（inn）通用名进行检索；inn 为空时用 translated。
2. WHO 检索无结果时，必须用 INN 通用名/别名再试一次，不得因商品名搜空直接放弃。
3. 优先使用 WHO ATC 检索并逐个核验最终候选编码。
4. 允许同一药物有多个 ATC 编码，但每个编码必须写明对应适应症或给药途径。
5. 不得把药理类别推断当作官方编码证据。WHO 未命中或证据冲突时，降低置信度并请求人工核验。
6. 输出编码必须是 7 位格式：字母、两位数字、两位字母、两位数字。

# 证据优先级

1. WHO ATC 官方索引及编码详情
2. 监管或权威药物术语数据库
3. PubChem、Tavily（网络搜索补充）
4. 项目向量知识库
5. 模型常识（仅背景，不得作为唯一依据）
