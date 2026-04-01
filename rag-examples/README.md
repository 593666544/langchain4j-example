# rag-examples 学习路线图（中文硬核版）

这份 README 面向“按源码学习 RAG”的同学，目标是：
1. 明确先看哪个示例。
2. 明确每个示例先看哪几行、哪个方法。
3. 明确关键方法/关键对象到底做什么。
4. 边读边跑，能快速验证理解。

---

## 0. 当前状态（已同步到本次升级）

- `rag-examples` 模块内所有 Java 示例文件都已补充“步骤 + 子步骤 + 关键对象职责”的中文学习注释。
- 你现在可以按“先看注释，再看代码，再跑例子”的方式学习。
- 本文中的行号已按当前版本更新；后续若继续改文件，行号会有小幅漂移。

---

## 1. 运行前准备（先过这一步）

### 1.1 环境要求

- JDK：17（必须）
- Maven：建议使用项目自带 `./mvnw`
- API Key：
- `OPENAI_API_KEY`（多数示例必需）
- `COHERE_API_KEY`（`_03_ReRanking` 需要）
- `TAVILY_API_KEY`（`_08_Web_Search` 需要）

### 1.2 编译验证（推荐命令）

```bash
JAVA_HOME=$(/usr/libexec/java_home -v 17) ./mvnw -pl rag-examples -DskipTests compile
```

---

## 2. 常见坑：`import static ...GPT_4_O_MINI` 无法导入

### 2.1 常见原因

- 依赖版本与代码不一致：当前依赖版本中可能没有该枚举常量。
- IDEA 依赖索引未刷新：Maven 依赖实际下载了，但 IDE 仍用旧索引。
- JDK/Importer 不一致：项目是 JDK17，但 Maven Importer/Runner 不是同一 JDK。

### 2.2 本模块当前处理方式

为避免你学习时被静态常量导入卡住，示例里统一使用：

```java
.modelName("gpt-4o-mini")
```

说明：
- 这与常量方式在效果上等价（本质都是传模型名）。
- 学习重点在 RAG 链路，不在常量导入细节。

---

## 3. 学习顺序（建议严格按顺序）

1. `shared` 公共层
2. `_1_easy` 先跑通
3. `_2_naive` 看清标准流程
4. `_4_low_level` 掌握手工全链路
5. `_3_advanced` 按编号 01 -> 10

---

## 4. 逐文件读码定位（最新行号）

> 建议每个文件都先看类注释，再看 `main(...)` 或测试入口，再看核心构建方法。

### 4.1 公共基础

- `src/main/java/shared/Assistant.java`
- `33`：`interface Assistant`
- `54`：`String answer(String query)`

- `src/main/java/shared/Utils.java`
- `57`：`startConversationWith(...)`
- `105`：`glob(...)`
- `132`：`toPath(...)`

### 4.2 Easy / Naive / Low-level 主线

- `src/main/java/_1_easy/Easy_RAG_Example.java`
- `70`：`main(...)`
- `114`：`createContentRetriever(...)`

- `src/main/java/_2_naive/Naive_RAG_Example.java`
- `59`：`main(...)`
- `88`：`createAssistant(...)`

- `src/main/java/_4_low_level/_01_Low_Level_Naive_RAG_Example.java`
- `62`：`main(...)`
- `118`：`EmbeddingSearchRequest.builder()`
- `128`：`PromptTemplate.from(...)`

### 4.3 Advanced（01 - 10）

- `src/main/java/_3_advanced/_01_Advanced_RAG_with_Query_Compression_Example.java`
- `57`：`main(...)`
- `79`：`createAssistant(...)`
- `119`：`new CompressingQueryTransformer(...)`

- `src/main/java/_3_advanced/_02_Advanced_RAG_with_Query_Routing_Example.java`
- `64`：`main(...)`
- `86`：`createAssistant(...)`
- `134`：`new LanguageModelQueryRouter(...)`

- `src/main/java/_3_advanced/_03_Advanced_RAG_with_ReRanking_Example.java`
- `60`：`main(...)`
- `82`：`createAssistant(...)`
- `126`：`ReRankingContentAggregator.builder()`

- `src/main/java/_3_advanced/_04_Advanced_RAG_with_Metadata_Example.java`
- `51`：`main(...)`
- `72`：`createAssistant(...)`
- `111`：`metadataKeysToInclude(...)`

- `src/main/java/_3_advanced/_05_Advanced_RAG_with_Metadata_Filtering_Examples.java`
- `73`：`Static_Metadata_Filter_Example()`
- `150`：`Dynamic_Metadata_Filter_Example()`
- `220`：`LLM_generated_Metadata_Filter_Example()`
- `183`、`262`：`dynamicFilter(...)`

- `src/main/java/_3_advanced/_06_Advanced_RAG_Skip_Retrieval_Example.java`
- `65`：`main(...)`
- `85`：`createAssistant(...)`
- `117`：`PromptTemplate.from(...)`

- `src/main/java/_3_advanced/_07_Advanced_RAG_Multiple_Retrievers_Example.java`
- `61`：`main(...)`
- `80`：`createAssistant(...)`
- `114`：`new DefaultQueryRouter(...)`

- `src/main/java/_3_advanced/_08_Advanced_RAG_Web_Search_Example.java`
- `66`：`main(...)`
- `85`：`createAssistant(...)`
- `113`：`WebSearchContentRetriever.builder()`
- `122`：`new DefaultQueryRouter(...)`

- `src/main/java/_3_advanced/_09_Advanced_RAG_Return_Sources_Example.java`
- `53`：内部 `Assistant` 接口
- `61`：`Result<String> answer(...)`
- `73`：`main(...)`
- `120`：`createAssistant(...)`

- `src/main/java/_3_advanced/_10_Advanced_RAG_SQL_Database_Retreiver_Example.java`
- `52`：`main(...)`
- `73`：`createAssistant(...)`
- `89`：`SqlDatabaseContentRetriever.builder()`
- `112`：`createDataSource()`

---

## 5. 高频方法词典（先理解这几个）

### 5.1 `createContentRetriever(...)`（Easy）

位置：`_1_easy/Easy_RAG_Example.java:114`

作用：
- 把 `List<Document>` 变成“可用于问答检索”的 `ContentRetriever`。

关键对象：
- `InMemoryEmbeddingStore<TextSegment>`：内存向量库，存“向量 + 文本切片”。
- `EmbeddingStoreIngestor`：一键摄取器，内部做切片、向量化、入库。
- `EmbeddingStoreContentRetriever`：检索器实现，问答时按相似度召回片段。

返回对象（响应对象）叫什么：
- 返回类型：`ContentRetriever`（接口）
- 具体实现：`EmbeddingStoreContentRetriever`

### 5.2 `createAssistant(...)`（Naive/Advanced）

作用：
- 统一装配 `chatModel + contentRetriever + chatMemory(+ retrievalAugmentor)`，
  返回可直接调用的 `Assistant` 代理。

关键对象：
- `ChatModel`：生成最终自然语言答案。
- `ContentRetriever`：召回知识片段。
- `ChatMemory`：保存多轮上下文。
- `RetrievalAugmentor`（部分 advanced）：拼装更复杂检索链路。

### 5.3 `dynamicFilter(...)`（Advanced 05）

作用：
- 在每次检索前，按当前请求上下文动态生成 metadata 过滤条件。

典型用途：
- 多租户隔离（`userId` 只能看自己的数据）。
- 业务域隔离（只检索某类型文档）。

---

## 6. 推荐学习节奏（可直接照做）

1. 第 1 天：`shared + _1_easy`
2. 第 2 天：`_2_naive`
3. 第 3 天：`_4_low_level`
4. 第 4 天：advanced `01/02/06`（查询改写/路由/跳检索）
5. 第 5 天：advanced `03/04/05`（重排/元数据/过滤）
6. 第 6 天：advanced `07/08/09/10`（多检索器/Web/sources/SQL）
7. 第 7 天：做一个 mini RAG（至少包含“过滤”或“路由”）

---

## 7. 每读完一个示例，自测 4 个问题

1. 这个示例比 Naive RAG 多了哪个组件？
2. 这个组件解决了什么失败场景？
3. 它对成本/延迟/复杂度带来什么变化？
4. 如果上生产，要补什么安全和监控？

---

## 8. 一句话复盘

- Easy：先跑通。
- Naive：看清标准链路。
- Low-level：掌握手工控制。
- Advanced：解决真实业务中的质量、隔离、可解释与多数据源问题。
