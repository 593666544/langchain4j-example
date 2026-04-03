package _3_advanced;

import _2_naive.Naive_RAG_Example;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import shared.Assistant;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static shared.Utils.*;

public class _02_Advanced_RAG_with_Query_Routing_Example {

    /**
     * 请先掌握 {@link Naive_RAG_Example} 的基础流程后再阅读本例。
     * <p>
     * Advanced RAG 设计可参考：https://github.com/langchain4j/langchain4j/pull/538
     * <p>
     * 中文学习说明（查询路由 / Query Routing）：
     * 当你的知识分散在多个数据源（文档库、代码库、数据库、搜索引擎）时，
     * 并不是每个问题都要去查所有数据源，否则会带来：
     * - 额外延迟与成本
     * - 更多噪声上下文
     * - 可能更差的回答质量
     *
     * Query Routing 的目标是：把问题送到“最合适”的检索器（一个或多个）。
     * 常见实现方式：
     * - 规则路由（权限、租户、区域）
     * - 关键词路由
     * - 语义分类路由
     * - LLM 路由（本例）
     */
    /**
     * 示例程序入口。
     * <p>
     * 关键对象：
     * - {@code assistant}：带查询路由能力的 RAG 助手。
     *
     * @param args 命令行参数（本示例未使用）
     */
    public static void main(String[] args) {

        Assistant assistant = createAssistant();

        // 建议测试：
        // - "What is the legacy of John Doe?"（应偏向 biography 检索器）
        // - "Can I cancel my reservation?"（应偏向 terms-of-use 检索器）
        // 观察日志可以看到路由决策结果。
        startConversationWith(assistant);
    }

    /**
     * 创建“查询路由版”Assistant。
     * <p>
     * 关键对象：
     * - {@code biographyContentRetriever}：人物传记检索器。
     * - {@code termsOfUseContentRetriever}：业务条款检索器。
     * - {@code queryRouter}：根据问题选择合适检索器。
     * - {@code retrievalAugmentor}：承载路由策略的 RAG 编排器。
     *
     * @return 支持 Query Routing 的 Assistant
     */
    private static Assistant createAssistant() {

        // 子步骤 2.1：初始化共享 embedding 模型。
        // 向量模型（供所有语料共享）。
        // 关键对象名：embeddingModel。
        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();

        // 子步骤 2.2：构建检索源 A（人物传记）。
        // 检索源 A：人物传记知识库。
        // 为什么要拆成独立检索源：
        // 语料域不同（人物传记 vs 条款制度）时，拆分索引更容易做路由与治理。
        EmbeddingStore<TextSegment> biographyEmbeddingStore =
                embed(toPath("documents/biography-of-john-doe.txt"), embeddingModel);
        // 关键对象名：biographyContentRetriever（传记语料检索器）。
        // 参数说明：
        // maxResults(2) 控制注入片段数量，minScore(0.6) 过滤弱相关结果；
        // 两者一起决定“召回覆盖率 vs 噪声/成本”的平衡。
        ContentRetriever biographyContentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(biographyEmbeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.6)
                .build();

        // 子步骤 2.3：构建检索源 B（租车条款）。
        // 检索源 B：租车条款知识库。
        // 这样做的价值：
        // 让每个 retriever 都有清晰的语义边界，避免“全量混合索引”带来的误召回。
        EmbeddingStore<TextSegment> termsOfUseEmbeddingStore =
                embed(toPath("documents/miles-of-smiles-terms-of-use.txt"), embeddingModel);
        // 关键对象名：termsOfUseContentRetriever（条款语料检索器）。
        // 保持与检索器 A 一致的参数，便于对比路由效果，避免“参数差异”干扰实验结论。
        ContentRetriever termsOfUseContentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(termsOfUseEmbeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.6)
                .build();

        // 子步骤 2.4：初始化聊天模型。
        // 关键对象名：chatModel（既用于路由判定，也用于最终回答）。
        ChatModel chatModel = OpenAiChatModel.builder()
                .apiKey(OPENAI_API_KEY)
                .modelName("gpt-4o-mini")
                .build();

        // 子步骤 2.5：准备“检索器 -> 语义描述”映射。
        // 路由器输入的是 “检索器 -> 描述” 映射。
        // LLM 会根据用户问题与描述语义匹配，决定路由到哪个检索器。
        // 映射结构：检索器对象 -> 语义描述文本。
        // LLM 路由器会阅读描述并决定该问题应该走哪个检索器。
        // 为什么描述文本很关键：
        // LLM 路由器并不理解你的变量名，它依赖这些自然语言描述做语义匹配决策。
        Map<ContentRetriever, String> retrieverToDescription = new HashMap<>();
        retrieverToDescription.put(biographyContentRetriever, "biography of John Doe");
        retrieverToDescription.put(termsOfUseContentRetriever, "terms of use of car rental company");
        // 子步骤 2.6：构建 LLM 路由器。
        // 关键对象名：queryRouter。
        // 为什么单独引入 queryRouter：
        // 把“选哪个知识源”从“如何回答”中剥离，后续可替换成规则路由或混合策略。
        QueryRouter queryRouter = new LanguageModelQueryRouter(chatModel, retrieverToDescription);

        // 子步骤 2.7：把路由策略注入 RetrievalAugmentor。
        // 将 queryRouter 注入 RetrievalAugmentor，形成可配置 RAG 流。
        // 关键对象名：retrievalAugmentor（承载路由策略）。
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // 子步骤 2.8：组装并返回 Assistant。
        // 组装最终助手。
        return AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();
    }

    /**
     * 将单个文档构建为可检索的向量库。
     * <p>
     * 关键对象：
     * - {@code DocumentSplitter}：控制分块策略。
     * - {@code EmbeddingModel}：把切片转成向量。
     * - {@code EmbeddingStore<TextSegment>}：保存向量与原文映射关系。
     *
     * @param documentPath 待索引文档路径
     * @param embeddingModel 向量模型
     * @return 已入库的向量存储对象
     */
    private static EmbeddingStore<TextSegment> embed(Path documentPath, EmbeddingModel embeddingModel) {
        // 子步骤 E1：读取文档。
        // 该辅助方法展示了“单文档 -> 向量库”最简流程：
        // 加载 -> 切片 -> 向量化 -> 入库。
        // 关键对象名：document / splitter / segments / embeddings / embeddingStore。
        DocumentParser documentParser = new TextDocumentParser();
        Document document = loadDocument(documentPath, documentParser);

        // 子步骤 E2：文档切片。
        // 为什么先切片再向量化：
        // 检索单元过大会降低召回精度，过小会丢上下文；切片是检索质量的基础杠杆。
        // 参数说明：
        // 300 控制块大小；0 表示不重叠（成本更低，但边界语义可能断裂）。
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 0);
        List<TextSegment> segments = splitter.split(document);

        // 子步骤 E3：切片向量化。
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        // 子步骤 E4：向量入库并返回。
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        return embeddingStore;
    }
}
