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
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import shared.Assistant;

import java.nio.file.Path;
import java.util.List;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static shared.Utils.*;

public class _08_Advanced_RAG_Web_Search_Example {


    /**
     * 请先掌握 {@link Naive_RAG_Example} 的基础流程后再阅读本例。
     * <p>
     * Advanced RAG 设计可参考：https://github.com/langchain4j/langchain4j/pull/538
     * <p>
     * 中文学习说明（Web 检索增强）：
     * 本地私有知识库通常只覆盖“内部知识”，而用户问题可能需要“外部实时信息”。
     * 因此可以把“向量库检索器 + Web 搜索检索器”组合起来，形成混合 RAG。
     *
     * 典型收益：
     * - 私有知识负责准确业务上下文
     * - Web 检索补齐外部知识
     *
     * 典型风险：
     * - 外部网页质量不稳定
     * - 响应时延和成本上升
     * <p>
     * 本示例依赖 `langchain4j-web-search-engine-tavily`。
     */
    /**
     * 示例程序入口。
     * <p>
     * 关键对象：
     * - {@code assistant}：融合“本地知识 + Web 搜索”的 RAG 助手。
     *
     * @param args 命令行参数（本示例未使用）
     */
    public static void main(String[] args) {

        Assistant assistant = createAssistant();

        // 可尝试同时涉及公司条款 + 外部事实的问题，观察融合效果。
        startConversationWith(assistant);
    }

    /**
     * 创建“本地检索 + Web 检索”混合版 Assistant。
     * <p>
     * 关键对象：
     * - {@code embeddingStoreContentRetriever}：本地私有知识检索器。
     * - {@code webSearchContentRetriever}：Web 搜索检索器。
     * - {@code queryRouter}：并行路由到两个检索通道。
     * - {@code retrievalAugmentor}：聚合混合检索结果用于回答。
     *
     * @return 支持 Web Search 的 Assistant
     */
    private static Assistant createAssistant() {

        // 子步骤 8.1：初始化向量模型（本地语料检索用）。
        // 检索器 1：本地向量库（内部文档）。
        // 关键对象名：embeddingModel。
        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();

        EmbeddingStore<TextSegment> embeddingStore =
                embed(toPath("documents/miles-of-smiles-terms-of-use.txt"), embeddingModel);

        // 子步骤 8.2：创建本地知识检索器。
        // 关键对象名：embeddingStoreContentRetriever（本地知识检索器）。
        ContentRetriever embeddingStoreContentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.6)
                .build();

        // 子步骤 8.3：初始化 Web 搜索引擎客户端。
        // 检索器 2：Web 搜索（外部知识）。
        // 关键对象名：webSearchEngine（Web 搜索引擎客户端）。
        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(System.getenv("TAVILY_API_KEY")) // 免费 key 获取地址：https://app.tavily.com/sign-in
                .build();

        // 子步骤 8.4：创建 Web 知识检索器。
        // 关键对象名：webSearchContentRetriever（外部知识检索器）。
        ContentRetriever webSearchContentRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .maxResults(3)
                .build();

        // 默认路由：每轮都查询“本地 + Web”两个检索器。
        // 子步骤 8.5：创建混合路由器。
        // 关键对象名：queryRouter（混合路由器）。
        // 同一问题并行走“本地向量检索 + Web 检索”。
        QueryRouter queryRouter = new DefaultQueryRouter(embeddingStoreContentRetriever, webSearchContentRetriever);

        // 子步骤 8.6：组装 RetrievalAugmentor（混合检索编排）。
        // 注入到 RetrievalAugmentor。
        // 关键对象名：retrievalAugmentor（混合检索编排器）。
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // 子步骤 8.7：初始化最终回答模型。
        // 关键对象名：model（最终回答模型）。
        ChatModel model = OpenAiChatModel.builder()
                .apiKey(OPENAI_API_KEY)
                .modelName("gpt-4o-mini")
                .build();

        // 子步骤 8.8：组装并返回 Assistant。
        // 组装 AI Service。
        return AiServices.builder(Assistant.class)
                .chatModel(model)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();
    }

    /**
     * 将本地文档构建为向量存储。
     *
     * @param documentPath 输入文档路径
     * @param embeddingModel 向量模型
     * @return 构建完成的向量存储
     */
    private static EmbeddingStore<TextSegment> embed(Path documentPath, EmbeddingModel embeddingModel) {
        // embed 子步骤 e1：读取文档。
        // 辅助方法：构建本地文档向量索引。
        // 关键对象名：document / segments / embeddings / embeddingStore。
        DocumentParser documentParser = new TextDocumentParser();
        Document document = loadDocument(documentPath, documentParser);

        // embed 子步骤 e2：文档切片。
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 0);
        List<TextSegment> segments = splitter.split(document);

        // embed 子步骤 e3：切片向量化。
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        // embed 子步骤 e4：向量入库并返回。
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        return embeddingStore;
    }
}
