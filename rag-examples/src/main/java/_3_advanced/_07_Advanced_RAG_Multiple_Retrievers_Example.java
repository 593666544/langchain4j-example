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
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import shared.Assistant;

import java.nio.file.Path;
import java.util.List;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static shared.Utils.*;

public class _07_Advanced_RAG_Multiple_Retrievers_Example {


    /**
     * 请先掌握 {@link Naive_RAG_Example} 的基础流程后再阅读本例。
     * <p>
     * Advanced RAG 设计可参考：https://github.com/langchain4j/langchain4j/pull/538
     * <p>
     * 中文学习说明（多检索器并行 / Multiple Retrievers）：
     * 当你的知识来自多个独立语料时，可以让一个问题同时查询多个检索器，
     * 然后把各路结果汇总后交给模型回答。
     *
     * 与“Query Routing”不同：
     * - Routing 强调“选一个/几个最合适的数据源”；
     * - 本例强调“每次都查多个数据源”。
     *
     * 适用场景：
     * - 问题可能跨域（比如同时涉及人物背景和业务条款）
     * - 数据源数量不多，可接受并行检索开销
     */
    /**
     * 示例程序入口。
     * <p>
     * 关键对象：
     * - {@code assistant}：带多检索器并行能力的 RAG 助手。
     *
     * @param args 命令行参数（本示例未使用）
     */
    public static void main(String[] args) {

        Assistant assistant = createAssistant();

        // 你可以尝试跨域问题，观察回答是否融合两个语料来源。
        startConversationWith(assistant);
    }

    /**
     * 创建“多检索器并行”Assistant。
     * <p>
     * 关键对象：
     * - {@code contentRetriever1}：条款语料检索器。
     * - {@code contentRetriever2}：传记语料检索器。
     * - {@code queryRouter}：把每个问题同时路由到多个检索器。
     * - {@code retrievalAugmentor}：执行并行检索并组装上下文。
     *
     * @return 支持 Multiple Retrievers 的 Assistant
     */
    private static Assistant createAssistant() {

        // 子步骤 7.1：初始化共享向量模型。
        // 关键对象名：embeddingModel（供两个语料库共享）。
        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();

        // 子步骤 7.2：构建检索器 1（租车条款）。
        // 检索器 1：租车条款文档。
        EmbeddingStore<TextSegment> embeddingStore1 =
                embed(toPath("documents/miles-of-smiles-terms-of-use.txt"), embeddingModel);
        // 关键对象名：contentRetriever1。
        ContentRetriever contentRetriever1 = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore1)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.6)
                .build();

        // 子步骤 7.3：构建检索器 2（人物传记）。
        // 检索器 2：John Doe 传记文档。
        EmbeddingStore<TextSegment> embeddingStore2 =
                embed(toPath("documents/biography-of-john-doe.txt"), embeddingModel);
        // 关键对象名：contentRetriever2。
        ContentRetriever contentRetriever2 = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore2)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.6)
                .build();

        // 子步骤 7.4：创建并行路由器。
        // 默认路由器：每次问题都并行路由到两个检索器。
        // 关键对象名：queryRouter。
        // DefaultQueryRouter 的行为：把同一 query 同时路由到传入的所有 retriever。
        QueryRouter queryRouter = new DefaultQueryRouter(contentRetriever1, contentRetriever2);

        // 子步骤 7.5：把多检索器路由注入 RetrievalAugmentor。
        // 将“多检索器路由”注入 RetrievalAugmentor。
        // 关键对象名：retrievalAugmentor（执行多检索器并行检索并聚合结果）。
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // 子步骤 7.6：初始化最终回答模型。
        // 关键对象名：model（最终回答模型）。
        ChatModel model = OpenAiChatModel.builder()
                .apiKey(OPENAI_API_KEY)
                .modelName("gpt-4o-mini")
                .build();

        // 子步骤 7.7：组装并返回 Assistant。
        // 组装最终助手。
        return AiServices.builder(Assistant.class)
                .chatModel(model)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();
    }

    /**
     * 将单个文档构建为向量存储。
     *
     * @param documentPath 输入文档路径
     * @param embeddingModel 向量模型
     * @return 构建完成的向量存储
     */
    private static EmbeddingStore<TextSegment> embed(Path documentPath, EmbeddingModel embeddingModel) {
        // embed 子步骤 e1：读取文档。
        // 辅助方法：文档向量化入库。
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
