package _3_advanced;

import _2_naive.Naive_RAG_Example;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
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
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;
import dev.langchain4j.rag.query.transformer.QueryTransformer;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import shared.Assistant;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static shared.Utils.*;

public class _01_Advanced_RAG_with_Query_Compression_Example {

    /**
     * 请先掌握 {@link Naive_RAG_Example} 的基础链路后再阅读本例。
     * <p>
     * Advanced RAG 设计可参考：https://github.com/langchain4j/langchain4j/pull/538
     * <p>
     * 中文学习说明（查询压缩 / Query Compression）：
     * 在多轮对话中，用户常说“他/它/这个”等省略式追问，导致检索词信息不足。
     * 例如：
     * - 用户：John Doe 的成就是什么？
     * - 助手：......
     * - 用户：他什么时候出生？
     *
     * 若直接拿“他什么时候出生”去检索，向量库很可能检不准。
     * Query Compression 的做法是：结合“当前问题 + 历史对话”，先让模型改写为可检索的完整查询，
     * 比如改写成 “When was John Doe born?”，再进入检索阶段。
     *
     * 优点：显著提升多轮场景的召回质量。
     * 代价：多一次模型调用，增加时延与成本。
     */
    /**
     * 示例程序入口。
     * <p>
     * 关键对象：
     * - {@code assistant}：带查询压缩能力的 RAG 助手。
     *
     * @param args 命令行参数（本示例未使用）
     */
    public static void main(String[] args) {

        Assistant assistant = createAssistant("documents/biography-of-john-doe.txt");

        // 建议体验：
        // 1) 先问："What is the legacy of John Doe?"
        // 2) 再问："When was he born?"
        // 观察日志可看到第二轮查询被压缩为更完整的独立查询。
        startConversationWith(assistant);
    }

    /**
     * 创建“查询压缩版”Assistant。
     * <p>
     * 关键对象：
     * - {@code queryTransformer}：把追问改写成可检索的独立查询。
     * - {@code contentRetriever}：基于改写后查询召回文本片段。
     * - {@code retrievalAugmentor}：把“改写 + 检索”串联为完整 RAG 流。
     *
     * @param documentPath 作为知识库输入的文档路径
     * @return 支持 Query Compression 的 Assistant
     */
    private static Assistant createAssistant(String documentPath) {

        // 子步骤 1.1：读取原始知识文档。
        // 准备知识文档（人物传记）。
        // 关键对象名：document（后续会被切片并写入向量库）。
        Document document = loadDocument(toPath(documentPath), new TextDocumentParser());

        // 子步骤 1.2：初始化 embedding 模型。
        // Embedding 模型：用于文档切片向量化 + 查询向量化。
        // 关键对象名：embeddingModel。
        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();

        // 子步骤 1.3：初始化向量库存储。
        // 向量库存储：保存切片向量及其原文。
        // 关键对象名：embeddingStore。
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        // 子步骤 1.4：构建摄取器（把文档加工成向量索引）。
        // 摄取器：一键执行切片、向量化、入库流水线。
        // 关键对象名：ingestor。
        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .documentSplitter(DocumentSplitters.recursive(300, 0))
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build();

        // 子步骤 1.5：执行摄取，把 document 变成可检索的向量索引。
        ingestor.ingest(document);

        // 子步骤 1.6：初始化聊天模型。
        // 聊天模型：同时用于查询改写与最终回答。
        // 关键对象名：chatModel。
        ChatModel chatModel = OpenAiChatModel.builder()
                .apiKey(OPENAI_API_KEY)
                .modelName("gpt-4o-mini")
                .build();

        // 子步骤 1.7：创建查询转换器。
        // 关键组件：查询转换器（压缩/改写查询）。
        // 它会把“用户当前问题 + 历史对话”压缩成更完整、更适合检索的查询文本。
        QueryTransformer queryTransformer = new CompressingQueryTransformer(chatModel);

        // 子步骤 1.8：创建检索器。
        // 检索器：使用改写后的查询在向量库中检索高相关片段。
        // 关键对象名：contentRetriever。
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.6)
                .build();

        // 子步骤 1.9：组装 RetrievalAugmentor。
        // RetrievalAugmentor 是 Advanced RAG 的装配中心。
        // 关键对象名：retrievalAugmentor。
        // 在这里把“查询改写 + 内容检索”等步骤串成完整 RAG 流。
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryTransformer(queryTransformer)
                .contentRetriever(contentRetriever)
                .build();

        // 子步骤 1.10：组装并返回 Assistant。
        // 最终把“聊天模型 + RAG 增强器 + 对话记忆”组装成可调用助手。
        return AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();
    }
}
