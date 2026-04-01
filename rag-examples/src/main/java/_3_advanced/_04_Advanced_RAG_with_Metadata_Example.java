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
import dev.langchain4j.rag.content.injector.ContentInjector;
import dev.langchain4j.rag.content.injector.DefaultContentInjector;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import shared.Assistant;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static java.util.Arrays.asList;
import static shared.Utils.*;

public class _04_Advanced_RAG_with_Metadata_Example {

    /**
     * 请先掌握 {@link Naive_RAG_Example} 的基础流程后再阅读本例。
     * <p>
     * Advanced RAG 设计可参考：https://github.com/langchain4j/langchain4j/pull/538
     * <p>
     * 中文学习说明（元数据注入 / Metadata Injection）：
     * 默认 RAG 只把“文本内容”注入提示词，但在很多业务中还需要附加上下文属性，
     * 比如：来源文件名、段落编号、作者、时间、租户、权限标签等。
     *
     * 本例演示如何把检索片段中的 metadata 一并注入 Prompt。
     * 这样模型不仅知道“内容是什么”，还知道“内容来自哪里”。
     */
    /**
     * 示例程序入口。
     * <p>
     * 关键对象：
     * - {@code assistant}：带元数据注入能力的 RAG 助手。
     *
     * @param args 命令行参数（本示例未使用）
     */
    public static void main(String[] args) {

        Assistant assistant = createAssistant("documents/miles-of-smiles-terms-of-use.txt");

        // 建议测试：
        // "What is the name of the file where cancellation policy is defined?"
        // 观察日志可看到 file_name 等 metadata 被注入到提示词中。
        startConversationWith(assistant);
    }

    /**
     * 创建“元数据注入版”Assistant。
     * <p>
     * 关键对象：
     * - {@code contentRetriever}：召回文本片段。
     * - {@code contentInjector}：控制哪些 metadata 被注入 Prompt。
     * - {@code retrievalAugmentor}：把检索与注入策略编排在一起。
     *
     * @param documentPath 作为知识库输入的文档路径
     * @return 支持 Metadata Injection 的 Assistant
     */
    private static Assistant createAssistant(String documentPath) {

        // 子步骤 4.1：准备文档与向量基础设施。
        // 基础准备：文档 -> 向量索引。
        // 关键对象名：document / embeddingModel / embeddingStore。
        Document document = loadDocument(toPath(documentPath), new TextDocumentParser());

        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        // 子步骤 4.2：构建摄取器。
        // 摄取器：把文档加工为向量索引。
        // 关键对象名：ingestor。
        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .documentSplitter(DocumentSplitters.recursive(300, 0))
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build();

        // 子步骤 4.3：执行摄取。
        // 执行摄取，完成切片与入库。
        ingestor.ingest(document);

        // 子步骤 4.4：创建内容检索器。
        // 关键对象名：contentRetriever（负责召回相关片段）。
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .build();

        // 关键点：配置 ContentInjector 将指定 metadata 键写入最终 Prompt。
        // 这里选择注入：
        // - file_name: 当前片段来自哪个文件
        // - index: 当前片段在分片序列中的位置
        // 子步骤 4.5：创建元数据注入器。
        // 关键对象名：contentInjector（控制 prompt 中注入哪些 metadata）。
        ContentInjector contentInjector = DefaultContentInjector.builder()
                // .promptTemplate(...) // 也可以自定义“内容 + 元数据”的注入模板格式
                .metadataKeysToInclude(asList("file_name", "index"))
                .build();

        // 子步骤 4.6：组装 RetrievalAugmentor（检索 + 元数据注入）。
        // 在 RetrievalAugmentor 中显式替换默认注入器。
        // 关键对象名：retrievalAugmentor。
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .contentRetriever(contentRetriever)
                .contentInjector(contentInjector)
                .build();

        // 子步骤 4.7：初始化最终回答模型。
        // 最终回答模型。
        // 关键对象名：chatModel。
        ChatModel chatModel = OpenAiChatModel.builder()
                .apiKey(OPENAI_API_KEY)
                .modelName("gpt-4o-mini")
                .logRequests(true)
                .build();

        // 子步骤 4.8：组装并返回 Assistant。
        // 组装 AI Service。
        return AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();
    }
}
