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
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.Result;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.List;
import java.util.Scanner;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static shared.Utils.OPENAI_API_KEY;
import static shared.Utils.toPath;

public class _09_Advanced_RAG_Return_Sources_Example {


    /**
     * 请先掌握 {@link Naive_RAG_Example} 的基础流程后再阅读本例。
     * <p>
     * Advanced RAG 设计可参考：https://github.com/langchain4j/langchain4j/pull/538
     * <p>
     * 中文学习说明（返回来源 / Return Sources）：
     * 生产中的 RAG 往往要求“可解释性”：
     * - 回答不仅要给结论，还要给依据（来自哪些片段）。
     *
     * 本例通过 `Result<String>` 返回：
     * - content(): 模型最终回答
     * - sources(): 本轮参与回答的检索内容列表
     *
     * 这对于审计、可视化溯源、人工复核非常有价值。
     */

    interface Assistant {

        /**
         * 与普通 String 返回值相比，Result<String> 额外包含 sources 信息。
         * 为什么要改返回类型：
         * “答案 + 证据”是生产可解释性的基础，便于做人工复核与前端溯源展示。
         *
         * @param query 用户问题
         * @return Result 对象：content 为答案，sources 为检索来源
         */
        Result<String> answer(String query);
    }

    /**
     * 示例程序入口。
     * <p>
     * 关键对象：
     * - {@code result}：包含“答案 + 来源”的结果容器。
     * - {@code sources}：本轮参与回答的检索内容集合。
     *
     * @param args 命令行参数（本示例未使用）
     */
    public static void main(String[] args) {

        // 子步骤 9.1：创建支持“答案 + 来源”返回结构的 Assistant。
        // 关键对象名：assistant（返回 Result<String> 的问答代理）。
        Assistant assistant = createAssistant();

        // 手动循环对话，便于每轮都同时打印“答案 + 来源”。
        Logger log = LoggerFactory.getLogger(shared.Assistant.class);
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                // 子步骤 9.2：读取用户输入。
                log.info("==================================================");
                log.info("User: ");
                String userQuery = scanner.nextLine();
                log.info("==================================================");

                // 子步骤 9.3：提供退出条件。
                if ("exit".equalsIgnoreCase(userQuery)) {
                    break;
                }

                // 子步骤 9.4：发起问答并拿到结构化结果。
                // result.content()：最终自然语言答案。
                // result.sources()：本轮参与回答的证据片段集合。
                // 为什么先拿 result 再分别读字段：
                // 这样你能同时处理“展示答案”和“展示证据”，而不是把证据链丢在框架内部。
                Result<String> result = assistant.answer(userQuery);
                log.info("==================================================");
                log.info("Assistant: " + result.content());

                log.info("Sources: ");
                // 子步骤 9.5：输出来源列表，便于溯源学习。
                // sources() 即本轮被检索并注入上下文的内容，通常包含文本与元数据。
                // 关键对象名：sources（证据集合）。
                // 没有这一步时，用户只能看到结论，看不到依据，排障与信任建设都更困难。
                List<Content> sources = result.sources();
                sources.forEach(content -> log.info(content.toString()));
            }
        }
    }

    /**
     * 创建“可返回来源”的 Assistant。
     * <p>
     * 关键对象：
     * - {@code contentRetriever}：负责召回相关片段。
     * - {@code chatModel}：基于片段生成最终回答。
     *
     * @return 返回 Result<String> 的 Assistant 代理对象
     */
    private static Assistant createAssistant() {

        // 子步骤 9.A1：初始化向量模型。
        // 基础准备：向量检索器。
        // 关键对象名：embeddingModel / embeddingStore / contentRetriever / chatModel。
        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();

        // 子步骤 9.A2：构建向量存储。
        EmbeddingStore<TextSegment> embeddingStore =
                embed(toPath("documents/miles-of-smiles-terms-of-use.txt"), embeddingModel);

        // 子步骤 9.A3：创建内容检索器。
        // 为什么这里仍用常规 contentRetriever：
        // “返回来源”不是新的检索算法，而是把已有检索结果暴露给调用方。
        // 参数说明：
        // maxResults(2) 与 minScore(0.6) 影响 sources 数量和质量，直接影响溯源可读性。
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.6)
                .build();

        // 子步骤 9.A4：初始化聊天模型。
        ChatModel chatModel = OpenAiChatModel.builder()
                .apiKey(OPENAI_API_KEY)
                .modelName("gpt-4o-mini")
                .build();

        // 关键：Assistant 接口返回类型是 Result<String>，因此可拿到 sources。
        // 子步骤 9.A5：组装并返回 Assistant。
        // chatMemory(10) 的意义：
        // 在多轮对话中保留近期上下文，避免来源解释与当前提问脱节。
        return AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .contentRetriever(contentRetriever)
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
        // 辅助方法：单文档构建向量索引。
        // 关键对象名：document / splitter / segments / embeddings / embeddingStore。
        DocumentParser documentParser = new TextDocumentParser();
        Document document = loadDocument(documentPath, documentParser);

        // embed 子步骤 e2：文档切片。
        // 参数说明：
        // 300 控制块大小；0 表示不重叠。若你希望 sources 更连续，可适当增加 overlap。
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
