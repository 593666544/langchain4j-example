package _3_advanced;

import _2_naive.Naive_RAG_Example;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import shared.Assistant;

import java.nio.file.Path;
import java.util.Collection;
import java.util.List;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static java.util.Collections.emptyList;
import static java.util.Collections.singletonList;
import static shared.Utils.*;

public class _06_Advanced_RAG_Skip_Retrieval_Example {


    /**
     * 请先掌握 {@link Naive_RAG_Example} 的基础流程后再阅读本例。
     * <p>
     * Advanced RAG 设计可参考：https://github.com/langchain4j/langchain4j/pull/538
     * <p>
     * 中文学习说明（按条件跳过检索 / Skip Retrieval）：
     * 不是所有请求都值得走 RAG。比如“Hi / 你好 / 谢谢”这类寒暄，检索只会增加成本与延迟。
     *
     * 本例用自定义 QueryRouter 做决策：
     * - 若应跳过检索，返回空 retriever 列表；
     * - 若应执行检索，返回目标 retriever。
     *
     * 决策来源可以是规则、关键词、分类器或 LLM。
     * 这里演示 LLM 决策法，便于快速验证效果。
     */
    /**
     * 示例程序入口。
     * <p>
     * 关键对象：
     * - {@code assistant}：带“可跳过检索”决策能力的助手。
     *
     * @param args 命令行参数（本示例未使用）
     */
    public static void main(String[] args) {

        Assistant assistant = createAssistant();

        // 建议测试：
        // 1) 输入 "Hi"：通常应跳过检索
        // 2) 输入 "Can I cancel my reservation?"：应走检索流程
        startConversationWith(assistant);
    }

    /**
     * 创建“可条件跳过检索”的 Assistant。
     * <p>
     * 关键对象：
     * - {@code contentRetriever}：正常业务问答时使用的文档检索器。
     * - {@code queryRouter}：判定当前问题是否需要检索。
     * - {@code retrievalAugmentor}：执行路由决策并驱动 RAG 流程。
     *
     * @return 支持 Skip Retrieval 的 Assistant
     */
    private static Assistant createAssistant() {

        // 子步骤 6.1：初始化向量模型。
        // 关键对象名：embeddingModel（文档与查询共用向量模型）。
        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();

        // 子步骤 6.2：构建默认文档检索器。
        // 先准备一个常规文档检索器（供“需要检索”的请求使用）。
        EmbeddingStore<TextSegment> embeddingStore =
                embed(toPath("documents/miles-of-smiles-terms-of-use.txt"), embeddingModel);

        // 关键对象名：contentRetriever（业务相关问题的默认检索器）。
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.6)
                .build();

        // 子步骤 6.3：初始化聊天模型（用于判定与回答）。
        // 关键对象名：chatModel（用于“是否检索”判定 + 最终回答）。
        ChatModel chatModel = OpenAiChatModel.builder()
                .apiKey(OPENAI_API_KEY)
                .modelName("gpt-4o-mini")
                .build();

        // 子步骤 6.4：实现 QueryRouter 决策逻辑。
        // 自定义路由器：在这里实现“是否检索”的决策逻辑。
        // 关键对象名：queryRouter（核心决策组件）。
        QueryRouter queryRouter = new QueryRouter() {

            // 用一个极简二分类提示词，让 LLM 只回答 yes/no/maybe。
            private final PromptTemplate PROMPT_TEMPLATE = PromptTemplate.from(
                    "Is the following query related to the business of the car rental company? " +
                            "Answer only 'yes', 'no' or 'maybe'. " +
                            "Query: {{it}}"
            );

            /**
             * 路由决策入口。
             * <p>
             * 方法职责：
             * 1. 用 LLM 判断问题是否与业务相关；
             * 2. 若不相关返回空列表（跳过检索）；
             * 3. 否则返回默认文档检索器。
             *
             * @param query 当前用户问题（含 query metadata）
             * @return 需要执行的检索器集合
             */
            @Override
            public Collection<ContentRetriever> route(Query query) {

                // route 子步骤 a：把 query 文本填入判定模板。
                Prompt prompt = PROMPT_TEMPLATE.apply(query.text());

                // route 子步骤 b：调用 LLM 获取“yes/no/maybe”判定结果。
                AiMessage aiMessage = chatModel.chat(prompt.toUserMessage()).aiMessage();
                System.out.println("LLM decided: " + aiMessage.text());

                // 决策为 no 时，直接返回空列表 => 本轮完全跳过检索。
                if (aiMessage.text().toLowerCase().contains("no")) {
                    return emptyList();
                }

                // 其余情况默认走文档检索器。
                // route 子步骤 c：返回要执行的检索器列表。
                return singletonList(contentRetriever);
            }
        };

        // 子步骤 6.5：把路由器装进 RetrievalAugmentor。
        // 将路由器装进 RetrievalAugmentor。
        // 关键对象名：retrievalAugmentor（执行路由决策）。
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // 子步骤 6.6：组装并返回 Assistant。
        // 组装 AI Service。
        return AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();
    }

    /**
     * 将单个文档构建为向量存储。
     * <p>
     * 关键对象：
     * - {@code segments}：切片文本集合；
     * - {@code embeddings}：切片对应向量；
     * - {@code embeddingStore}：向量库（保存向量到文本的映射）。
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
