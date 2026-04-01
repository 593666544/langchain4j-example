package _1_easy;

import _2_naive.Naive_RAG_Example;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import shared.Assistant;

import java.util.List;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocuments;
import static shared.Utils.*;

public class Easy_RAG_Example {

    /**
     * 负责“最终回答”的聊天模型（LLM）。
     * <p>
     * 步骤 1：指定 API Key（身份凭据）。
     * 子步骤 1.1：读取 shared.Utils.OPENAI_API_KEY。
     * 子步骤 1.2：若你没配置环境变量，会使用 demo 占位值（真实请求会失败）。
     * <p>
     * 步骤 2：指定模型名 gpt-4o-mini。
     * 子步骤 2.1：这是“生成最终回答”的模型，不是 embedding 模型。
     * <p>
     * 步骤 3：build() 得到 ChatModel 实例。
     * 子步骤 3.1：后续会注入到 AiServices 中，用于每轮回答生成。
     */
    private static final ChatModel CHAT_MODEL = OpenAiChatModel.builder()
            .apiKey(OPENAI_API_KEY)
            .modelName("gpt-4o-mini")
            .build();

    /**
     * 中文学习说明（Easy RAG）：
     * 该示例展示“最低心智负担”的 RAG 接入方式。
     * <p>
     * Easy 的核心是“先跑通闭环，再下钻细节”：
     * - 不手写文档切分、向量化、索引入库等底层步骤；
     * - 借助高层 API 一次性完成 ingestion + retrieval；
     * - 先把输入问题 -> 检索 -> 回答这个主链路跑起来。
     * <p>
     * 适合场景：
     * - 你刚开始学 RAG，先跑通“文档问答”闭环；
     * - 你要先做 PoC（概念验证），后续再逐步替换为高级策略。
     * <p>
     * 如果你希望理解每个原子步骤如何工作，请继续学习 {@link Naive_RAG_Example}。
     */
    /**
     * 示例程序入口。
     * <p>
     * 关键对象：
     * - {@code documents}：知识库原始文档集合（输入语料）。
     * - {@code assistant}：最终可调用的 AI Service 代理对象（对外问答入口）。
     * <p>
     * 方法职责：
     * 1. 加载文档；
     * 2. 组装带检索能力的 Assistant；
     * 3. 启动命令行多轮对话。
     *
     * @param args 命令行参数（本示例未使用）
     */
    public static void main(String[] args) {

        // 步骤 1：加载知识文档（输入语料）。
        // 子步骤 1.1：toPath("documents/") 把 resources/documents 转成绝对路径。
        // 子步骤 1.2：glob("*.txt") 只匹配 txt 文件，避免把其他文件混入知识库。
        // 子步骤 1.3：loadDocuments(...) 批量读取文档，返回 List<Document>。
        // 子步骤 1.4：每个 Document 代表一份“还没切片”的原始文本。
        List<Document> documents = loadDocuments(toPath("documents/"), glob("*.txt"));

        // 步骤 2：构建 AI Service（Assistant 代理对象）。
        // 子步骤 2.1：AiServices.builder(Assistant.class) 以接口定义“能力契约”。
        // 子步骤 2.2：chatModel(CHAT_MODEL) 注入“答案生成器”。
        // 子步骤 2.3：chatMemory(...) 注入“多轮上下文窗口记忆”。
        // 子步骤 2.4：contentRetriever(...) 注入“每轮召回知识片段”的检索器。
        // 子步骤 2.5：build() 生成运行时代理对象，赋值给 Assistant assistant。
        Assistant assistant = AiServices.builder(Assistant.class)
                // 子步骤 2.2 细化：chatModel 是最终自然语言答案生成器（LLM）。
                .chatModel(CHAT_MODEL)
                // 子步骤 2.3 细化：MessageWindowChatMemory.withMaxMessages(10)
                // 仅保留最近 10 条消息，用于控制 token 成本与上下文长度。
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                // 子步骤 2.4 细化：contentRetriever 是 RAG 检索入口。
                // 每次提问时会做语义检索，召回片段后注入 prompt 作为回答依据。
                .contentRetriever(createContentRetriever(documents))
                .build();

        // 步骤 3：启动命令行对话（输出观测层）。
        // 子步骤 3.1：startConversationWith(assistant) 进入 REPL 循环。
        // 子步骤 3.2：每轮从控制台读用户输入 -> 调 assistant.answer(...) -> 打印回答。
        // 子步骤 3.3：输入 exit 结束循环。
        startConversationWith(assistant);
    }

    /**
     * 根据文档集合创建内容检索器（ContentRetriever）。
     * <p>
     * 关键对象：
     * - {@code InMemoryEmbeddingStore<TextSegment>}：内存向量库，保存“切片文本 + 向量”。
     * - {@code EmbeddingStoreIngestor}：摄取器，负责把文档加工并写入向量库。
     * - {@code EmbeddingStoreContentRetriever}：检索器实现，负责按语义相似度召回片段。
     *
     * @param documents 待摄取的知识文档集合
     * @return 可被 RAG 流程调用的内容检索器
     */
    private static ContentRetriever createContentRetriever(List<Document> documents) {

        // 步骤 1：创建向量库对象（内存实现）。
        // 子步骤 1.1：InMemoryEmbeddingStore<TextSegment> 表示“向量对应原文类型是 TextSegment”。
        // 子步骤 1.2：该对象内部保存“向量 + 对应文本切片”的映射关系。
        // 子步骤 1.3：后续检索时会返回最相似向量对应的 TextSegment。
        InMemoryEmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        // 步骤 2：执行摄取（Ingestion）。
        // 子步骤 2.1：读取 documents 中每个 Document 的文本内容。
        // 子步骤 2.2：按默认切分策略分块为 TextSegment。
        // 子步骤 2.3：对每个 TextSegment 计算 embedding 向量。
        // 子步骤 2.4：将“向量 + 原文切片”写入 embeddingStore。
        // 子步骤 2.5：这一行是 Easy RAG 的核心抽象：把多步底层流程打包成一个调用。
        EmbeddingStoreIngestor.ingest(documents, embeddingStore);

        // 步骤 3：由向量库构建检索器。
        // 子步骤 3.1：返回接口类型 ContentRetriever，便于上层解耦。
        // 子步骤 3.2：具体实现是 EmbeddingStoreContentRetriever。
        // 子步骤 3.3：运行时每轮会执行“问题向量化 -> 相似度检索 -> 返回候选片段”。
        return EmbeddingStoreContentRetriever.from(embeddingStore);
    }
}
