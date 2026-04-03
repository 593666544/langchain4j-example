package _2_naive;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import shared.Assistant;

import java.util.List;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static shared.Utils.*;

public class Naive_RAG_Example {

    /**
     * 中文学习说明（Naive RAG）：
     * 这是“手动版”RAG，目标是让你看清完整管线，而不是追求最强效果。
     *
     * 每轮问答的核心链路：
     * 1. 接收用户原始问题（不做查询改写）。
     * 2. 将问题向量化。
     * 3. 到向量库检索最相近的文本片段（TopK）。
     * 4. 将这些片段拼接进提示词上下文。
     * 5. 交给聊天模型生成答案。
     *
     * “Naive”的局限：
     * - 若用户问题表达不完整（如“他什么时候出生？”），检索可能失败。
     * - 检索到的片段可能带噪声，影响回答质量。
     *
     * 但它是学习 RAG 的最佳起点：所有关键零件都明确可见。
     */
    /**
     * 示例程序入口。
     * <p>
     * 关键对象：
     * - {@code assistant}：封装了聊天模型、检索器与记忆的 AI Service。
     * <p>
     * 方法职责：
     * 1. 创建 Naive RAG 助手；
     * 2. 进入命令行多轮问答。
     *
     * @param args 命令行参数（本示例未使用）
     */
    public static void main(String[] args) {

        // 步骤 1：通过工厂方法创建 Assistant。
        // 子步骤 1.1：传入 documentPath，声明“哪份文档是知识源”。
        // 子步骤 1.2：createAssistant(...) 内部会完成“切片 -> 向量化 -> 入库 -> 检索器装配”。
        Assistant assistant = createAssistant("documents/miles-of-smiles-terms-of-use.txt");

        // 步骤 2：启动交互式问答循环。
        // 子步骤 2.1：你在控制台输入问题。
        // 子步骤 2.2：内部调用 assistant.answer(...) 触发 Naive RAG 检索与回答流程。
        // 子步骤 2.3：输入 exit 退出。
        startConversationWith(assistant);
    }

    /**
     * 创建一个可运行的 Naive RAG 助手。
     * <p>
     * 关键对象：
     * - {@code ChatModel chatModel}：最终答案生成器。
     * - {@code Document document}：知识源文档。
     * - {@code List<TextSegment> segments}：切片后的候选知识块。
     * - {@code EmbeddingModel embeddingModel}：文本向量化模型。
     * - {@code EmbeddingStore<TextSegment> embeddingStore}：向量存储（语义检索底座）。
     * - {@code ContentRetriever contentRetriever}：检索器（每轮按问题召回相关片段）。
     * - {@code ChatMemory chatMemory}：对话上下文记忆。
     *
     * @param documentPath 资源目录下的文档相对路径
     * @return 装配完成的 Assistant 代理对象
     */
    private static Assistant createAssistant(String documentPath) {

        // 步骤 1：创建聊天模型 ChatModel（回答生成器）。
        // 子步骤 1.1：OpenAiChatModel.builder() 进入构建器模式。
        // 子步骤 1.2：apiKey(...) 注入访问凭据。
        // 子步骤 1.3：modelName("gpt-4o-mini") 指定回答模型。
        // 子步骤 1.4：build() 生成不可变模型实例。
        ChatModel chatModel = OpenAiChatModel.builder()
                .apiKey(OPENAI_API_KEY)
                .modelName("gpt-4o-mini")
                .build();


        // 步骤 2：加载知识文档（原始语料输入）。
        // 子步骤 2.1：创建 DocumentParser，决定“文件如何转成 Document”。
        // 子步骤 2.2：TextDocumentParser 表示按纯文本读取，不做结构化解析。
        // 子步骤 2.3：loadDocument(...) 返回 Document 对象。
        // 子步骤 2.4：该 Document 仍是整篇内容，尚未切片。
        DocumentParser documentParser = new TextDocumentParser();
        Document document = loadDocument(toPath(documentPath), documentParser);


        // 步骤 3：文档切片（Chunking）。
        // 子步骤 3.1：DocumentSplitters.recursive(300, 0) 创建递归切分器。
        // 子步骤 3.2：300 表示目标块大小（字符/Token近似，取决于实现）。
        //            作用：决定“每个检索单元装多少信息”。
        //            - 过大：语义太杂，检索命中后会带入更多噪声，且上下文成本上升；
        //            - 过小：语义可能不完整，检索容易命中碎片，回答缺关键信息。
        // 子步骤 3.3：0 表示块间无重叠（教学场景先简化）。
        //            作用：避免相邻块重复内容，降低索引体积与检索冗余。
        //            代价：跨段信息可能被“切断”，真实生产常设 20~80 的 overlap 兜底。
        // 子步骤 3.4：split(document) 输出 List<TextSegment>。
        // 子步骤 3.5：每个 TextSegment 将成为向量索引的最小检索单元。
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 0);
        List<TextSegment> segments = splitter.split(document);


        // 步骤 4：文本向量化（Embedding）。
        // 子步骤 4.1：new BgeSmallEnV15QuantizedEmbeddingModel() 初始化 embedding 模型。
        //            作用：把文本映射到向量空间，供“相似度检索”使用。
        //            为什么要单独一个 embedding 模型：检索阶段比的是向量距离，不是原文字符串。
        // 子步骤 4.2：embedAll(segments) 对所有切片批量向量化。
        //            批量处理的意义：减少逐条调用开销，入库效率更高。
        // 子步骤 4.3：content() 取出 List<Embedding>。
        // 子步骤 4.4：embeddings 与 segments 按索引一一对应。
        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();


        // 步骤 5：写入向量库（Embedding Store）。
        // 子步骤 5.1：创建 InMemoryEmbeddingStore（内存向量库实现）。
        // 子步骤 5.2：addAll(embeddings, segments) 批量写入“向量 + 原文切片”映射。
        // 子步骤 5.3：后续检索命中向量后即可反查对应 TextSegment 内容。
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // 注：上面“切片 + 向量化 + 入库”也可用 EmbeddingStoreIngestor 一键完成。
        // 本例故意手写，是为了帮助你看清每个阶段。


        // 步骤 6：构建内容检索器（RAG 查询入口）。
        // 子步骤 6.1：embeddingStore(...) 声明“到哪个向量库检索”。
        // 子步骤 6.2：embeddingModel(...) 声明“如何把用户问题转查询向量”。
        // 子步骤 6.3：maxResults(2) 限制召回条数为 2，降低噪声与 token 成本。
        //            调参经验：数值越大召回率通常越高，但 prompt 会更长、噪声也更容易混入。
        // 子步骤 6.4：minScore(0.5) 设置最低相似度阈值，过滤弱相关片段。
        //            调参经验：阈值越高越“保守”（更干净但可能漏召回），越低越“宽松”（更全但更杂）。
        // 子步骤 6.5：build() 得到 ContentRetriever 实例。
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                // 子步骤 6.1 代码位。
                .embeddingStore(embeddingStore)
                // 子步骤 6.2 代码位。
                .embeddingModel(embeddingModel)
                .maxResults(2) // 子步骤 6.3：最多召回 2 条
                .minScore(0.5) // 子步骤 6.4：低于阈值直接丢弃
                .build();


        // 步骤 7：配置对话记忆（可选但推荐）。
        // 子步骤 7.1：MessageWindowChatMemory.withMaxMessages(10) 创建窗口记忆。
        // 子步骤 7.2：仅保留最近 10 条消息，避免上下文无限膨胀。
        //            参数作用：窗口越大，上下文连续性越强；但 token 成本和“历史噪声”也会增加。
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);


        // 步骤 8：组装 AI Service 代理对象（输出）。
        // 子步骤 8.1：AiServices.builder(Assistant.class) 绑定接口契约。
        // 子步骤 8.2：chatModel(...) 注入回答模型。
        // 子步骤 8.3：contentRetriever(...) 注入检索器。
        // 子步骤 8.4：chatMemory(...) 注入多轮记忆。
        // 子步骤 8.5：build() 返回运行时代理，实现 Assistant.answer(...)。
        //            这样分层的好处：回答模型、检索策略、记忆策略可独立替换，不互相耦合。
        return AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .contentRetriever(contentRetriever)
                .chatMemory(chatMemory)
                .build();
    }
}
