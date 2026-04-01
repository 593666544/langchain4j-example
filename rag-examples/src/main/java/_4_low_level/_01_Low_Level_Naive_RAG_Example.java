package _4_low_level;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiTokenCountEstimator;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static java.util.stream.Collectors.joining;
import static shared.Utils.OPENAI_API_KEY;
import static shared.Utils.toPath;

public class _01_Low_Level_Naive_RAG_Example {

    /**
     * 中文学习说明（Low-level Naive RAG）：
     * 该示例展示不依赖 AI Service 封装，直接使用底层 API 手工搭建 RAG 流程。
     * <p>
     * 与高层写法的区别：
     * - 高层（AiServices）会自动帮你做检索注入与对话编排；
     * - 低层写法需要你自己构造检索请求、拼 Prompt、发送消息。
     * <p>
     * 适合学习目标：
     * - 想彻底理解 RAG 执行细节；
     * - 想自定义 prompt 模板与检索策略。
     */
    /**
     * 低层 API 示例入口。
     * <p>
     * 关键对象：
     * - {@code splitter}：文档切片器。
     * - {@code embeddingModel}：向量模型。
     * - {@code embeddingStore}：向量库。
     * - {@code embeddingSearchRequest}：检索请求（包含 TopK 与阈值）。
     * - {@code promptTemplate/prompt}：手工构建的增强提示词。
     * - {@code chatModel}：最终回答模型。
     * <p>
     * 方法职责：完整展示“加载 -> 切片 -> 向量化 -> 检索 -> 注入 -> 生成”全过程。
     *
     * @param args 命令行参数（本示例未使用）
     */
    public static void main(String[] args) {

        // 步骤 1：加载知识文档（输入阶段）。
        // 子步骤 1.1：创建 TextDocumentParser，定义“文件 -> Document”的解析规则。
        // 子步骤 1.2：toPath(...) 定位 resources 目录中的目标文本文件。
        // 子步骤 1.3：loadDocument(...) 真正读取文件并构造 Document 对象。
        // 子步骤 1.4：此时 document 是整篇原文，尚未被切成检索块。
        DocumentParser documentParser = new TextDocumentParser();
        Document document = loadDocument(toPath("example-files/story-about-happy-carrot.txt"), documentParser);

        // 步骤 2：文档切片（Chunking 阶段）。
        // 子步骤 2.1：DocumentSplitters.recursive(...) 创建递归切分器。
        // 子步骤 2.2：300 代表目标片段大小（近似 token/字符控制）。
        // 子步骤 2.3：0 代表块间不重叠。
        // 子步骤 2.4：OpenAiTokenCountEstimator("gpt-4o-mini") 让切分更贴近目标模型 token 计数。
        // 子步骤 2.5：split(document) 执行切分，输出 List<TextSegment>。
        DocumentSplitter splitter = DocumentSplitters.recursive(
                300,
                0,
                new OpenAiTokenCountEstimator("gpt-4o-mini")
        );
        List<TextSegment> segments = splitter.split(document);

        // 步骤 3：向量化片段（Embedding 阶段）。
        // 子步骤 3.1：初始化 embeddingModel（BGE small 量化版）。
        // 子步骤 3.2：embedAll(segments) 对全部片段做批量向量化。
        // 子步骤 3.3：content() 取出 List<Embedding>。
        // 子步骤 3.4：embeddings 与 segments 的索引一一对齐。
        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        // 步骤 4：写入向量库（Indexing 阶段）。
        // 子步骤 4.1：创建 InMemoryEmbeddingStore（内存版向量库）。
        // 子步骤 4.2：addAll(...) 批量写入“向量 + 文本片段”关联数据。
        // 子步骤 4.3：后续 search(...) 将基于这些索引做相似度匹配。
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // 步骤 5：准备用户问题（Query 输入阶段）。
        // 子步骤 5.1：question 是本轮用户自然语言问题。
        // 子步骤 5.2：后续会先向量化 question，再去向量库检索相关片段。
        String question = "Who is Charlie?";

        // 步骤 6：把问题向量化（Query Embedding）。
        // 子步骤 6.1：embeddingModel.embed(question) 产出查询向量。
        // 子步骤 6.2：content() 拿到 Embedding 实体，命名为 questionEmbedding。
        Embedding questionEmbedding = embeddingModel.embed(question).content();

        // 步骤 7：执行向量检索（Retrieval 阶段）。
        // 子步骤 7.1：创建 EmbeddingSearchRequest（检索参数对象）。
        // 子步骤 7.2：queryEmbedding(questionEmbedding) 注入查询向量。
        // 子步骤 7.3：maxResults(3) 指定最多召回 3 条候选片段。
        // 子步骤 7.4：minScore(0.7) 过滤低相关结果。
        // 子步骤 7.5：embeddingStore.search(request).matches() 获取命中列表。
        // 子步骤 7.6：relevantEmbeddings 类型是 List<EmbeddingMatch<TextSegment>>，
        //            每个元素包含：匹配分数 + 命中文本片段。
        EmbeddingSearchRequest embeddingSearchRequest = EmbeddingSearchRequest.builder()
                .queryEmbedding(questionEmbedding)
                .maxResults(3)
                .minScore(0.7)
                .build();
        List<EmbeddingMatch<TextSegment>> relevantEmbeddings = embeddingStore.search(embeddingSearchRequest).matches();

        // 步骤 8：手动构造增强 Prompt（Augmentation 阶段）。
        // 子步骤 8.1：定义 PromptTemplate，包含两个占位符：{{question}} 与 {{information}}。
        // 子步骤 8.2：模板语义是“请基于给定信息回答问题”。
        PromptTemplate promptTemplate = PromptTemplate.from(
                "Answer the following question to the best of your ability:\n"
                        + "\n"
                        + "Question:\n"
                        + "{{question}}\n"
                        + "\n"
                        + "Base your answer on the following information:\n"
                        + "{{information}}");

        // 子步骤 8.3：把命中片段拼接为 information 文本上下文。
        // 说明：joining("\n\n") 使用双换行分段，便于模型区分不同片段。
        String information = relevantEmbeddings.stream()
                .map(match -> match.embedded().text())
                .collect(joining("\n\n"));

        // 子步骤 8.4：构造变量映射 variables（占位符 -> 实际值）。
        Map<String, Object> variables = new HashMap<>();
        // 子步骤 8.5：填充 question 占位符。
        variables.put("question", question);
        // 子步骤 8.6：填充 information 占位符。
        variables.put("information", information);

        // 子步骤 8.7：应用模板得到最终 Prompt 对象（可发送给聊天模型）。
        Prompt prompt = promptTemplate.apply(variables);

        // 步骤 9：调用聊天模型生成答案（Generation 阶段）。
        // 子步骤 9.1：创建 OpenAiChatModel（回答生成器）。
        // 子步骤 9.2：timeout(60s) 防止网络慢时无限等待。
        // 子步骤 9.3：prompt.toUserMessage() 把 Prompt 转成用户消息格式。
        // 子步骤 9.4：chatModel.chat(...) 发送请求并获取响应。
        // 子步骤 9.5：aiMessage 保存模型返回的消息对象。
        ChatModel chatModel = OpenAiChatModel.builder()
                .apiKey(OPENAI_API_KEY)
                .modelName("gpt-4o-mini")
                .timeout(Duration.ofSeconds(60))
                .build();
        // toUserMessage() 将 Prompt 转为标准用户消息结构，再发给 chatModel。
        AiMessage aiMessage = chatModel.chat(prompt.toUserMessage()).aiMessage();

        // 步骤 10：提取并输出最终答案（输出阶段）。
        // 子步骤 10.1：aiMessage.text() 拿到纯文本答案。
        // 子步骤 10.2：System.out.println(...) 打印到控制台。
        String answer = aiMessage.text();
        System.out.println(answer); // Charlie is a cheerful carrot living in VeggieVille...
    }
}
