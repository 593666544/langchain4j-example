package _3_advanced;

import _2_naive.Naive_RAG_Example;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.cohere.CohereScoringModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.scoring.ScoringModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.aggregator.ContentAggregator;
import dev.langchain4j.rag.content.aggregator.ReRankingContentAggregator;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import shared.Assistant;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static shared.Utils.*;

public class _03_Advanced_RAG_with_ReRanking_Example {

    /**
     * 请先掌握 {@link Naive_RAG_Example} 的基础流程后再阅读本例。
     * <p>
     * Advanced RAG 设计可参考：https://github.com/langchain4j/langchain4j/pull/538
     * <p>
     * 中文学习说明（重排 / Re-ranking）：
     * 初次召回（first-stage retrieval）通常追求速度，可能会混入“看起来相关但其实无关”的片段。
     * 如果把这些噪声都交给 LLM，会导致：
     * - prompt 成本上升
     * - 注意力被干扰
     * - 回答偏题甚至幻觉风险增加
     *
     * Re-ranking 的思路是“两段式筛选”：
     * 1. 先宽松召回更多候选（例如 top5）
     * 2. 再用更强评分模型重新排序并过滤低分结果
     *
     * 这是一种典型的“速度与质量平衡”策略。
     * <p>
     * 本示例依赖 `langchain4j-cohere`（使用 Cohere rerank 模型）。
     */
    /**
     * 示例程序入口。
     * <p>
     * 关键对象：
     * - {@code assistant}：带重排能力的 RAG 助手。
     *
     * @param args 命令行参数（本示例未使用）
     */
    public static void main(String[] args) {

        Assistant assistant = createAssistant("documents/miles-of-smiles-terms-of-use.txt");

        // 建议测试：
        // 1) 先问 "Hi"：初检索的噪声可能被重排全部过滤掉
        // 2) 再问 "Can I cancel my reservation?"：观察仅保留高相关片段
        startConversationWith(assistant);
    }

    /**
     * 创建“重排版”Assistant。
     * <p>
     * 关键对象：
     * - {@code contentRetriever}：第一阶段召回器（偏召回率）。
     * - {@code scoringModel}：第二阶段评分模型（偏精确率）。
     * - {@code contentAggregator}：按分数过滤与重排候选片段。
     * - {@code retrievalAugmentor}：串联“召回 + 重排”的编排器。
     *
     * @param documentPath 作为知识库输入的文档路径
     * @return 支持 Re-ranking 的 Assistant
     */
    private static Assistant createAssistant(String documentPath) {

        // 子步骤 3.1：准备文档与向量基础设施。
        // 基础准备：文档、向量模型、向量库。
        // 关键对象名：document / embeddingModel / embeddingStore。
        Document document = loadDocument(toPath(documentPath), new TextDocumentParser());

        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        // 子步骤 3.2：构建摄取器。
        // 摄取器：负责把原始文档加工成可检索向量索引。
        // 关键对象名：ingestor。
        // recursive(300, 0) 的含义：
        // - 300：让每个候选块尽量“可读且可检索”；
        // - 0：减少重复块，便于先观察重排本身的效果（避免 overlap 干扰实验）。
        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .documentSplitter(DocumentSplitters.recursive(300, 0))
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build();

        // 子步骤 3.3：执行摄取（文档 -> 索引）。
        ingestor.ingest(document);

        // 子步骤 3.4：第一阶段召回（偏召回率）。
        // 第一阶段：召回时先放宽，拿更多候选。
        // 关键对象名：contentRetriever（第一阶段召回器）。
        // 为什么 maxResults 设为 5：
        // 重排前先“多捞一些”候选，给第二阶段提供排序空间，否则重排价值会被削弱。
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(5) // 多召回一些，给第二阶段重排留余地
                .build();

        // 子步骤 3.5：第二阶段评分模型初始化（偏精确率）。
        // 第二阶段：重排模型（Cohere）。
        // 注册并获取免费 key：https://dashboard.cohere.com/welcome/register
        // 关键对象名：scoringModel（第二阶段精排评分器）。
        // 为什么不用原始相似度直接当最终排序：
        // 向量相似度是“粗排信号”，重排模型能利用更细粒度语义进行精排。
        ScoringModel scoringModel = CohereScoringModel.builder()
                .apiKey(System.getenv("COHERE_API_KEY"))
                .modelName("rerank-multilingual-v3.0")
                .build();

        // 子步骤 3.6：构建重排聚合器。
        // 根据分数阈值过滤低相关片段，只将高质量上下文送入 LLM。
        // 关键对象名：contentAggregator。
        // 参数取舍：
        // minScore(0.8) 更偏保守，宁可少给上下文，也尽量避免噪声干扰最终回答。
        ContentAggregator contentAggregator = ReRankingContentAggregator.builder()
                .scoringModel(scoringModel) // 用更强模型重新打分排序
                .minScore(0.8) // 只保留高相关内容，减少噪声注入
                .build();

        // 子步骤 3.7：组装 RetrievalAugmentor（召回 + 重排）。
        // 将 “初检索 + 重排聚合” 组装成完整 RetrievalAugmentor。
        // 关键对象名：retrievalAugmentor。
        // 为什么要显式配置 contentAggregator：
        // 默认聚合不会做二阶段重排，必须在 augmentor 中替换聚合策略才能生效。
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .contentRetriever(contentRetriever)
                .contentAggregator(contentAggregator)
                .build();

        // 子步骤 3.8：初始化最终回答模型。
        // 最终回答模型。
        // 关键对象名：model。
        ChatModel model = OpenAiChatModel.builder()
                .apiKey(OPENAI_API_KEY)
                .modelName("gpt-4o-mini")
                .build();

        // 子步骤 3.9：组装并返回 Assistant。
        // 组装 AI Service。
        return AiServices.builder(Assistant.class)
                .chatModel(model)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();
    }
}
