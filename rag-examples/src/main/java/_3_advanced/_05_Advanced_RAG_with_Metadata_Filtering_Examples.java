package _3_advanced;

import _2_naive.Naive_RAG_Example;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.MemoryId;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.filter.Filter;
import dev.langchain4j.store.embedding.filter.builder.sql.LanguageModelSqlFilterBuilder;
import dev.langchain4j.store.embedding.filter.builder.sql.TableDefinition;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.junit.jupiter.api.Test;
import shared.Assistant;
import shared.Utils;

import java.util.function.Function;

import static dev.langchain4j.data.document.Metadata.metadata;
import static dev.langchain4j.store.embedding.filter.MetadataFilterBuilder.metadataKey;
import static org.assertj.core.api.Assertions.assertThat;

class _05_Advanced_RAG_with_Metadata_Filtering_Examples {

    /**
     * 请先掌握 {@link Naive_RAG_Example} 的基础流程后再阅读本例。
     * <p>
     * 中文学习说明（元数据过滤 / Metadata Filtering）：
     * 在多租户或多域数据中，“检索前过滤”是非常关键的安全与质量手段。
     * 目标是：让向量检索只在允许的数据子集里进行，避免跨域污染与越权召回。
     * <p>
     * 本文件以单元测试形式演示 3 种常见策略：
     * 1) 静态过滤：固定规则，所有请求都一样。
     * 2) 动态过滤：根据本次请求上下文（如 userId）实时构建规则。
     * 3) LLM 生成过滤：让模型把自然语言约束转换为结构化过滤条件。
     * <p>
     * 更多资料：https://github.com/langchain4j/langchain4j/pull/610
     */

    /**
     * 关键对象：聊天模型（用于生成最终回答，也用于 LLM 生成过滤条件示例）。
     * <p>
     * 步骤 1：创建 OpenAI 聊天模型实例。
     * 子步骤 1.1：注入 API Key。
     * 子步骤 1.2：指定模型 gpt-4o-mini。
     * 子步骤 1.3：build() 后可被多个测试方法复用。
     */
    ChatModel chatModel = OpenAiChatModel.builder()
            .apiKey(Utils.OPENAI_API_KEY)
            .modelName("gpt-4o-mini")
            .build();

    /**
     * 关键对象：向量模型（负责把文本与查询映射到同一向量空间）。
     * <p>
     * 步骤 1：初始化 BGE embedding 模型。
     * 子步骤 1.1：后续每个测试都会复用它做文档向量化与查询向量化。
     */
    EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();

    /**
     * 静态元数据过滤示例。
     * <p>
     * 方法职责：演示如何在构建检索器时写死过滤条件（如只允许 animal=dog）。
     */
    @Test
    void Static_Metadata_Filter_Example() {

        // 步骤 1：构造两条带 metadata 的知识片段。
        // 子步骤 1.1：dogsSegment 文本为 dog 主题，metadata("animal","dog")。
        // 子步骤 1.2：birdsSegment 文本为 bird 主题，metadata("animal","bird")。
        // 子步骤 1.3：这两条数据将共同写入同一个向量库。
        TextSegment dogsSegment = TextSegment.from("Article about dogs ...", metadata("animal", "dog"));
        TextSegment birdsSegment = TextSegment.from("Article about birds ...", metadata("animal", "bird"));

        // 步骤 2：建立向量库并写入两条数据。
        // 子步骤 2.1：创建 InMemoryEmbeddingStore。
        // 子步骤 2.2：对 dogsSegment 向量化后写入。
        // 子步骤 2.3：对 birdsSegment 向量化后写入。
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.add(embeddingModel.embed(dogsSegment).content(), dogsSegment);
        embeddingStore.add(embeddingModel.embed(birdsSegment).content(), birdsSegment);
        // 子步骤 2.4：当前库中同时存在 dog/bird 两类候选。

        // 步骤 3：定义静态过滤条件 onlyDogs。
        // 子步骤 3.1：metadataKey("animal").isEqualTo("dog") 固定筛选 dog。
        // 子步骤 3.2：该过滤条件与用户问题无关，每次请求都一致。
        Filter onlyDogs = metadataKey("animal").isEqualTo("dog");

        // 步骤 4：构建带静态过滤的检索器。
        // 子步骤 4.1：embeddingStore(...) 指定检索数据源。
        // 子步骤 4.2：embeddingModel(...) 指定查询向量化模型。
        // 子步骤 4.3：filter(onlyDogs) 在检索阶段仅允许 animal=dog 数据参与召回。
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .filter(onlyDogs) // 静态过滤：检索阶段只看 animal=dog 的片段
                .build();

        // 步骤 5：组装 Assistant。
        // 子步骤 5.1：注入 chatModel（回答生成）。
        // 子步骤 5.2：注入 contentRetriever（静态过滤检索）。
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .contentRetriever(contentRetriever)
                .build();

        // 步骤 6：执行提问。
        // 子步骤 6.1：用户问题 Which animal? 是故意模糊问法。
        // 子步骤 6.2：期望通过过滤器保证只会引用 dog 片段。
        String answer = assistant.answer("Which animal?");

        // 步骤 7：断言输出正确性。
        // 子步骤 7.1：必须包含 dog。
        // 子步骤 7.2：必须不包含 bird。
        assertThat(answer)
                .containsIgnoringCase("dog")
                .doesNotContainIgnoringCase("bird");
    }


    interface PersonalizedAssistant {

        /**
         * 带会话隔离（MemoryId）的聊天接口。
         * <p>
         * 关键对象：
         * - {@code userId}：会话/租户标识，用于动态过滤。
         * - {@code userMessage}：用户问题文本。
         *
         * @param userId 用户 ID（会写入 query metadata）
         * @param userMessage 用户输入问题
         * @return 结合过滤后知识生成的回答
         */
        String chat(@MemoryId String userId, @dev.langchain4j.service.UserMessage String userMessage);
    }

    /**
     * 动态元数据过滤示例。
     * <p>
     * 方法职责：根据每次请求上下文（当前 userId）实时生成过滤条件。
     */
    @Test
    void Dynamic_Metadata_Filter_Example() {

        // 步骤 1：准备多租户私有数据。
        // 子步骤 1.1：user1Info 带 metadata("userId","1")，内容是偏好绿色。
        // 子步骤 1.2：user2Info 带 metadata("userId","2")，内容是偏好红色。
        // 子步骤 1.3：目标是“同库共存，检索隔离”。
        TextSegment user1Info = TextSegment.from("My favorite color is green", metadata("userId", "1"));
        TextSegment user2Info = TextSegment.from("My favorite color is red", metadata("userId", "2"));

        // 步骤 2：写入向量库。
        // 子步骤 2.1：创建 InMemoryEmbeddingStore。
        // 子步骤 2.2：写入 user1 数据（向量 + 片段）。
        // 子步骤 2.3：写入 user2 数据（向量 + 片段）。
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.add(embeddingModel.embed(user1Info).content(), user1Info);
        embeddingStore.add(embeddingModel.embed(user2Info).content(), user2Info);
        // 子步骤 2.4：库中数据并存，但后续靠 dynamicFilter 做检索隔离。

        // 步骤 3：定义动态过滤函数 filterByUserId。
        // 子步骤 3.1：函数输入是 Query（每轮请求上下文）。
        // 子步骤 3.2：从 query.metadata().chatMemoryId() 读取当前会话ID。
        // 子步骤 3.3：输出 Filter：metadataKey("userId").isEqualTo(当前会话ID)。
        // 子步骤 3.4：这样每个会话只检索自己的数据。
        Function<Query, Filter> filterByUserId =
                // 核心表达式：按 query 的 chatMemoryId 动态生成过滤条件。
                (query) -> metadataKey("userId").isEqualTo(query.metadata().chatMemoryId().toString());

        // 步骤 4：构建动态过滤检索器。
        // 子步骤 4.1：dynamicFilter(filterByUserId) 声明“每次检索前动态算过滤规则”。
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                // 每次请求都会执行 filterByUserId(query)。
                .dynamicFilter(filterByUserId)
                .build();

        // 步骤 5：构建 PersonalizedAssistant 代理。
        // 子步骤 5.1：接口方法 chat(userId, userMessage) 中 userId 会进入 @MemoryId。
        // 子步骤 5.2：@MemoryId 会成为 query metadata 的 chatMemoryId。
        // 子步骤 5.3：从而与 dynamicFilter 逻辑联动，实现“按用户隔离检索”。
        PersonalizedAssistant personalizedAssistant = AiServices.builder(PersonalizedAssistant.class)
                .chatModel(chatModel)
                .contentRetriever(contentRetriever)
                .build();

        // 步骤 6：用户 1 提问并验证结果。
        // 子步骤 6.1：传入 userId="1"。
        // 子步骤 6.2：期望只检索到绿色信息，不出现红色信息。
        String answer1 = personalizedAssistant.chat("1", "Which color would be best for a dress?");

        assertThat(answer1)
                .containsIgnoringCase("green")
                .doesNotContainIgnoringCase("red");

        // 步骤 7：用户 2 提问并验证结果。
        // 子步骤 7.1：传入 userId="2"。
        // 子步骤 7.2：期望只检索到红色信息，不出现绿色信息。
        String answer2 = personalizedAssistant.chat("2", "Which color would be best for a dress?");

        assertThat(answer2)
                .containsIgnoringCase("red")
                .doesNotContainIgnoringCase("green");
    }

    /**
     * LLM 生成元数据过滤示例。
     * <p>
     * 方法职责：把自然语言约束交给 LLM 转换为结构化过滤表达式，再用于检索阶段。
     */
    @Test
    void LLM_generated_Metadata_Filter_Example() {

        // 步骤 1：准备电影知识片段（带结构化 metadata）。
        // 子步骤 1.1：forrestGump：genre=drama, year=1994。
        // 子步骤 1.2：groundhogDay：genre=comedy, year=1993。
        // 子步骤 1.3：dieHard：genre=action, year=1998。
        TextSegment forrestGump = TextSegment.from("Forrest Gump", metadata("genre", "drama").put("year", 1994));
        TextSegment groundhogDay = TextSegment.from("Groundhog Day", metadata("genre", "comedy").put("year", 1993));
        TextSegment dieHard = TextSegment.from("Die Hard", metadata("genre", "action").put("year", 1998));

        // 步骤 2：定义“可过滤字段说明”（给 LLM 的 schema）。
        // 子步骤 2.1：表名 movies（语义化描述数据集合）。
        // 子步骤 2.2：列 genre，类型 VARCHAR，并提示合法值集合。
        // 子步骤 2.3：列 year，类型 INT。
        // 子步骤 2.4：这个定义帮助 LLM 把自然语言约束转成结构化过滤表达式。
        TableDefinition tableDefinition = TableDefinition.builder()
                .name("movies")
                .addColumn("genre", "VARCHAR", "one of: [comedy, drama, action]")
                .addColumn("year", "INT")
                .build();

        // 步骤 3：创建 SQL 风格过滤构建器 sqlFilterBuilder。
        // 子步骤 3.1：输入用户 query。
        // 子步骤 3.2：输出 Filter 对象（可用于 embedding 检索过滤）。
        LanguageModelSqlFilterBuilder sqlFilterBuilder = new LanguageModelSqlFilterBuilder(chatModel, tableDefinition);

        // 步骤 4：写入向量库。
        // 子步骤 4.1：创建 InMemoryEmbeddingStore。
        // 子步骤 4.2：分别向量化并写入 3 部电影片段。
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.add(embeddingModel.embed(forrestGump).content(), forrestGump);
        embeddingStore.add(embeddingModel.embed(groundhogDay).content(), groundhogDay);
        embeddingStore.add(embeddingModel.embed(dieHard).content(), dieHard);

        // 步骤 5：构建“LLM 生成过滤条件”的检索器。
        // 子步骤 5.1：dynamicFilter(query -> sqlFilterBuilder.build(query))。
        // 子步骤 5.2：即每轮提问时，先让 LLM 把问题转成过滤规则。
        // 子步骤 5.3：再在向量检索阶段应用该规则缩小候选集合。
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                // LLM 根据用户问题动态生成过滤表达式（类似 SQL WHERE）。
                .dynamicFilter(query -> sqlFilterBuilder.build(query))
                .build();

        // 步骤 6：装配 Assistant 作为最终问答入口。
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .contentRetriever(contentRetriever)
                .build();

        // 步骤 7：执行自然语言约束提问。
        // 子步骤 7.1：问题是 “Recommend me a good drama from 90s”。
        // 子步骤 7.2：预期过滤后只保留 genre=drama 且 year 在 1990s 的条目。
        String answer = assistant.answer("Recommend me a good drama from 90s");

        // 步骤 8：断言结果只包含目标电影。
        // 子步骤 8.1：应包含 Forrest Gump（drama, 1994）。
        // 子步骤 8.2：不应包含 Groundhog Day（comedy）。
        // 子步骤 8.3：不应包含 Die Hard（action）。
        assertThat(answer)
                .containsIgnoringCase("Forrest Gump")
                .doesNotContainIgnoringCase("Groundhog Day")
                .doesNotContainIgnoringCase("Die Hard");
    }
}
