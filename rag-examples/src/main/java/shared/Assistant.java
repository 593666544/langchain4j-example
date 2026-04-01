package shared;

/**
 * 中文学习说明（硬核逐行版）：
 * 这个接口文件虽然代码只有几行，但它在整个 rag-examples 里是“总入口契约”。
 * <p>
 * 步骤 1：定义一个纯 Java 接口（没有实现类）。
 * 子步骤 1.1：接口名叫 Assistant，语义是“可回答问题的助手”。
 * 子步骤 1.2：接口方法叫 answer(...)，输入是用户问题字符串，输出是回答字符串。
 * <p>
 * 步骤 2：在其它示例中用 AiServices.builder(Assistant.class) 生成运行时代理。
 * 子步骤 2.1：你只写“能力声明”（接口），不写“调用细节”（HTTP 请求、prompt 拼接等）。
 * 子步骤 2.2：把 ChatModel、ContentRetriever、ChatMemory 这些组件装配给 builder。
 * 子步骤 2.3：builder.build() 后拿到的就是 Assistant 的代理实现对象。
 * <p>
 * 步骤 3：业务代码直接 assistant.answer("你的问题")。
 * 子步骤 3.1：LangChain4j 内部会自动执行“构造提示词 ->（可选）检索 -> 调用模型 -> 返回文本”。
 * 子步骤 3.2：因此你在业务层看到的是一个普通 Java 方法调用体验。
 * <p>
 * 关键对象名称与职责：
 * - Assistant（接口名）：能力契约（Contract），定义“外部可调用什么能力”。
 * - answer(String query)（方法名）：统一问答入口，模块内几乎所有 demo 都依赖它。
 * - query（参数名）：用户原始自然语言输入。
 * - String（返回类型）：最终输出给用户的自然语言答案。
 * <p>
 * 设计价值：
 * - 解耦：把“调用方式”与“模型/RAG 细节”分离。
 * - 一致性：所有示例都能复用同一个对话接口。
 * - 可测试：接口天然易于 mock 或替换实现。
 * <p>
 * More info: https://docs.langchain4j.dev/tutorials/ai-services
 */
public interface Assistant {

    /**
     * 向助手提问并获得答案（硬核说明）。
     * <p>
     * 步骤 1：接收用户问题文本。
     * 子步骤 1.1：调用方把自然语言问题放进 query。
     * 子步骤 1.2：query 不限制格式，可以是英文、中文或混合内容。
     * <p>
     * 步骤 2：由运行时代理执行内部流程。
     * 子步骤 2.1：如果装配了检索器，会先执行 RAG 召回相关片段。
     * 子步骤 2.2：如果装配了记忆，会注入最近对话上下文。
     * 子步骤 2.3：调用 ChatModel 生成最终答案文本。
     * <p>
     * 步骤 3：返回字符串结果给调用方。
     * 子步骤 3.1：返回值是纯文本，不是结构化 JSON。
     * 子步骤 3.2：调用方可以直接打印、存库、或回传到前端。
     *
     * @param query 用户输入的问题
     * @return 代理执行完整链路后得到的答案文本
     */
    String answer(String query);
}
