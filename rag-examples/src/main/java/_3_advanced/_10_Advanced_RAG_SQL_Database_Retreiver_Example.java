package _3_advanced;

import _2_naive.Naive_RAG_Example;
import dev.langchain4j.experimental.rag.content.retriever.sql.SqlDatabaseContentRetriever;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.service.AiServices;
import org.h2.jdbcx.JdbcDataSource;
import shared.Assistant;

import javax.sql.DataSource;
import java.io.IOException;
import java.nio.file.Files;
import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Statement;

import static shared.Utils.*;

public class _10_Advanced_RAG_SQL_Database_Retreiver_Example {


    /**
     * 请先掌握 {@link Naive_RAG_Example} 的基础流程后再阅读本例。
     * <p>
     * Advanced RAG 设计可参考：https://github.com/langchain4j/langchain4j/pull/538
     * <p>
     * 中文学习说明（SQL 数据库检索器）：
     * 本例演示如何让模型把自然语言问题转换为 SQL 查询，再从数据库取数作为上下文。
     * 这是“结构化数据 RAG”常见方案之一，尤其适合报表、统计、运营查询类问题。
     * <p>
     * 安全警告（务必阅读）：
     * {@link SqlDatabaseContentRetriever} 很强大，但默认存在安全风险。
     * 即使框架会做一定校验（例如尝试限制为 SELECT），也不能保证绝对安全。
     * 生产环境必须使用权限极小的只读账号，并进行严格审计与防护。
     * <p>
     * 本示例用 H2 内存库模拟 3 张表（customers / products / orders），
     * 表结构与测试数据位于 resources/sql 目录。
     * <p>
     * 本示例依赖 `langchain4j-experimental-sql`。
     */
    /**
     * 示例程序入口。
     * <p>
     * 关键对象：
     * - {@code assistant}：支持“自然语言 -> SQL -> 回答”的助手。
     *
     * @param args 命令行参数（本示例未使用）
     */
    public static void main(String[] args) {

        // 子步骤 10.1：创建 SQL 检索版 Assistant。
        Assistant assistant = createAssistant();

        // 可尝试问题：
        // - How many customers do we have?
        // - What is our top seller?
        startConversationWith(assistant);
    }

    /**
     * 创建 SQL Retriever 版 Assistant。
     * <p>
     * 关键对象：
     * - {@code dataSource}：数据库连接源（H2 内存库）。
     * - {@code chatModel}：用于理解问题并辅助生成 SQL。
     * - {@code contentRetriever}：执行 SQL 检索并回填上下文。
     *
     * @return 支持 SQL 检索的 Assistant
     */
    private static Assistant createAssistant() {

        // 第 1 步：准备可查询的数据源（此处为内存 H2）。
        // 关键对象名：dataSource（SQL 检索的底层数据来源）。
        DataSource dataSource = createDataSource();

        // 第 2 步：准备聊天模型（既用于自然语言理解，也用于 SQL 生成）。
        // 关键对象名：chatModel。
        ChatModel chatModel = OpenAiChatModel.builder()
                .apiKey(OPENAI_API_KEY)
                .modelName("gpt-4o-mini")
                .build();

        // 第 3 步：构建 SQL 内容检索器。
        // 它会把用户问题转成 SQL 并执行，把查询结果回填到上下文中。
        // 关键对象名：contentRetriever（SQL Retriever 实例）。
        ContentRetriever contentRetriever = SqlDatabaseContentRetriever.builder()
                .dataSource(dataSource)
                .chatModel(chatModel)
                .build();

        // 第 4 步：组装 AI Service。
        return AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .contentRetriever(contentRetriever)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();
    }

    /**
     * 创建并初始化示例数据库。
     * <p>
     * 方法职责：
     * 1. 建立 H2 DataSource；
     * 2. 执行建表脚本；
     * 3. 执行预置数据脚本。
     *
     * @return 初始化完成的 DataSource
     */
    private static DataSource createDataSource() {

        // 子步骤 10.D1：创建 H2 内存数据库连接参数。
        // 关键对象名：dataSource（实现类 JdbcDataSource）。
        JdbcDataSource dataSource = new JdbcDataSource();
        dataSource.setURL("jdbc:h2:mem:test;DB_CLOSE_DELAY=-1");
        dataSource.setUser("sa");
        dataSource.setPassword("sa");

        // 子步骤 10.D2：读取并执行建表 SQL。
        // 关键对象名：createTablesScript（DDL 脚本文本）。
        String createTablesScript = read("sql/create_tables.sql");
        execute(createTablesScript, dataSource);

        // 子步骤 10.D3：读取并执行预置数据 SQL。
        // 关键对象名：prefillTablesScript（DML 脚本文本）。
        String prefillTablesScript = read("sql/prefill_tables.sql");
        execute(prefillTablesScript, dataSource);

        return dataSource;
    }

    /**
     * 从 resources 读取文本文件内容（这里用于读取 SQL 脚本）。
     *
     * @param path 资源相对路径（例如 sql/create_tables.sql）
     * @return 文件全文字符串
     */
    private static String read(String path) {
        try {
            // read 子步骤 r1：从 resources 读取 SQL 文件内容。
            // toPath(path) 负责把 classpath 相对路径定位到真实文件路径。
            return new String(Files.readAllBytes(toPath(path)));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * 执行 SQL 脚本文本。
     * <p>
     * 关键对象：
     * - {@code Connection}：数据库连接。
     * - {@code Statement}：SQL 执行器。
     *
     * @param sql 多条 SQL 组成的脚本文本（以分号分隔）
     * @param dataSource 数据源
     */
    private static void execute(String sql, DataSource dataSource) {
        try (Connection connection = dataSource.getConnection(); Statement statement = connection.createStatement()) {
            // 为简单起见按分号拆分并逐条执行。
            // 真实生产建议使用成熟迁移工具（Flyway/Liquibase）管理 SQL 生命周期。
            // 关键对象名：connection / statement。
            // execute 子步骤 x1：将多语句脚本按分号切分。
            for (String sqlStatement : sql.split(";")) {
                // execute 子步骤 x2：逐条执行当前 SQL 语句。
                statement.execute(sqlStatement.trim());
            }
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }
}
