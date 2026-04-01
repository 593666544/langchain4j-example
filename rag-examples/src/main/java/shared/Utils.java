package shared;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.nio.file.PathMatcher;
import java.nio.file.Paths;
import java.util.Scanner;

import static dev.langchain4j.internal.Utils.getOrDefault;

public class Utils {

    /**
     * 从环境变量读取 OpenAI API Key（硬核逐行说明）。
     * <p>
     * 步骤 1：读取系统环境变量 OPENAI_API_KEY。
     * 子步骤 1.1：System.getenv("OPENAI_API_KEY") 可能返回 null（未配置）。
     * <p>
     * 步骤 2：使用 getOrDefault 做兜底。
     * 子步骤 2.1：如果环境变量存在，使用真实 key。
     * 子步骤 2.2：如果环境变量缺失，回退到字符串 "demo"。
     * <p>
     * 步骤 3：暴露为 public static final 常量。
     * 子步骤 3.1：所有示例可直接复用该常量，无需重复写读取逻辑。
     * 子步骤 3.2：注意 "demo" 仅用于教学占位，无法通过真实 OpenAI 鉴权。
     */
    public static final String OPENAI_API_KEY = getOrDefault(System.getenv("OPENAI_API_KEY"), "demo");

    /**
     * 启动一个最简单的命令行多轮对话循环（硬核逐行版）。
     * <p>
     * 输入：
     * - {@code assistant}：已装配完成的 AI Service 代理对象。
     * <p>
     * 处理：
     * - 循环读取用户输入；
     * - 若输入 exit 则终止；
     * - 否则调用 assistant.answer(userQuery) 得到回复；
     * - 输出日志供学习观察。
     * <p>
     * 输出：
     * - 控制台日志中的 User/Assistant 对话内容。
     * <p>
     * 关键对象名与职责：
     * - Logger log：负责输出学习日志，帮助你观察链路。
     * - Scanner scanner：阻塞读取终端输入，一次读取一行。
     * - String userQuery：当前轮用户问题。
     * - String agentAnswer：当前轮助手回答。
     *
     * @param assistant 已装配完成的 AI Service 代理对象
     */
    public static void startConversationWith(Assistant assistant) {
        // 步骤 1：创建日志器。
        // 子步骤 1.1：LoggerFactory.getLogger(Assistant.class) 使用 Assistant 类型名作为日志分类。
        Logger log = LoggerFactory.getLogger(Assistant.class);
        // 步骤 2：创建 Scanner 并进入资源托管块。
        // 子步骤 2.1：try-with-resources 保证方法结束时自动关闭 scanner。
        try (Scanner scanner = new Scanner(System.in)) {
            // 步骤 3：进入无限循环，模拟持续对话。
            while (true) {
                // 子步骤 3.1：打印分隔线，提升日志可读性。
                log.info("==================================================");
                // 子步骤 3.2：提示用户输入。
                log.info("User: ");
                // 子步骤 3.3：阻塞读取一整行用户输入。
                String userQuery = scanner.nextLine();
                log.info("==================================================");

                // 子步骤 3.4：处理退出指令（不区分大小写）。
                if ("exit".equalsIgnoreCase(userQuery)) {
                    // 子步骤 3.5：跳出循环，结束会话。
                    break;
                }

                // 子步骤 3.6：调用助手核心方法，触发模型/RAG链路。
                String agentAnswer = assistant.answer(userQuery);
                // 子步骤 3.7：打印回答日志。
                log.info("==================================================");
                log.info("Assistant: " + agentAnswer);
            }
        }
    }

    /**
     * 构造 glob 匹配器（硬核说明）。
     * <p>
     * 输入：
     * - glob 表达式，例如 "*.txt"、"递归匹配 markdown 文件的表达式"。
     * <p>
     * 处理：
     * - 调用 FileSystems.getDefault().getPathMatcher("glob:" + glob)。
     * - Java NIO 会返回一个可重复使用的路径匹配器对象。
     * <p>
     * 输出：
     * - PathMatcher，可用于文档加载器筛选文件。
     *
     * @param glob glob 表达式（例如 *.txt）
     * @return 可用于路径匹配的 PathMatcher
     */
    public static PathMatcher glob(String glob) {
        // 步骤 1：给表达式加上 "glob:" 协议前缀，让 NIO 按 glob 语法解析。
        // 步骤 2：返回 PathMatcher 给调用方复用。
        return FileSystems.getDefault().getPathMatcher("glob:" + glob);
    }

    /**
     * 将 resources 下的相对路径转换为可读的绝对 Path（硬核逐行版）。
     * <p>
     * 输入：
     * - relativePath：classpath 资源相对路径。
     * <p>
     * 处理：
     * - 先通过 ClassLoader.getResource(relativePath) 获取资源 URL；
     * - 再把 URL 转为 URI，再转为 Path；
     * - 若 URI 语法非法，抛出运行时异常中断流程。
     * <p>
     * 输出：
     * - java.nio.file.Path，可被文件加载器直接读取。
     * <p>
     * 示例：
     * - toPath("documents/miles-of-smiles-terms-of-use.txt")
     * - toPath("sql/create_tables.sql")
     *
     * @param relativePath resources 下的相对路径
     * @return 资源文件对应的 Path 对象
     */
    public static Path toPath(String relativePath) {
        try {
            // 步骤 1：从 classpath 中查找资源，得到 URL。
            URL fileUrl = Utils.class.getClassLoader().getResource(relativePath);
            // 步骤 2：URL -> URI -> Path，统一成 Java 文件 API 可用的对象。
            return Paths.get(fileUrl.toURI());
        } catch (URISyntaxException e) {
            // 步骤 3：将受检异常包装为 RuntimeException，避免每个调用方重复 try/catch。
            throw new RuntimeException(e);
        }
    }
}
