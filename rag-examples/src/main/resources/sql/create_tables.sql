-- SQL Retriever 示例用到的最小业务库结构。
-- 该结构刻意保持简单，便于观察“自然语言 -> SQL -> 结果回填”的完整链路。

-- 客户表：保存下单客户基本信息。
CREATE TABLE customers
(
    customer_id INT PRIMARY KEY,
    first_name  VARCHAR(50),
    last_name   VARCHAR(50),
    email       VARCHAR(100)
);

-- 商品表：保存可售商品及价格。
CREATE TABLE products
(
    product_id   INT PRIMARY KEY,
    product_name VARCHAR(100),
    price        DECIMAL(10, 2)
);

-- 订单表：记录客户购买了哪个商品、数量和下单日期。
-- 通过外键与 customers/products 关联，便于做聚合查询（如“最畅销商品”）。
CREATE TABLE orders
(
    order_id    INT PRIMARY KEY,
    customer_id INT,
    product_id  INT,
    quantity    INT,
    order_date  DATE,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
    FOREIGN KEY (product_id) REFERENCES products (product_id)
);
