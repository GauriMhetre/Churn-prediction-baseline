-- 03_churn_by_contract.sql: Churn rate and customer count by Contract type
SELECT
    Contract AS segment,
    COUNT(*) AS n,
    SUM(churn) AS churners,
    ROUND(SUM(churn) * 1.0 / COUNT(*), 4) AS churn_rate
FROM customers
GROUP BY Contract
ORDER BY churn_rate DESC;
