-- 04_churn_by_tenure_bucket.sql: Churn rate by tenure bucket (0-12, 13-24, 25-48, 49+)
SELECT
    CASE
        WHEN tenure BETWEEN 0 AND 12 THEN '0-12'
        WHEN tenure BETWEEN 13 AND 24 THEN '13-24'
        WHEN tenure BETWEEN 25 AND 48 THEN '25-48'
        ELSE '49+'
    END AS segment,
    COUNT(*) AS n,
    SUM(churn) AS churners,
    ROUND(SUM(churn) * 1.0 / COUNT(*), 4) AS churn_rate
FROM customers
GROUP BY segment
ORDER BY
    CASE segment
        WHEN '0-12' THEN 1
        WHEN '13-24' THEN 2
        WHEN '25-48' THEN 3
        WHEN '49+' THEN 4
    END;
