-- 06_rank_segments.sql: Rank segments by churn rate using window function (Contract × tenure bucket)
WITH segments AS (
    SELECT
        Contract,
        CASE
            WHEN tenure BETWEEN 0 AND 12 THEN '0-12'
            WHEN tenure BETWEEN 13 AND 24 THEN '13-24'
            WHEN tenure BETWEEN 25 AND 48 THEN '25-48'
            ELSE '49+'
        END AS tenure_bucket,
        COUNT(*) AS n,
        SUM(churn) AS churners,
        ROUND(SUM(churn) * 1.0 / COUNT(*), 4) AS churn_rate
    FROM customers
    GROUP BY Contract, tenure_bucket
)
SELECT
    Contract,
    tenure_bucket,
    segment,
    n,
    churners,
    churn_rate,
    RANK() OVER (ORDER BY churn_rate DESC) AS rank
FROM (
    SELECT
        Contract,
        tenure_bucket,
        Contract || ' / ' || tenure_bucket AS segment,
        n,
        churners,
        churn_rate
    FROM segments
)
ORDER BY rank;
