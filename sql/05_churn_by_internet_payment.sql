-- 05_churn_by_internet_payment.sql: Churn rate by InternetService × PaymentMethod (n >= 30)
SELECT
    InternetService,
    PaymentMethod,
    COUNT(*) AS n,
    SUM(churn) AS churners,
    ROUND(SUM(churn) * 1.0 / COUNT(*), 4) AS churn_rate
FROM customers
GROUP BY InternetService, PaymentMethod
HAVING COUNT(*) >= 30
ORDER BY churn_rate DESC, n DESC;
