-- 01_profile.sql: High-level summary profile of the customers table
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT customerID) AS distinct_customers,
    SUM(CASE WHEN TotalCharges IS NULL THEN 1 ELSE 0 END) AS null_total_charges,
    SUM(churn) AS total_churners,
    ROUND(AVG(churn), 5) AS overall_churn_rate
FROM customers;
