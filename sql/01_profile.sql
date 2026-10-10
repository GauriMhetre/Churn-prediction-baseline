-- 01_profile.sql: High-level summary profile of the customers table
-- Includes null count for every column, duplicate check, and overall churn rate
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT customerID) AS distinct_customers,
    SUM(CASE WHEN customerID IS NULL THEN 1 ELSE 0 END) AS null_customerID,
    SUM(CASE WHEN gender IS NULL THEN 1 ELSE 0 END) AS null_gender,
    SUM(CASE WHEN SeniorCitizen IS NULL THEN 1 ELSE 0 END) AS null_SeniorCitizen,
    SUM(CASE WHEN Partner IS NULL THEN 1 ELSE 0 END) AS null_Partner,
    SUM(CASE WHEN Dependents IS NULL THEN 1 ELSE 0 END) AS null_Dependents,
    SUM(CASE WHEN tenure IS NULL THEN 1 ELSE 0 END) AS null_tenure,
    SUM(CASE WHEN PhoneService IS NULL THEN 1 ELSE 0 END) AS null_PhoneService,
    SUM(CASE WHEN MultipleLines IS NULL THEN 1 ELSE 0 END) AS null_MultipleLines,
    SUM(CASE WHEN InternetService IS NULL THEN 1 ELSE 0 END) AS null_InternetService,
    SUM(CASE WHEN OnlineSecurity IS NULL THEN 1 ELSE 0 END) AS null_OnlineSecurity,
    SUM(CASE WHEN OnlineBackup IS NULL THEN 1 ELSE 0 END) AS null_OnlineBackup,
    SUM(CASE WHEN DeviceProtection IS NULL THEN 1 ELSE 0 END) AS null_DeviceProtection,
    SUM(CASE WHEN TechSupport IS NULL THEN 1 ELSE 0 END) AS null_TechSupport,
    SUM(CASE WHEN StreamingTV IS NULL THEN 1 ELSE 0 END) AS null_StreamingTV,
    SUM(CASE WHEN StreamingMovies IS NULL THEN 1 ELSE 0 END) AS null_StreamingMovies,
    SUM(CASE WHEN Contract IS NULL THEN 1 ELSE 0 END) AS null_Contract,
    SUM(CASE WHEN PaperlessBilling IS NULL THEN 1 ELSE 0 END) AS null_PaperlessBilling,
    SUM(CASE WHEN PaymentMethod IS NULL THEN 1 ELSE 0 END) AS null_PaymentMethod,
    SUM(CASE WHEN MonthlyCharges IS NULL THEN 1 ELSE 0 END) AS null_MonthlyCharges,
    SUM(CASE WHEN TotalCharges IS NULL THEN 1 ELSE 0 END) AS null_TotalCharges,
    SUM(CASE WHEN churn IS NULL THEN 1 ELSE 0 END) AS null_churn,
    SUM(churn) AS total_churners,
    ROUND(AVG(churn), 5) AS overall_churn_rate
FROM customers;
