-- 02_duplicates.sql: Check for duplicate customerID entries
SELECT
    customerID,
    COUNT(*) AS occurrences
FROM customers
GROUP BY customerID
HAVING COUNT(*) > 1
ORDER BY occurrences DESC;
