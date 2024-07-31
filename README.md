# Flask API for predicting Fraudulent Transactions

This Flask API is designed to predict whether a transaction is fraudulent based on various parameters like location, time, date, and more. It uses an XGBoost Classifier for the predictions.

-   **Route:** `/predict` (POST)
-   **Input:** JSON object with transaction details.
-   **Output:** JSON response with the prediction (1 for fraud, 0 for not fraud).
