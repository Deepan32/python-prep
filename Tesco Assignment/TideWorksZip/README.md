### Receipt - Transaction Ranking Model

Problem Statement:
To rank the transaction based on the relevance to the uploaded receipt.

Approach:
Based on the initial analysis, we have only the matched transaction and feature transaction rank attribute vector. But to do list level ranking, we would require recipet-transaction ranking to generate the ranking.

So I have break down the problem into two phases

Phase 1: Generate the relevance score generation using Tree based algorithms
Target Variable: Matched (If matched_txn == featured_txn then 1 else 0)
Independent Variable: Similiarity Attribute Vector
Algorithm Tested: XGBoost, Random Forest.
Evaluation Metrics : AUC

Phase 2: Generate the relevance rank for the transaction list for a given receipt
Target Variable: Reciept Level Transaction ranking (Derived from the prob score from Phase I)
Independent Variable: Similiarity Attribute Vector
Algorithm Tested: LightGBM - lambda rank
purpose, I have adopted the same.
Evaluation Metrics : nDCG and MRR

Project Structure :

1. src - Folder is Parent Folder for the Project
2. Notebook - Folder has the Basic EDA, Phase 1 and 2 Implementation testing
3. src/components Folder has following
   1. data_ingestion.py - It handles the data ingestion. Could be extended to db extraction
   2. dt_relevance_score.py - It handles the data transformation work for the Phase I Scoring
   3. dt_ranker.py - It handles the data transformation for the Phase II
   4. model_trainer_relevance_score - It handles the model training, grid search and best model selection and prediction for relevance Scoring
   5. model_trainer_trans_ranker.py - It handles the model traning for the ranking the transaction.
4. artifacts - Folder has

   1. Model - Relevance Scorer and Ranker
   2. Preprocessor Pipeline - For both Phases

5. Pipeline - Folder has the placeholder for training and prediction pipeline
6. Utils Folder has the files related to logging, exception handling

Explanation for Choice:

1. Phase I

   - I have used Bagging and Boosting models along with Grid Search to fine tune the hyper parameter and predict the relevance score for each transaction based on the attribute vector.

   - I have used the AUC evaluation metrics since it can handle imbalance. Relevance Scoring model has AUC around 0.8 and closer the value to 1 better the peformance. Our model is delivering good performance. The performance can be improved using additional dataset and features.

2. Phase II

   - I have leveraged the LightGBM based lambdarank algo for listwise ranking. Lamddarank performs really well on the listwise ranking problems.
   - Reason for Metrics:

   1. nDCG - Provides a normalized view on the performance of the model. nDCG = 0.99 describes that in relation to original ranking order, model is able to rank more accurately.

   2. MRR - If we could show the first relevant transaction effectively, higher chance for the customer engagement and experience. MRR = 0.75 can proves the model is able to identify top most relevant transaction.

Next Steps:

1. Improve the Relevance score generator model based additional features and large dataset
2. Add LightGBM wrapper to handle the GridSearch/Hyperparameter tuning
3. Explore the deep learning based ranking algorithm - ListNet , Listwise ANN
4. Add in additional features improve the ranker model performance
