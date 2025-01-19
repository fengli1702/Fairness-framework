> Epoch = 10 , Batch_size=256 , no_fairness , group_size = 11 ,所有集全是../data/a0910/all_virtual_user_data.csv 


Average NDCG Higher is better : 0.6979437526062638
Average Kendall Tau Distance Lower is better: 24.07139991977537
Average Spearman's Rank Correlation Coefficient Higher is better: 0.16607956824563325
Average Cosine Similarity Higher is better  : 0.7617370194987525
Average Mean Reciprocal Rank Higher is better: 0.2745343040797587

这是在test集下检测accuracy的结果。
C:\programming\codefile\newgit\fairness\IRT>python test_acc.py
evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████| 218/218 [00:00<00:00, 310.61it/s]
Test AUC: 0.6651903084738731, Test Accuracy: 0.6730631276901005