import pandas as pd
t = pd.read_csv("sheets/eval_per_class_kdtree.csv")
print(t[["n", "precision_at_k", "auroc"]].corr(method="spearman"))
ps = pd.read_csv("sheets/eval_per_shape_kdtree.csv")
print(ps["chance_precision"].mean(), t["chance_precision"].mean())   # micro vs macro chance