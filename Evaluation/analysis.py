# 常用的评估方法
#   [accuracy, TP/TN/FP/FN, precision, recall, f1, PRC, ROC, AUC, map, rank1~10]
#
# Copyright (c) 2018 @WiederSeele.
# =============================================


 gt = [0,  1,  0,  0,  1,  0,  1,  1,  1,  0]
 pdp = [0.01, 0.05, 0.87, 0.13, 0.92, 0.44, 0.88, 0.68, 0.75, 0.33]
 pd = [0,  0,  1,  0,  1,  0,  1,  1,  0,  0]
 ax = [tn, fn, fp, tn, tp, tn, tp, tp, fn, tn]
 TN = 4
 TP = 3
 FN = 2
 FP = 1
 accuracy = TP+TN / all = 7/10
 precision = TP / TP+FP = 3/4
 recall    = TP / TP+FN = 3/5
 f1 = 2*p*r / (p+r)  # 调和均值

 TPR = TP / TP+FN = 3/5
 FPR = FP / FP+TN = 1/5
对pdp设置不同的阈值,可以产生不同的pd,对应的FPR也会从0到1变化,于是得到ROC曲线
ROC曲线 <-- TPR / FPR
AUC = ROC曲线的面积  物理意义: 正样本的score大于负样本score的概率.
 