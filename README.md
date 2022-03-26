implement  Chinese Segmentation with a Word-Based Perceptron Algorithm

### 数据集

训练集：人民日报一月份语料

测试集：人民日报二月份语料

### 实验结果

model-9:
precision: 0.950
recall: 0.947
F1: 0.949

---------------------------------------------------------------------------------------
将人民日报一月份语料和人民日报二月份语料融合后做5-fold cross validation的相应操作在cross_valid_ver.py和test_0325.ipynb中实现（仅是简单调用了sklearn.model_selection.KFold）。

其中，slides10-Sequence Segmentation.pdf来自于张岳老师编写的《Natural Language Processing: A Machine Learning Perspective （机器学习视角下的自然语言处理）》一书的对应课程，他们的课程主页为：https://westlakenlp.github.io/nlpml/ ，他们的课程在西湖大学文本智能实验室的bilibili账号[WestlakeNLP](https://space.bilibili.com/639900532)上更新。
