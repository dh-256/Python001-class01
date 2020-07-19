学习笔记

学习数据集
```python
from sklearn import datasets # import datasets
from sklearn import train_test_split # split training data and test data at certain ratio
```

常用库
- pandas
- numpy
- matplotlib

获取当前文件的目录
```python
os.path.realpath(__file__)
```

### pandas
#### 筛选列
```python
df['column_name']
```

#### 筛选行
```python
df[1:3]
```

#### 增加列名
```python
df.columns = ['column1', 'column2', 'column3']
```

#### 选择特定的行+列
```python
df.loc[1:3, ['column2']]
```

#### 过滤数据
```python
df['column2'] == 'value1' # series of test results
df[df['column2'] == 'value1'] # select rows of which conditions are true
```

#### 清除缺失数据
```python
df.dropna()
```

#### 数据聚合
```python
df.groupby('column2').sum()
```

#### 创建新列
```python
map = {
	'val1': 1,
	'val2': 2
}
df['new_column'] = df['column2'].map(map)
```

#### 基本数据类型
- Series
- DataFrame

Series
基于 numpy 的概念
一维数据结构
被 numpy 以*列*的形式展现

基本属性
- index
- value

```python
pd.Series(['a', 'b', 'c']) # 基于列表创建 series
pd.Series({'a': 1, 'b':2, 'c':3}) # 创建基于字典的 series, key 会作为 index
pd.Series([11, 22, 33], index=['a', 'b', 'c']) # 基于列表，并给定 index

s1.index # 获取全部索引
s1.values # 获取全部的值 -> numpy.ndarray
s1.values.tolist() # 转换成 python list

s1.map(lambda) # 创建 filter，Series of True/False with index
s1[filter] # 使用 filter
```

DataFrame
二维数据结构

```python
pd.DataFrame(['a', 'b', 'c']) # 创建一列的 df
pd.DataFrame([[], [], []]) # 基于二维数组创建 df
df.columns = ['column1', 'column2'] # 指定列索引
df.index = ['row1', 'row2'] # 指定行索引
```

导入数据
```python
pd.read_excel(r'1.xlsx', sheet_name=0) # import from excel
pd.read_csv(r'a.csv', sep=' ', nrows=10, encoding='utf-8') # import from csv
pd.read_table('text.txt', sep='')
pd.read_sql(sql, conn)  # import from db, needs pymysql
```

熟悉数据
```python
df.head(n) # 显示前 n 行
df.shape # 显示行列数
df.info() # 显示详细信息，按行整理
df.describe() # 一些基本的统计数据，平均，最大最小，分布等
```

数据预处理
1. 缺失值处理
	1. 反复爬取补全
	2. 用户手动输入
2. 数据的重复值处理

```python
# Series
s.hasnans # 判断 df 中是否有缺失值
s.fillna(value = df.mean()) # 以平均值填充 nan

# DataFrame
df.isnull().sum() # 查看缺失值汇总
df.ffill() # 用上一行填充
df.ffill(axis=1) # 用前一列填充
df.info() # 查看每一列有多少非空数值
df.dropna() # 删除带空值的行
df.fillna('默认值') # 填充默认值 
```

数据调整
```python
# 行列调整
df[['column1', 'column3']] # 选择列
df.iloc[:, [0, 2] # 选择所有行，第0和第2列
df.loc[[0,2]] # 选择第0行和第2行
df.loc[0:2] # 选择第0行到第2行
df[(condition1) & (condition2)] # 比较
df[(df['A'] < 5) & (df['C'] > 4)] # 比较例子

# 数值替换
df['C'].replace(4, 40) # 把列中的 4 替换成 40
df.replace(np.NaN, 0) # 把列中的 nan 替换成 0
df.replace([1,2,3], 100) # 多对一替换
df.replace({1: 100, 2:200}) # 多对多替换

# 排序
df.sort_values(by=['column1', ascending=True]) # 按照指定列升序排序
df.sort_values(by=['column1', 'column2', ascending=[True, False]]) # 多列排序

# 删除
df.drop('A', axis = 1) # 删除列
df.drop(3, axis = 0) # 删除行
df[df['A'] < 4] # 通过条件过滤（删除）行
df.T # 行列互换
df.T.T # 行列互换

# 数据透视
df.stack() # 表格按照行优先堆叠展开
df.unstack() # 表格按照列优先堆叠展开
df.stack().reset_index() # 堆叠展开以后添加索引，填充level变成 df
```

数据计算
```python
# 算数运算
df['A'] + df['C'] # 两列的四则运算
df['A'] + 5 # 列的常数四则运算
df['A'] < df['C'] # 列的比较运算
df.count() # 按列进行非空值计数
df.sum() # 按列进行 sum
df['A'].sum() # 指定列 sum

## window functions
# mean 均值
# max 最大值
# min 最小值
# median 求中位数
# mode 众数
# var 方差
# std 标准差
```

数据分组聚合
```python
# 聚合
df.groupby('column2').groups # 按 column 把 df 分成几个小 df
df.groupby('column2').count() # 查看每个 group 的大小
df.groupby('column2').aggregate({'column2':'count', 'column3': 'sum'}) # 按 group 显示聚合后的统计数据
# df.groupby().agg() 同理
df.groupby('group').agg('mean') # 对分组按列均值
df.groupby('group').mean() # 同上，但直接调用
...to_dict() # 转成 python 数据类型
df.groupby('group').transform('mean') # 分组计算的结果应用回原来的行，而不是按组显示结果
```

数据透视表
```python
pd.pivot_table(
	df,
	values='column2',
	columns='column1',
	index='column3',
	aggfunc='count',
	margins=True
).reset_index()
```

多表拼接
```python
pd.merge(df1, df2) # 一对一，自动寻找公共列
pd.merge(df1, df2 on='column2') # 在多个公共列情况下指定公共列 merge
pd.merge(df1, df2) # 多对多自行寻找公共列
pd.merge(df1, df2, left_on='column1', right_on='column3') # 没有公共列的情况下指定公共列 merge
pd.merge(df1, df2, on='column1', how='inner') # 指定连接方式，默认内连接, 还可以是 left, right, outer
pd.concat([df1, df2]) # 纵向连接
```

#### 输出
```python
df.to_excel(excel_writer='file.xlsx', sheet_name='sheet1', index=False) # write to excel file
df.to_csv() # write to csv file
df.to_pickle() # write to pickle (performance optimized)
```

性能
- pickle is better than csv or excel
- agg(sum) is better than agg(lambda)

#### matplotlib
```python
dates = pd.date_range('20200101', period=12) # generate dates
```

```python
from matplotlib.pyplot as plt
```

```python
plt.plot(df.index, df['A'], )
plt.show()
plt.plot(
	df.index, # x
	df['A'], # y
	color='#FFAA00', # stoke color
	linestyle='--',	# line style
	linewidth=3, # line width
	marker='D' # 点标记 (Dot)
)
```

Seaborn 是在 matplotlib 上的封装

```python
import seaborn as sns
```

```python
plt.scatter(df.index, df['A']) # 散点图
sns.set_style('darkgrid') # 将 plot 变成带灰度的表格
```

### jieba 分词与关键词提取
```python
import jieba
result = jieba.cut(string, cut_all=False) # 精确模式，默认
result = jieba.cut(string, cut_all=True) # 全模式，会列出不同的排列组合，方便搜索引擎
```

Viterbi 启发式搜索

关键词提取
两种算法
- TF-IDF
- TextRank

```python
import jieba.analyse
# TF-IDF
tfidf = jieba.analyse.extract_tags(text, topK=5, withWeight=True) # 返回权重最大的 K 个词，返回每个关键字的权重值
# TextRank
text_rank = jieba.analyse.textrank(text, topK=5, withWeight=False)
```

stop_words
不考虑的词
```python
jieba.analyse.set_stop_words('stop_words.txt')
```

自定义词典
```python
jieba.load_userdict('userR_dict.txt')
jieba.del_word('word') #  动态删除词，一般用于去掉错分的词
jieba.suggest_freq('word', True) # 调整分词，合并
jieba.suggest_freq(('中', '将'), True) # 调整分词，拆分
```

### SnowNLP
自然语言分析工具
语义的情感分析
```python
from snownlp import SnowNLP
s = SnowNLP(text)
s.words # 中文分词
list(s.tags) # 词性标注(隐马尔可夫模型)
s.sentiments # 情感分析(朴素贝叶斯分类器，1是正向，0是负向)
from snownlp import seq
seg.train('data.txt')
seg.save('seg.marshal')
# 修改 snownlp/seg/__init__.py 的 data_path 指向新的训练的模型
```

```python
s.pinyin # 拼音 (Trie 树)
s.han # 繁体转换简体
s.keywords(limit=5) # 提取关键词
```

```python
# 情感分析和 pandas 结合
df = pd.read_csv()
df.shorts.apply(_sentiment) # 先定义 SnowNLP 分析
df.sentiment.mean() # 获取平均值
```