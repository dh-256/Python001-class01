import pandas as pd
import random

data = [[random.randint(0, 100), 'name'+str(i), random.randint(0, 100),'content'+str(i), random.randint(0, 100)] for i in range(0, 2000)]
df = pd.DataFrame(data, columns=['id', 'name', 'age','content', 'order_id'])

# SELECT * FROM data;
df

# SELECT * FROM data LIMIT 10;
df.head(10)

# SELECT id FROM data;  //id 是 data 表的特定一列
df['id']

# SELECT COUNT(id) FROM data;
df['id'].count()

# SELECT * FROM data WHERE id<1000 AND age>30;
df[(df['id'] < 1000) & (df['age'] > 30)]

# SELECT id,COUNT(DISTINCT order_id) FROM table1 GROUP BY id;
df[['id', 'order_id']].groupby('id').count()

# SELECT * FROM table1 t1 INNER JOIN table2 t2 ON t1.id = t2.id;
data1 = [[random.randint(0, 1000), 'name'+str(i), random.randint(0, 100),'content'+str(i), random.randint(0, 100)] for i in range(0, 100)]
df1 = pd.DataFrame(data1, columns=['id', 'name', 'age','content', 'order_id'])

data2 = [[random.randint(0, 1000), 'attr2'+str(i)] for i in range(0, 100)]
df2 = pd.DataFrame(data2, columns=['id', 'att2'])

pd.merge(df1, df2, on='id', how='inner')

# SELECT * FROM table1 UNION SELECT * FROM table2;
pd.concat([df1, df2])

# DELETE FROM table1 WHERE id=10;
df.drop(df[df['id'] == 10].index, inplace=True)

# ALTER TABLE table1 DROP COLUMN column_name;
del df['att2']
