import pandas as pd

tableA = [(100, 'chocolate, sprinkles'),
         (101, 'chocolate, filled'),
         (102, 'glazed')]
labels = ['product', 'tags']
dfA = pd.DataFrame.from_records(tableA, columns=labels)

tableB = [('A', 100),
           ('A', 101),
           ('B', 101),
           ('C', 100),
           ('C', 102),
           ('B', 101),
           ('A', 100),
           ('C', 102)]
labels = ['customer', 'product']
dfB = pd.DataFrame.from_records(tableB, columns=labels)

print(dfA)

print(dfB)

result=pd.merge(dfA,dfB,how='inner', on='product')

print(result)

result1=(result.set_index(result.columns.drop('tags',1)
        .tolist()).tags.str.split(', ', expand=True).stack().reset_index()
        .rename(columns={0:'tags'}).loc[:, result.columns])
        
dfC=result1.pivot_table(values='tags',index=['customer'], columns=['tags'], 
                         aggfunc='size')
print(dfC)

dfC.fillna(0, inplace=True)

dfC=dfC.astype(int) #converting float value type to int

print(dfC)
