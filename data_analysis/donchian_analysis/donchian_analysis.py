# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 22:29:25 2016

@author: gsantiago
"""
import glob
import json
import pandas as pd

results_dir='/git_repos/github/stock_analysis/localdata/jsonresults'
files = ["file://"+f for f in glob.glob(results_dir+"/*.json")] 
typesDict = {
                'initialDate':'timestamp',
                'finalDate':'timestamp'
            }
jsonData = [pd.read_json(f, typ='series', dtype=typesDict) for f in files ]
df = pd.DataFrame(jsonData)
df = df.sort(columns=('finalDate'))

df = df[['trainingSizeInMonths', 'windowSizeInMonths', 'riskRate', 'initialDate', 'finalDate','initialBalance', 'finalBalance']]
df.columns = ['trainingSize', 'windowSize', 'riskRate', 'initialDate', 'finalDate','initialBalance', 'finalBalance']
    
#keys = [(k1, k2, str(k3)) for (k1, k2, k3), group in df.groupby(['trainingSize', 'windowSize', 'riskRate']) if len(group[group.finalBalance>group.initialBalance])==len(group)]
keys = [(k1, k2, str(k3), sum(group.finalBalance-group.initialBalance)) for (k1, k2, k3), group in df.groupby(['trainingSize', 'windowSize', 'riskRate']) if sum(group.finalBalance-group.initialBalance)>0]
for k in keys:
    print(k)
    print(df[(df.trainingSize==k[0]) & (df.windowSize==k[1]) & (df.riskRate==float(k[2]))][['initialDate', 'finalDate', 'finalBalance']])
    
    
print(df[(df.trainingSize==1) & (df.windowSize==4) & (df.riskRate==0.03)][['initialDate', 'finalDate', 'finalBalance']])


