
import pickle
import csv
import pandas as pd
from transformers import AutoTokenizer  # 还有其他与模型相关的tokenizer，如BertTokenizer
import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import json

# # 读取.pkl文件
# with open('./reveal/reveal_val.pkl', 'rb') as f:
#     data = pickle.load(f)
# # 将数据转化为DataFrame格式
# df = pd.DataFrame(data)
# print(df.columns)
# df.rename(columns={'functionSource':'text'},inplace=True)
# data=df[['text','label']]
# # 将DataFrame中的数据保存为.csv文件
# data.to_csv('./reveal/text_val.csv', index=False)


# #读csv文件
# df = pd.read_csv('./d2a/function/d2a_lbv1_function_test.csv')
# df.rename(columns={'code':'text'},inplace=True)
# data=df[['text']]
# # 将DataFrame中的数据保存为.csv文件
# data.to_csv('./d2a/function/d2a_test.csv', index=False)


# #打标签
# df = pd.read_csv('./VulCNN/vul.txt',sep='\t',header=None)
# n=len(df[0])
# list = [1 for i in range(n)]
# df.insert(1,column='label',value=list)
# df.to_csv('./VulCNN/vul.csv',index=False)

# #划分数据集
# df1=pd.read_csv('./VulCNN/vul.csv')
# df2=pd.read_csv('./VulCNN/no_vul.csv')
#
# train_data1,val_data1 = train_test_split(df1,train_size=0.8,test_size=0.2,shuffle=True)
# train_data2,val_data2 = train_test_split(df2,train_size=0.8,test_size=0.2,shuffle=True)
# test_data1,val_data1 = train_test_split(val_data1,train_size=0.5,test_size=0.5,shuffle=True)
# test_data2,val_data2 = train_test_split(val_data2,train_size=0.5,test_size=0.5,shuffle=True)
#
# train_data = pd.concat([train_data1,train_data2],axis=0)
# val_data = pd.concat([val_data1,val_data2],axis=0)
# test_data = pd.concat([test_data1,test_data2],axis=0)
#
# train_values = train_data.values
# np.random.shuffle(train_values)
# train_data = pd.DataFrame(train_values,columns=train_data.columns)
#
# val_values = val_data.values
# np.random.shuffle(val_values)
# val_data = pd.DataFrame(val_values,columns=val_data.columns)
# test_values = test_data.values
# np.random.shuffle(test_values)
# test_data = pd.DataFrame(test_values,columns=test_data.columns)
#
# print(train_data)
# print(val_data)
# print(test_data)
# train_data.to_csv('./VulCNN/vulcnn_train2.csv', index=False)
# val_data.to_csv('./VulCNN/vulcnn_val2.csv', index=False)
# test_data.to_csv('./VulCNN/vulcnn_test2.csv', index=False)


# #读取json文件
#
# df=pd.read_json('./devign/Devign.json')
# print(df)
# print(df[['func','target']][0:1])
# file=open("./devign/valid.txt",'r')
# list=[]
# for line in file:
#     list.append(int(line.split('\n')[0]))
#
# dft=pd.DataFrame(columns=['func','target'])
#
# print(dft)
# for i in list:
#     dft=pd.concat([dft,df[['func','target']][i:i+1]],ignore_index=True)
# print(dft)
# dft.to_csv('./devign/val.csv',index=False)

# # 读取.pkl文件
# with open('./reveal/reveal_val.pkl', 'rb') as f:
#     data = pickle.load(f)
# # 将数据转化为DataFrame格式
# df = pd.DataFrame(data)
# #part_data =df.iloc[100:200,:]
# print(df[['functionSource']])
# # df.to_pickle('')

# #读csv文件
# df = pd.read_csv('./ours/our_traintest.csv')
# df.rename(columns={'text':'code'},inplace=True)
# print(df)
# part_data1=df.iloc[0:40000,:]
# print(part_data1)
# part_data2=df.iloc[40000:80000,:]
# print(part_data2)
# part_data3=df.iloc[80000:120000,:]
# print(part_data3)
# part_data4=df.iloc[120000:160000,:]
# print(part_data4)
# part_data5=df.iloc[160000:,:]
# print(part_data5)
# part_data1.to_csv('./ours/our_train1.csv',index=False)
# part_data2.to_csv('./ours/our_train2.csv',index=False)
# part_data3.to_csv('./ours/our_train3.csv',index=False)
# part_data4.to_csv('./ours/our_train4.csv',index=False)
# part_data5.to_csv('./ours/our_train5.csv',index=False)


# df = pd.read_csv('./ours/our_val.csv')
# df.rename(columns={'text':'code'},inplace=True)
# code_train=df
# data = pd.DataFrame(code_train)
# print(data)
# list = []
# for i in range(len(data)):
#     s = data['code'][i]
#     t=''
#     str = s
#     # print(str)
#     for j in range(len(str)):
#         if str[j]==';'or str[j]=='{':
#             t+=str[j]
#             t+='\n'
#         else:
#             t+=str[j]
#     list.append(t)
# rows = zip(list)
# i=0
# with open('./ours/our_valtest.csv', "w") as f:
#     writer = csv.writer(f)
#     writer.writerow(["code","label"])
#     for row in rows:
#         # writer.writerows({row,data['label'][i]})
#         writer.writerow(row)
#         i+=1
# df2 = pd.read_csv('./ours/our_valtest.csv')
# for i in range(len(data)):
#     df2['label']=data['label']
# df2.to_csv('./ours/our_valtest.csv',index=False)

#code label合并
# df1 = pd.read_csv('./devign/val.csv')
# df2 = pd.read_csv('./devign/devign_val_text.csv')
# df1 = df1['label']
# df2 = df2['text']
# merged_df = pd.concat([df2, df1], axis=1)
# print(merged_df)
# merged_df.to_csv('./devign/devign_val.csv',index=False)

code_train = pd.read_csv('./d2a/function/d2a_test.csv')
data = pd.DataFrame(code_train)
novul=0
vul=0
for i in range(len(data)):
    s = data['label'][i]
    if(s==1):
        vul+=1
    else:
        novul+=1
print("vul",vul)
print("novul",novul)