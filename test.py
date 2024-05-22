import pandas as pd
import csv
import torch
import torch.utils.data as Data
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# #提取csv文件中的函数源码
# code_train=pd.read_csv("./pytorch-image-models/vuldeepecker/code_train.csv")
# data=pd.DataFrame(code_train)
# data=data[['text']]
# data.to_csv("./pytorch-image-models/vuldeepecker/test.txt",index=False, header=None,quoting=csv.QUOTE_NONE,escapechar=' ')


# TextCNN Parameter
embedding_size = 16*16*3  #每一个token要用多少维度向量去编码
#sequence_length = len(sentences[0]) # every sentences contains sequence_length(=3) words
sentences_length = 40
num_classes = 2  # num_classes=2  二分类
batch_size = 4   #每次取4个训练集为一组
sentences_num = 512

code_train=pd.read_csv('./pytorch-image-models/vuldeepecker/code_train.csv')
text_train=pd.read_csv('./pytorch-image-models/vuldeepecker/code_train_text.csv')
data=pd.DataFrame(code_train)
text_data=pd.DataFrame(text_train)
print(data)


#处理数据
t_code=np.array(text_data[['text']]).tolist()
n=len(t_code)
print(n)
s=[[1]for i in range (n)]
for i in range(n):
    s[i]= ''.join(t_code[i]).split('\n')
print(s[0])
print(s[1])
print(s[128117])

#处理标签 ，转为一维Tensor类型
label=torch.tensor(np.array(data[['label']])).view(-1)
# label1=np.array(data[['label']]).tolist()
# label=[[1]for i in range (n)]
# for i in range(n):
#     label[i]=label1[i]
# print(label)

#hash词表
word_list=[]
for i in range(n):
    word_list += " ".join(s[i]).split()
vocab = list(set(word_list))  #所有词去重
word2idx = {w: i+1 for i, w in enumerate(vocab)}
# print(word2idx)
vocab_size = len(vocab)+1
print(vocab_size)

#数据预处理
def make_data(sentences, labels):
  inputs = []
  for sen in sentences:
      inputs.append([word2idx[n] for n in sen.split()])

  targets = []
  for out in labels:
      targets.append(out) # To using Torch Softmax Loss function
  return inputs, targets


#向量化表示
# code_list=[[1]for i in range (n)]
# result = torch.LongTensor().to(device)
result = []
for i in range(2):
    sentences=s[i]
    labels = np.zeros(len(sentences), dtype=int)
    input_batch, target_batch = make_data(sentences, labels)
    # print(input_batch[0])
    for j in range(len(input_batch)):
        if len(input_batch[j]) < sentences_length:
            input_batch[j].extend(0 for j in range(sentences_length - len(input_batch[j])))
        else:
            input_batch[j]=input_batch[j][:sentences_length]
            # print('-------------------------')
            # print(len(input_batch[j]))
    if(len(input_batch)<sentences_num):
        input_batch.extend([0 for i in range(sentences_length)] for j in range(sentences_num-len(input_batch)))
    else:
        input_batch=input_batch[:sentences_num]
    # print(input_batch)
    # print(target_batch)
    # print(len(input_batch))
    # print(len(target_batch))
    # input_batch, target_batch = torch.LongTensor(input_batch), torch.LongTensor(target_batch)
    # code_list[i]=input_batch
    # input_batch = torch.v.unsqueeze(0).to(device)
    # result = torch.cat((result,input_batch),0)
    # print(type(input_batch))
    # input_batch = [[sublist] for sublist in input_batch]

    # input_batch = np.array(input_batch)

    result.append(input_batch)
    print('----------------------------------'+'\n')
    print(i)
result = torch.LongTensor(result).to(device)
print(result)
# input_batch=code_list
target_batch=[0 for i in range (2)]
# target_batch=label

input_batch=result
# print(input_batch)
# print(len(input_batch))
target_batch=torch.LongTensor(target_batch)
torch_dataset = Data.TensorDataset(input_batch, target_batch)
# print(torch_dataset[1])

loader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=torch_dataset,
    batch_size=4,
    shuffle=True,
    # num_workers=2,
)

for bidx, batch in enumerate(loader):
    sentence = batch[0].to(device)  # sentence 为Tensor类型
    print(sentence)