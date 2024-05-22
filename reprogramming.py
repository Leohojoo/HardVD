
import time
import argparse
import torch.nn as nn
import torch
import timm
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
import data_utils
import json
import numpy as np
import pprint
import reprogramming_model
import pandas as pd
import torch.utils.data as Data
import tqdm
import random
import sklearn
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

train_hps = {  #设置默认参数值
    'num_epochs' : 50,  #学习次数
    'max_iterations' : 30000000, # overridden by args 样本数epochs*batch
    'lr' : 0.0001, # overridden by args 0.0001
    'batch_size' : 2,          # vuldeepecker 2:64059
    'validate_every' : 105929, # validates on small subset of val set       ours  2:105929  vulcnn 2:15551 vulcnn2 2:15986 devign 2:10927
    'evaluate_every' : 105929, # evaluates on full test set using best ckpt #d2a 4:1161 2:2322   reveal 4:4547 2:9094
    'label_reduction' : 'max' # overridden by args
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def unnormalize_image(tensor, mean, std):
    """
    tensor: Normalized image of shape (nc, h, w)
    """
    mean = torch.tensor(mean)[:,None,None].to(device)
    std = torch.tensor(std)[:,None,None].to(device)
    return tensor * std + mean

def normalize_image(tensor, mean, std):
    """
    tensor: Unnormalized image of shape (nc, h, w)
    """
    mean = torch.tensor(mean)[:,None,None].to(device)
    std = torch.tensor(std)[:,None,None].to(device)
    return (tensor - mean) / std

def get_mapped_logits(logits, class_mapping, multi_label_remapper):
    """
    logits : Tensor of shape (batch_size, 1000) # imagenet class logits
    class_mapping: class_mapping[i] = list of image net labels for text class i
    reduction : max or mean
    """
    if multi_label_remapper is None:
        #print("Here in old remapper")
        reduction = train_hps['label_reduction']
        mapped_logits = []
        for class_no in range(len(class_mapping)):
            if reduction == "max":
                class_logits, _ = torch.max(logits[:,class_mapping[class_no]], dim = 1) # batch size
            elif reduction == "mean":
                class_logits = torch.mean(logits[:,class_mapping[class_no]], dim = 1) # batch size
            else:
                raise NotImplentedException()

            mapped_logits.append(class_logits)
        return torch.stack(mapped_logits, dim = 1)
    else:
        orig_prob_scores = nn.Softmax(dim=-1)(logits)
        mapped_logits = multi_label_remapper(orig_prob_scores)
        return mapped_logits

def create_label_mapping(n_classes, m_per_class, image_net_labels = None):
    """
    n_classes: No. of classes in text dataset
    m_per_class: Number of imagenet labels to be mapped to each text class
    """
    if image_net_labels is None:
        image_net_labels = range(1000)

    class_mapping = [[] for i in range(n_classes)]

    idx = 0
    for _m in range(m_per_class):
        for _class_no in range(n_classes):
            class_mapping[_class_no].append(image_net_labels[idx])
            idx += 1
    return class_mapping

def get_imagenet_label_list(vision_model, base_image, img_size):
    if base_image is None:
        torch.manual_seed(42)
        base_image = 2 * torch.rand(3, img_size, img_size).to(device) - 1.0

    logits = vision_model(base_image[None])[0]
    label_sort = torch.argsort(logits)
    label_list = label_sort.detach().cpu().numpy().tolist()

    return label_list


def evaluate(dataloader, vision_model, reprogrammer, tb_writer, 
    iter_no, class_mapping, multi_label_remapper, image_net_labels, reduced_labels = None, 
    max_batches = None, 
    img_mean=(0.5, 0.5, 0.5), img_std=(0.5, 0.5, 0.5)):
    start_time = time.time()
    total_correct = 0.0
    total_examples = 0.0
    reprogrammer.eval()
    l_inf_norms = []
    all_pre = []
    all_labels=[]
    for bidx, batch in enumerate(dataloader):
        # sentence = batch['input_ids'].to(device)
        # labels = batch['label'].to(device)
        sentence = batch[0].to(device)#----
        labels = batch[1].to(device)#------
        all_labels+=labels.tolist()
        programmed_img, pert_norm = reprogrammer(sentence)
        if bidx == 0:
            vis_image = unnormalize_image(programmed_img[0], img_mean, img_std)
            tb_writer.add_image("ProgrammedImage", vis_image, iter_no)
        if reprogrammer.base_image is not None:
            _N = sentence.size(0)
            base_image_batch = reprogrammer.base_image[None].repeat((_N, 1, 1, 1))
            pert_normalized = programmed_img - base_image_batch
            pert_unnormalized = unnormalize_image(programmed_img, img_mean, img_std) - unnormalize_image(base_image_batch, img_mean, img_std)
            #pert_unnormalized = unnormalize_image(pert_normalized, img_mean, img_std)
            _l_inf_norm = torch.max(torch.abs(pert_unnormalized)).item()
            l_inf_norms.append(_l_inf_norm)

        logits = vision_model(programmed_img)
        if reduced_labels is not None:
            logits = logits[:,image_net_labels[:reduced_labels]]
        mapped_logits = get_mapped_logits(logits, class_mapping, multi_label_remapper)
        prediction = torch.argmax(mapped_logits, dim = 1)
        #print(prediction)
        all_pre+=prediction.tolist()
        correct = torch.sum(prediction == labels)
        total_correct += correct.item()
        total_examples += int(sentence.size(0))
        #print("Evaluating", "{} out of {}".format(bidx, len(dataloader)))
        if (max_batches is not None) and bidx > max_batches:
            break
    acc = total_correct/total_examples
    f1 = f1_score(y_true=all_labels,y_pred=all_pre)
    rec = recall_score(y_true=all_labels, y_pred=all_pre)
    prec = precision_score(y_true=all_labels, y_pred=all_pre)
    mean_linf_norm = None
    if len(l_inf_norms) > 0:
        mean_linf_norm = float(np.mean(l_inf_norms))
    print("mean_linf_norm", mean_linf_norm)
    print('F1-test'+str(f1))
    print('Recall召回率:' + str(rec))
    print('Precision: ' + str(prec))
    reprogrammer.train()
    end_time = time.time()
    execution_time = end_time - start_time

    print("val_time:", execution_time, "seconds")
    return {
        'acc' : acc,
        'f1' : f1,
        'rec': rec,
        'prec' : prec,
        'mean_linf_norm' : mean_linf_norm
    }


def save_checkpoint(model, learning_rate, acc, iteration, image_net_labels, class_mapping, filepath):
    print("Saving model state at iteration {} to {}".format(iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'acc' : acc,
                'image_net_labels' : image_net_labels,
                'class_mapping' : class_mapping,
                'learning_rate': learning_rate}, filepath)

def load_checkpoint(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    iteration = checkpoint_dict['iteration']

    image_net_labels = None
    if 'image_net_labels' in checkpoint_dict:
        image_net_labels = checkpoint_dict['image_net_labels']

    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, iteration, image_net_labels

def main():
    p = argparse.ArgumentParser(#解析命令行参数
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--text_dataset', type=str,default='ld')
    p.add_argument('--logdir', type=str, default = "./data2/paarth/ReprogrammingTransformers/ReprogrammingModels")
    p.add_argument('--cache_dir', type=str, default = "./data2/paarth/HuggingFaceDatasets")
    p.add_argument('--img_patch_size', type=int, default = 16)
    p.add_argument('--img_size', type=int, default = 384)
    p.add_argument('--vision_model', type=str, default = 'vit_base_patch16_384')#resnet50#vit_base_patch16_384
    p.add_argument('--base_image_path', type=str, default = None)
    p.add_argument('--label_reduction', type=str, default = 'max')
    p.add_argument('--pert_alpha', type=float, default = 0.2)
    p.add_argument('--lr', type=float, default = 0.0001)#0.0001
    p.add_argument('--resume_training', type=int, default = 0)
    p.add_argument('--resume_model_ckpt_path', type=str, default = None)
    p.add_argument('--m_per_class', type=int, default = 10)
    p.add_argument('--pretrained_vm', type=int, default = 1)
    p.add_argument('--max_validation_batches', type=int, default = 100)
    p.add_argument('--max_iterations', type=int, default = 30000000)#100000
    p.add_argument('--exp_name_extension', type=str, default = "")
    p.add_argument('--reg_alpha', type=float, default = 1e-4)
    p.add_argument('--n_training', type=int, default = None)
    p.add_argument('--use_char_tokenizer', type=int, default = 0)
    p.add_argument('--reduced_labels', type=int, default = None)
    p.add_argument('--batch_size', type=int, default = 2)
    p.add_argument('--use_sign_method', type=int, default = 0)
    args = p.parse_args()

    # train_hps['lr'] = args.lr
    # train_hps['batch_size'] = args.batch_size
    # train_hps['label_reduction'] = args.label_reduction
    # train_hps['max_iterations'] = args.max_iterations
    #
    # dataset_configs = data_utils.text_dataset_configs#导入数据集配置参数
    # assert args.text_dataset in dataset_configs
    # text_dataset_config = dataset_configs[args.text_dataset]#判断输入数据集信息是否存在，并提取

    #依次输入数据集配置信息
    # subset = text_dataset_config['subset']
    # val_split = text_dataset_config['val_split']
    # text_key = text_dataset_config['sentence_mapping']
    # data_files = text_dataset_config['data_files']
    # dataset_name = args.text_dataset if data_files is None else 'json'

    # train_split = "train"
    # if args.n_training is not None:
    #     train_split = "train[0:{}]".format(args.n_training)#train[0:{args.n_training}]


    sentences_length = 40
    sentences_num = 512

    start_time = time.time()
    # #d2a
    # code_train = pd.read_csv('./pytorch-image-models/d2a/function/d2a_train.csv')
    # text_train = pd.read_csv('./pytorch-image-models/d2a/function/d2a_train_text.csv')
    # code_val = pd.read_csv('./pytorch-image-models/d2a/function/d2a_dev.csv')
    # text_val = pd.read_csv('./pytorch-image-models/d2a/function/d2a_dev_text.csv')

    # #reveal
    # code_train = pd.read_csv('./pytorch-image-models/reveal/text_train.csv')
    # text_train = pd.read_csv('./pytorch-image-models/reveal/code_train_text.csv')
    # code_val = pd.read_csv('./pytorch-image-models/reveal/text_val.csv')
    # text_val = pd.read_csv('./pytorch-image-models/reveal/code_val_text.csv')

    # #ours
    code_train = pd.read_csv('./data/ours/our_train.csv')
    text_train = pd.read_csv('./data/ours/code_train_text.csv')
    code_val = pd.read_csv('./data/ours/our_val.csv')
    text_val = pd.read_csv('./data/ours/code_val_text.csv')

    # #vulcnn
    # code_train = pd.read_csv('./pytorch-image-models/VulCNN/vulcnn_train.csv')
    # text_train = pd.read_csv('./pytorch-image-models/VulCNN/vulcnn_train_text.csv')
    # code_val = pd.read_csv('./pytorch-image-models/VulCNN/vulcnn_val.csv')
    # text_val = pd.read_csv('./pytorch-image-models/VulCNN/vulcnn_val_text.csv')

    #devign
    #code_train = pd.read_csv('./pytorch-image-models/devign/train.csv')
    #text_train = pd.read_csv('./pytorch-image-models/devign/devign_train_text.csv')
    #code_val = pd.read_csv('./pytorch-image-models/devign/val.csv')
    #text_val = pd.read_csv('./pytorch-image-models/devign/devign_val_text.csv')

    # # vuldeepecker
    # code_train = pd.read_csv('./pytorch-image-models/vuldeepecker/code_train.csv')
    # text_train = pd.read_csv('./pytorch-image-models/vuldeepecker/code_train_text.csv')
    # code_val = pd.read_csv('./pytorch-image-models/vuldeepecker/code_val.csv')
    # text_val = pd.read_csv('./pytorch-image-models/vuldeepecker/code_val_text.csv')


    data = pd.DataFrame(code_train)
    text_data = pd.DataFrame(text_train)
    val_data = pd.DataFrame(code_val)
    val_text_data = pd.DataFrame(text_val)
    print(data)

    # 处理数据
    t_code = np.array(text_data[['text']]).tolist()
    val_code = np.array(val_text_data[['text']]).tolist()
    n = len(t_code)
    val_n = len(val_code)
    s = [[1] for i in range(n)]
    s_val = [[1] for i in range(val_n)]
    for i in range(n):
        s[i] = ''.join(t_code[i]).split('\n')
    for i in range(val_n):
        s_val[i] = ''.join(val_code[i]).split('\n')

    # 处理标签 ，转为一维Tensor类型
    label = torch.tensor(np.array(data[['label']])).view(-1)
    val_label = torch.tensor(np.array(val_data[['label']])).view(-1)


    # #随机抽取数据
    # t_code = np.array(text_data[['text']]).tolist()
    # val_code = np.array(val_text_data[['text']]).tolist()
    # n=10
    # m=len(t_code)
    # val_n = len(val_code)
    # rd=random.sample(range(m),n)
    # print('-------------------------------------------')
    # print(rd)
    # print('-------------------------------------------')
    # s = [[1] for i in range(n)]
    # s_val = [[1] for i in range(val_n)]
    # for i in range(n):
    #     s[i]=''.join(t_code[rd[i]]).split('\n')
    # for i in range(val_n):
    #     s_val[i] = ''.join(val_code[i]).split('\n')
    #
    # #随机抽取数据对应标签
    # label = torch.tensor(np.take(np.array(data[['label']]),rd)).view(-1)
    # val_label = torch.tensor(np.array(val_data[['label']])).view(-1)
    # print(len(label))
    # print(label)



    # hash词表
    word_list = []
    for i in range(n):
        word_list += " ".join(s[i]).split()
    for i in range(val_n):
        word_list += " ".join(s_val[i]).split()
    vocab = list(set(word_list))  # 所有词去重
    word2idx = {w: i + 1 for i, w in enumerate(vocab)}
    # print(word2idx)
    vocab_size = len(vocab) + 1
    print(vocab_size)

    # 数据预处理
    def make_data(sentences):
        inputs = []
        for sen in sentences:
            inputs.append([word2idx[n] for n in sen.split()])
        return inputs

    # 向量化表示
    # result = torch.LongTensor().to(device)
    # val_result = torch.LongTensor().to(device)

    result = []
    val_result = []
    for i in range(n):
        sentences = s[i]
        # labels = np.zeros(len(sentences), dtype=int)
        input_batch = make_data(sentences)
        for j in range(len(input_batch)):
            if len(input_batch[j]) < sentences_length:
                input_batch[j].extend(0 for j in range(sentences_length - len(input_batch[j])))
            else:
                input_batch[j] = input_batch[j][:sentences_length]
        if (len(input_batch) < sentences_num):
            input_batch.extend([0 for i in range(sentences_length)] for j in range(sentences_num - len(input_batch)))
        else:
            input_batch = input_batch[:sentences_num]
        # input_batch = torch.LongTensor(input_batch).unsqueeze(0).to(device)
        # result = torch.cat((result, input_batch), 0)
        result.append(input_batch)
        if(i%1000==0):
            print(str(i)+' train\n')
        # print(i)
    result = torch.LongTensor(result)

    for i in range(val_n):
        sentences = s_val[i]
        # labels = np.zeros(len(sentences), dtype=int)
        input_batch = make_data(sentences)
        for j in range(len(input_batch)):
            if len(input_batch[j]) < sentences_length:
                input_batch[j].extend(0 for j in range(sentences_length - len(input_batch[j])))
            else:
                input_batch[j] = input_batch[j][:sentences_length]
        if (len(input_batch) < sentences_num):
            input_batch.extend([0 for i in range(sentences_length)] for j in range(sentences_num - len(input_batch)))
        else:
            input_batch = input_batch[:sentences_num]
        # input_batch = torch.LongTensor(input_batch).unsqueeze(0).to(device)
        # val_result = torch.cat((val_result, input_batch), 0)
        val_result.append(input_batch)
        if(i%1000==0):
            print(str(i)+ ' val\n')
        # print(i)
    val_result = torch.LongTensor(val_result).to(device)

    target_batch=label
    input_batch = result
    target_batch = torch.LongTensor(target_batch)
    torch_dataset = Data.TensorDataset(input_batch, target_batch)

    #设置数据集
    train_loader = Data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=torch_dataset,
        batch_size=train_hps['batch_size'],
        shuffle=True,
        # num_workers=2,
    )
    target_batch = val_label
    input_batch = val_result
    target_batch = torch.LongTensor(target_batch)
    torch_dataset = Data.TensorDataset(input_batch, target_batch)

    # 设置数据集
    val_loader = Data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=torch_dataset,
        batch_size=train_hps['batch_size'],
        shuffle=True,
        # num_workers=2,
    )

    #创建网络模型
    print("Pretrained VM", args.pretrained_vm==1)
    vision_model = timm.create_model(args.vision_model, pretrained=args.pretrained_vm==1)
    for parameter in vision_model.parameters():
        # https://discuss.pytorch.org/t/best-practice-for-freezing-layers/58156
        parameter.requires_grad = False # to avoid gradient accumulation.
    vision_model.eval()
    vision_model.to(device)
    print("Vision model Frozen!")
    
    # vocab_size = len(tokenizer.get_vocab())#获取字典长度


    #导入模型的均值与方差
    img_mean = data_utils.image_model_configs[args.vision_model]['mean']
    img_std = data_utils.image_model_configs[args.vision_model]['std']

    #设置分类，数据集标签数
    n_classes = 2
    
    #对抗重编程
    reprogrammer = reprogramming_model.ReprogrammingFuntion(vocab_size, args.img_patch_size, 
        args.img_size, 
        img_path = args.base_image_path, alpha = args.pert_alpha, img_mean = img_mean, img_std = img_std)
    reprogrammer.to(device)#指定gpu或cpu
    
    image_net_labels = get_imagenet_label_list(vision_model, reprogrammer.base_image, args.img_size)
    print("Imagenet Label Ordering..", image_net_labels[:20])
    
    loss_criterion = nn.CrossEntropyLoss()

    base_image_name = None
    if args.base_image_path is not None:
        base_image_name = args.base_image_path.split("/")[-1].split(".")[0]
    exp_name = "ds_{}_lr_{}_bimg_{}_vm_{}_alpha_{}_m_label_{}_{}".format(
        args.text_dataset, train_hps['lr'], base_image_name, 
        args.vision_model, args.pert_alpha, args.m_per_class, args.label_reduction
    )
    exp_name = "{}_{}".format(exp_name, args.exp_name_extension)
    logdir = os.path.join(args.logdir, exp_name)
    ckptdir = os.path.join(logdir, "CKPTS")
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)

    tb_writer = SummaryWriter(logdir = logdir)

    iter_no = 0
    best_acc = 0.0
    best_f1 = 0.0
    best_rec = 0.0
    best_prec = 0.0
    best_iter_no = 0
    best_model_path = None
    prev_best_eval_iter = None

    if args.resume_training == 1:
        if args.resume_model_ckpt_path is None:
            resume_model_path = os.path.join(ckptdir, "model.p")
        else:
            resume_model_path = args.resume_model_ckpt_path
        print("Resuming from ckpt", resume_model_path)
        if not os.path.exists(resume_model_path):
            raise Exception("model not found")
        reprogrammer, iter_no, image_net_labels = load_checkpoint(resume_model_path, reprogrammer)

    class_mapping = None
    multi_label_remapper = None
    num_imagenet_labels = 1000
    if args.reduced_labels is not None:
        num_imagenet_labels = args.reduced_labels

    if n_classes < num_imagenet_labels:
        class_mapping = create_label_mapping(n_classes, args.m_per_class, image_net_labels)
        print("Class Mapping")
        pprint.pprint(class_mapping)
        optimizer = optim.Adam(reprogrammer.parameters(), lr=train_hps['lr'])
    else:
        print("Using Multi Label Remapper!")
        multi_label_remapper = reprogramming_model.MultiLabelRemapper(num_imagenet_labels, n_classes)
        multi_label_remapper.to(device)
        params = list(reprogrammer.parameters()) + list(multi_label_remapper.parameters())
        optimizer = optim.Adam(params, lr=train_hps['lr'])
    # end_time = time.time()
    # execution_time = end_time - start_time
    #
    # print("time:", execution_time, "seconds")
    #迭代学习次数
    for epoch in range(train_hps['num_epochs']):
        for bidx, batch in enumerate(tqdm.tqdm(train_loader,"epoch "+str(epoch),leave=True)):
            # sentence = batch['input_ids'].to(device)#sentence 为Tensor类型
            sentence = batch[0].to(device)#------
            # print(sentence)
            # print(type(sentence))
            # labels = batch['label'].to(device)
            labels = batch[1].to(device)#------
            programmed_img, pert_norm = reprogrammer(sentence)
            logits = vision_model(programmed_img)
            if args.reduced_labels is not None:
                logits = logits[:,image_net_labels[:args.reduced_labels]]
            mapped_logits = get_mapped_logits(logits, class_mapping, multi_label_remapper)
            loss = loss_criterion(mapped_logits, labels)
            reg_loss = pert_norm
            loss_total = loss + args.reg_alpha * reg_loss
            optimizer.zero_grad()
            loss_total.backward()
            if args.use_sign_method == 1:
                # print("Using sign method")
                for param in reprogrammer.parameters():
                    param.grad = torch.sign(param.grad)
                # print("gradients updated")
            optimizer.step()
            if iter_no % 10000 == 0:
                # print (iter_no, "Loss:", loss.item())
                tb_writer.add_scalar('train_loss', loss, iter_no)
                tb_writer.add_scalar('train_loss_reg', reg_loss, iter_no)
                tb_writer.add_scalar('train_loss_total', loss_total, iter_no)

            if iter_no % train_hps['validate_every'] == 0:
                print("Evaluating")
                metrics = evaluate(val_loader, vision_model,
                    reprogrammer, tb_writer, 
                    iter_no, class_mapping, multi_label_remapper, image_net_labels,
                    args.reduced_labels, 
                    max_batches = args.max_validation_batches, img_mean = img_mean, img_std = img_std)
                tb_writer.add_scalar('val_acc', metrics['acc'], iter_no)
                print(metrics)
                model_path = os.path.join(ckptdir, "model.p")
                save_checkpoint(reprogrammer, train_hps['lr'], metrics['acc'], iter_no, image_net_labels, class_mapping, model_path)
                if metrics['acc'] >= best_acc:
                    best_model_path = os.path.join(ckptdir, "model_best.p")
                    save_checkpoint(reprogrammer, train_hps['lr'], metrics['acc'], iter_no, image_net_labels, class_mapping, best_model_path)
                    best_acc = metrics['acc']
                    best_f1 = metrics['f1']
                    best_rec = metrics['rec']
                    best_prec = metrics['prec']
                    best_iter_no = iter_no
                print("Best acc|f1|rec|prec . till now:", best_acc,best_f1,best_rec,best_prec, best_iter_no)


            if (iter_no + 1) % train_hps['evaluate_every'] == 0 and prev_best_eval_iter != best_iter_no:
                # Run evaluation on whole test set using the new best checkpoint (if found)
                print("Running full evaluation!")
                backup_ckpt_path = os.path.join(ckptdir, "model_temp.p")
                save_checkpoint(reprogrammer, train_hps['lr'], metrics['acc'], iter_no, image_net_labels, class_mapping, backup_ckpt_path)

                reprogrammer, _, _ = load_checkpoint(best_model_path, reprogrammer)
                metrics = evaluate(val_loader, vision_model,
                    reprogrammer, tb_writer, 
                    iter_no, class_mapping, multi_label_remapper, image_net_labels, args.reduced_labels, 
                    img_mean = img_mean, img_std = img_std)
                log_fn = os.path.join(logdir, "best_metrics.json")
                metrics['iter_no'] = best_iter_no
                metrics['base_image_path'] = args.base_image_path
                metrics['alpha'] = args.pert_alpha
                metrics['train_hps'] = train_hps
                with open(log_fn, "w") as f:
                    f.write(json.dumps(metrics))
                prev_best_eval_iter = best_iter_no
                reprogrammer, _, _ = load_checkpoint(backup_ckpt_path, reprogrammer)
                print("Ran full evaluation!")

            iter_no += 1
            # if iter_no > train_hps['max_iterations']:
            #     break

if __name__ == '__main__':
    main()

