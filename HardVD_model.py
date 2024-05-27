import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Normalize
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReprogrammingFuntion(nn.Module):
    def __init__(self, vocab_size, img_patch_size = 16, img_size = 384, 
        img_path=None, alpha=0.2, 
        img_mean=(0.5, 0.5, 0.5), img_std=(0.5, 0.5, 0.5)):
        super(ReprogrammingFuntion, self).__init__()

        assert img_size % img_patch_size == 0
        
        self.img_patch_size = img_patch_size
        self.img_size = img_size
        self.token_embedding = nn.Embedding(vocab_size, img_patch_size * img_patch_size * 3)#嵌入vec  tokens 768表示
        self.num_patches_row = int(img_size/img_patch_size)
        self.num_patches = self.num_patches_row * self.num_patches_row
        self.base_image = None
        if img_path is not None:
            image = Image.open(img_path)
            transform=transforms.Compose([#进行组合操作
                                  Resize((img_size, img_size)),
                                  ToTensor(),#转向量
                                Normalize(img_mean,img_std),#归一化处理
                                ])

            image = transform(image) 
            self.base_image = torch.tensor(image, requires_grad=False).to(device)
            self.alpha = alpha

        self.image_mean_tensor = torch.tensor(img_mean)[None,:,None,None].to(device)
        self.image_std_tensor = torch.tensor(img_std)[None,:,None,None].to(device)

        self.W = nn.Embedding(vocab_size, 300)  # vocab_size hash表大小 ，embedding_size每一个token多少位向量表示
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 256, (k, 300)) for k in (2,3,4)])
        self.dropout = nn.Dropout(0.5)

        output_channel = 3

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def textCnn(self,X):
        '''
       X: [4,512,40]
       '''
        batch_size = X.shape[0]
        num_channels=X.shape[1]
        sequence_length=X.shape[2]
        embedding_dim=300

        embedding_X = self.W(X)  # [batch_size,num_channels, sequence_length, embedding_size]
        embedding_X = embedding_X.view(-1,  sequence_length , embedding_dim)  # batch_size*512,40,300
        embedding_X = embedding_X.unsqueeze(1) #2048 ,1,40,300
        embedding_X = torch.cat([self.conv_and_pool(embedding_X, conv) for conv in self.convs], 1)  # 2048 768
        embedding_X = self.dropout(embedding_X)
        embedding_X = embedding_X.view(batch_size, 512, 768)
        # embedding_X = embedding_X.unsqueeze(
        #     1)  # add channel(=1) [batch_size, channel(=1), sequence_length, embedding_size] 在第二维增加一个维度
        # conved = self.conv(embedding_X)  # [batch_size, output_channel*1*1]
        # flatten = conved.view(batch_size, -1)
        # embedding_X = embedding_X.view(-1,num_channels, sequence_length * embedding_dim)#4,512,40*768
        # embedding_X = self.conv(embedding_X)
        # embedding_X = self.pool(embedding_X) #4,512,808
        # embedding_X = embedding_X.view(batch_size,num_channels,embedding_dim)
        # print(embedding_X.view())
        # flatten = embedding_X.view(batch_size, num_channels, embedding_dim)
        return embedding_X

    def unnormalize_image(self, x):
        return x * self.image_std_tensor + self.image_mean_tensor

    def normalize_image(self, x):
        return (x - self.image_mean_tensor) / self.image_std_tensor
        
    def forward(self, sentence_batch):
        # sentence_embedding = torch.tanh(self.token_embedding(sentence_batch)) # (N, l, 16*16*3)

        # batch_embeddings=torch.FloatTensor()
        # batch_embeddings=batch_embeddings.to(device)
        # for batch_index in sentence_batch:
        #     batch_index=batch_index.to(device)
        #     embeddings=self.textCnn(batch_index)
        #     embeddings=embeddings.unsqueeze(dim=0)
        #     batch_embeddings=torch.cat((batch_embeddings,embeddings),0)
        # sentence_embedding=batch_embeddings # (N, l, 16*16*3)
        sentence_embedding=self.textCnn(sentence_batch)#------
        # print(sentence_embedding.size())
        _N, _L, _ = sentence_embedding.size()
        sentence_embedding = sentence_embedding.view(_N, _L, 3, self.img_patch_size, self.img_patch_size)

        reprogrammed_image = torch.zeros(_N, 3, self.img_size, self.img_size).to(device)
        
        for patch_idx in range(self.num_patches):
            i_start = int(patch_idx / self.num_patches_row) * self.img_patch_size
            j_start = (patch_idx % self.num_patches_row) * self.img_patch_size
            i_end = i_start + self.img_patch_size
            j_end = j_start + self.img_patch_size
            if patch_idx < _L:
                reprogrammed_image[:,:,i_start:i_end,j_start:j_end] = sentence_embedding[:,patch_idx]
            else:
                # adding the padding embedding all the way till the end
                reprogrammed_image[:,:,i_start:i_end,j_start:j_end] = sentence_embedding[:,_L-1]

        # normalizing by batch size
        pert_norm = torch.norm(reprogrammed_image, p=2)/_N
        
        if self.base_image is not None:
            base_image_batch = self.base_image[None].repeat((_N, 1, 1, 1))
            reprogrammed_image = base_image_batch + self.alpha * reprogrammed_image
        
        unnormalized_image = self.unnormalize_image(reprogrammed_image)
        unnormalized_image_clipped = torch.clamp(unnormalized_image, 0.0, 1.0)
        reprogrammed_image_clipped = self.normalize_image(unnormalized_image_clipped)
        
        return reprogrammed_image_clipped, pert_norm
