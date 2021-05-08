import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from vit_pytorch import ViT


class ImgEncoder(nn.Module):

    def __init__(self, embed_size):
        """(1) Load the pretrained model as you want.
               cf) one needs to check structure of model using 'print(model)'
                   to remove last fc layer from the model.
           (2) Replace final fc layer (score values from the ImageNet)
               with new fc layer (image feature).
           (3) Normalize feature vector.
        """
        super(ImgEncoder, self).__init__()
        model = models.vgg19(pretrained=True)
        # input size of feature vector
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1])    # remove last fc layer

        # loaded model without last fc layer
        self.model = model
        # feature vector of image
        self.fc = nn.Linear(in_features, embed_size)

    def forward(self, image):
        """Extract feature vector from image vector.
        """
        with torch.no_grad():
            # [batch_size, vgg16(19)_fc=4096]
            img_feature = self.model(image)
        # [batch_size, embed_size]
        img_feature = self.fc(img_feature)

        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        # l2-normalized feature vector
        img_feature = img_feature.div(l2_norm)

        return img_feature


class ImgViTEncoder(nn.Module):
    def init(self, patch_size, image_size=224, num_classes=1024, dim=6, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1):
        self.vit = ViT(image_size=image_size,
                       patch_size=patch_size,
                       num_classes=num_classes,
                       dim=dim,
                       depth=depth,
                       heads=heads,
                       mlp_dim=mlp_dim,
                       dropout=dropout,
                       emb_dropout=emb_dropout)

    def forward(self, img):
        return self.vit(img)


class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        # 2 for hidden and cell states
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)

    def forward(self, question):

        # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.word2vec(question)
        qst_vec = self.tanh(qst_vec)
        # [max_qst_length=30, batch_size, word_embed_size=300]
        qst_vec = qst_vec.transpose(0, 1)
        # [num_layers=2, batch_size, hidden_size=512]
        _, (hidden, cell) = self.lstm(qst_vec)
        # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = torch.cat((hidden, cell), 2)
        # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)
        # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)
        qst_feature = self.tanh(qst_feature)
        # [batch_size, embed_size]
        qst_feature = self.fc(qst_feature)

        return qst_feature


class TransformerAttention(nn.Module):
    """
    Used in Transformer Block
    """

    def __init__(self, embedding_size, n_hid, n_head, n_k, n_v):
        super().__init__()
        self.n_head = n_head
        self.n_k = n_k
        self.n_v = n_v
        self.qkv = nn.Linear(embedding_size, n_head * (n_k * 2 + n_v))
        self.out = nn.Linear(n_head * n_v, embedding_size)

    def forward(self, x, mask=None):
        n_batch, n_batch_max_in, n_hid = x.shape
        q_k_v = self.qkv(x).view(n_batch, n_batch_max_in,
                                 self.n_head, 2 * self.n_k + self.n_v).transpose(1, 2)
        q, k, v = q_k_v.split([self.n_k, self.n_k, self.n_v], dim=-1)

        q = q.reshape(n_batch * self.n_head, n_batch_max_in, self.n_k)
        k = k.reshape_as(q).transpose(1, 2)
        qk = q.bmm(k) / np.sqrt(self.n_k)

        if mask is not None:
            qk = qk.view(n_batch, self.n_head, n_batch_max_in,
                         n_batch_max_in).transpose(1, 2)
            qk[~mask] = -np.inf
            qk = qk.transpose(1, 2).view(
                n_batch * self.n_head, n_batch_max_in, n_batch_max_in)
        qk = qk.softmax(dim=-1)
        v = v.reshape(n_batch * self.n_head, n_batch_max_in, self.n_v)
        qkv = qk.bmm(v).view(n_batch, self.n_head, n_batch_max_in, self.n_v).transpose(
            1, 2).reshape(n_batch, n_batch_max_in, self.n_head * self.n_v)
        out = self.out(qkv)
        return x + out


class TransformerBlock(nn.Module):
    """
    Custom Transformer
    """

    def __init__(self, embedding_size, hidden_size, num_head=8, n_k=64, n_v=64):
        super().__init__()
        self.attn = TransformerAttention(
            embedding_size, hidden_size, num_head, n_k, n_v)
        n_inner = hidden_size * 4
        self.inner = nn.Sequential(
            nn.Linear(embedding_size, n_inner),
            nn.ReLU(inplace=True),
            nn.Linear(n_inner, embedding_size)
        )

    def forward(self, x, mask=None):
        x = x + self.attn(x, mask=mask)
        return x + self.inner(x)


class QstTransformerEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size, max_qst_length=30):

        super(QstTransformerEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.transformer_layers = nn.ModuleList(TransformerBlock(
            word_embed_size, hidden_size) for _ in range(num_layers))
        # 2 for hidden and cell states
        self.fc = nn.Linear(max_qst_length*word_embed_size, embed_size)
        self.apply(self.init_weight)

    def init_weight(self, m):
        if type(m) in [nn.Embedding]:
            nn.init.normal_(m.weight, 0, 0.1)

    def forward(self, question):

        # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.word2vec(question)
        for layer in self.transformer_layers:
            qst_vec = layer(qst_vec)
        # [batch_size, max_qst_length*word_embed_size]
        qst_feature = qst_vec.reshape(qst_vec.size()[0], -1)
        # [batch_size, embed_size]
        qst_feature = self.fc(qst_feature)

        return qst_feature


class VqaModel(nn.Module):

    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size, use_transformer, use_vit, patch_size):

        super(VqaModel, self).__init__()
        if use_vit:
          print(patch_size)
          print("Using ViT")
          self.img_encoder = ImgViTEncoder(patch_size)
        else:
          self.img_encoder = ImgEncoder(embed_size)
        if use_transformer:
            print("Using transformer")
            self.qst_encoder = QstTransformerEncoder(
                qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        else:
            self.qst_encoder = QstEncoder(
                qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

    def forward(self, img, qst):

        # [batch_size, embed_size]
        img_feature = self.img_encoder(img)
        # [batch_size, embed_size]
        qst_feature = self.qst_encoder(qst)
        # [batch_size, embed_size]
        combined_feature = torch.mul(img_feature, qst_feature)
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        # [batch_size, ans_vocab_size=1000]
        combined_feature = self.fc1(combined_feature)
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        # [batch_size, ans_vocab_size=1000]
        combined_feature = self.fc2(combined_feature)

        return combined_feature
