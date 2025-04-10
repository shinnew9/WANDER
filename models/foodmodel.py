import torch
import torch.nn as nn
from transformers import ViTModel, BertModel
import torch.nn.functional as F
from models.adapter import Adapter
from models.transformer import TransformerEncoder

class FoodModel(nn.Module):             # 왜 모델이름이 Food일까?
    def __init__(self, vis_path='ViT', text_path='BERT',
                 output_dim = 101, out_dropout=0.1, embed_dim=768, num_heads=8, layers=2):
        super(FoodModel, self).__init__()
        self.num_mod = 2
        self.out_dropout = out_dropout
        self.vision_encoder = ViTModel.from_pretrained(vis_path)
        self.text_encoder = BertModel.from_pretrained(text_path)
        self.vision_encoder.requires_grad(False)
        self.text_encoder.requires_grad_(False)
        self.fusion = TransformerEncoder(embed_dim=embed_dim, num_heads=num_heads, layers=layers)
        self.proj1 = nn.Linear(768, 768)
        self.out_layer = nn.Linear(768, output_dim)

    def get_embed(self, v, t):
        ti, ta, tt = t
        v = self.vision_encoder(v)["last_hidden_state"]
        t = self.text_encoder(ti, ta, tt)["last_hidden_state"]
        feature = torch.cat([v, t], dim=1)
        fusion = self.fusion(feature)
        v_f = fusion[:, :v.shape[1], :]
        t_f = fusion[:, :v.shape[1]:, :]
        return v_f, t_f


    def forward(self, v, t):
        ti, ta, tt = t
        v = self.vision_encoder(v)["last_hidden_state"]
        t = self.text_encoder(ti, ta, tt)["last_hidden_state"]
        feature = torch.cat([v,t], dim =1)
        fusion = self.fusion(feature)
        cls_h = fusion[:, 0, :]
        last_hs_proj= F.dropout(
            F.gelu(self.proj1(cls_h)), p=self.out_dropout, training=self.training
        )
        output = self.out_layer(last_hs_proj)
        return output
    

class FoodModelWander(nn.Module):
    def __init__(
        self, vis_path='ViT', text_path='BERT',
        output_dim = 101, t_dim =[], rank=8, drank=8, trank=9, out_dropout=0.1
    ):
        super(FoodModelWander, self).__init__()
        self.num_mond = 2
        self.out_dropout = out_dropout
        self.basemodel = FoodModel(
            vit_path = vis_path,
            bert_path = text_path,
            output_dim = output_dim,
            out_dropout = out_dropout,
        )

        self.basemodel.requires_grad_(False)
        # prediction head
        self.basemodel.proj1.requires_grad(True)
        self.basemodel.out_layer.requires_grad(True)
        self.adapter = Adapter(self.num_mod, 768, t_dim, rank, drank, trank)


    def forward(self, v, t):
        v, t = self.get_embed(v, t)
        fusion = self.adapter([v, t])
        last_hs_proj = F.dropout(
            F.relu(self.basemodel.proj1(fusion[:, 0, :])), p=self.out_dropout, training=self.training
        )
        output = self.basemodel.out_layer(last_hs_proj)
        return output