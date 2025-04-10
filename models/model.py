import torch
from torch import nn
import torch.nn.functional as F

from models.transformer import TransformerEncoder
from models.adapter import Adapter


class Latefusion(nn.Module):
    def __init__(
        self,
        orig_dim,
        output_dim=1,
        proj_dim=40,
        num_heads=5,
        layers=5,
        relu_dropout=0.1,
        embed_dropout=0.15,
        res_dropout=0.1,
        out_dropout=0.1,
        attn_dropout=0.2,
    ):
        super(Latefusion, self).__init__()

        self.proj_dim = proj_dim
        self.orig_dim = orig_dim
        self.num_mod = len(orig_dim)
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout

        # Projection Layers
        self.proj = nn.ModuleList(
            [
                nn.Conv1d(self.orig_dim[i], self.proj_dim, kernel_size=1, padding=0)
                for i in range(self.num_mod)
            ]
        )

        # Encoders
        self.encoders = nn.ModuleList(
            [
                TransformerEncoder(
                    embed_dim = proj_dim,
                    num_heads = self.num_heads,
                    layers = self.layers,
                    attn_dropout = self.attn_dropout,
                    res_dropout = self.res_dropout,
                    relu_dropout = self.relu_dropout,
                    embed_dropout = self.embed_dropout,
                )
                for _ in range(self.num_mod)
            ]
        )

        # Output layers
        self.out_layer_proj0 = nn.Linear(3*self.proj_dim, self.proj_dim)  # 여기서 3을 곱하는 이유는 layer가 3개라서?
        self.out_layer_proj1 = nn.Linear(self.proj_dim, self.proj_dim)
        self.out_layer_proj2 = nn.Linear(self.proj_dim, self.proj_dim)
        self.out_layer = nn.Linear(self.proj_dim, output_dim)

    def get_emb(self, x):
        hs = list()

        for i in range(self.num_mod):
            x[i] = x[i].transpose(1, 2)  # transpose와
            x[i] = self.proj[i](x[i])
            x[i] = x[i].permute(2, 0, 1)  # permute
            h_tmp = self.encoders[i](x[i])
            hs.append(h_tmp[0])

        last_hs_out = torch.cat(hs, dim=-1)

        last_hs = F.relu(self.out_layer_proj0(last_hs_out))
        last_hs_proj = self.out_layer_proj2(
            F.dropout(
                F.relu(self.out_layer_proj1(last_hs)),
                p =self.out_dropout,
                training=self.training,
            )
        )
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
    
        return output

class AdapterModel(nn.Module):
    def __init__(
        self,
        orig_dim, 
        t_dim,
        rank,
        drank,
        trank,
        output_dim =1,
        proj_dim=40,
        num_heads=5,
        layers=5,
        relu_dropout=0.1,
        embed_dropout=0.15,
        res_dropout=0.1,
        out_dropout=0.1,
        attn_dropout=0.2
    ):
        
        super().__init__()
        self.num_mod = len(orig_dim)
        self.basemodel = Latefusion(
            orig_dim,
            output_dim,
            proj_dim,
            num_heads,
            layers,
            relu_dropout,
            embed_dropout,
            res_dropout,
            out_dropout,
            attn_dropout,
        )

        self.adapter = Adapter(self.num_mod, proj_dim, t_dim, rank, drank, trank)
        self.basemodel.requires_grad_(False)
        # prediction head
        self.basemodel.out_layer_proj0.requires_grad_(True)
        self.basemodel.out_layer_proj1.requires_grad_(True)
        self.basemodel.out_layer_proj2.requires_grad_(True)
        self.basemodel.out_layer.requires_grad_(True)
    

    def forward(self, x):
        hs = self.basemodel.get_emb(x)
        fusion = self.adapter(hs)
        fusion = fusion.permute(0, 2, 1)
        fusion = torch.cat([fusion[:, :, 0] for _ in range(self.num_mod)], dim=-1)
        output = self.basemodel.get_res(fusion)
        return output