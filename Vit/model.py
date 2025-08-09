import torch
import torch.nn as nn


# 分割层
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # 图片大小
        self.img_size = img_size
        img_size = (img_size, img_size)
        # 分割的块大小
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        # 行列要分割成多少块
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # 要分割块的总数
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # 用卷积核进行分割
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)


    def forward(self,x):
        # 展平，把高宽展平，然后调换展平后的维度和通道数的位置
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x




# 多头注意力
class Attention(nn.Module):
    def __init__(self,
                 dim,                # 输入token的维度
                 heads=8,            # 注意力头数
                 qkv_bias=False,     # 是否在QKV变换中添加偏置项
                 qk_scale=None,      # 手动指定QK点积的缩放因子
                 attn_drop_ratio=0., # 注意力权重的丢弃比率
                 proj_drop_ratio=0., # 输出投影的丢弃比率
                ):
        super().__init__()
        self.heads = heads
        # 每个头分到的维度
        head_dim = dim // heads
        # 输入dim 输出dim * 3
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # 将多个注意力头的结果进行合并，用一个可学习的线性层进行学习
        self.proj = nn.Linear(dim, dim)
        self.scare = head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj_drop = nn.Dropout(proj_drop_ratio)


    def forward(self, x):
        # N为词的数量 C为词的维度
        B, N, C = x.shape
        # 将输入映射到q、k、v三个矩阵
        qkv = self.qkv(x)
        # 拆分成多头 [B, N, 3, num_heads, head_dim]
        qkv = qkv.repeat(B,N,3,self.heads,C // self.heads)
        # 重新分配维度 [3, B, num_heads, N, head_dim]
        qkv = qkv.permute(2, 0, 3, 1,4)
        # 分割出q、k、v三个矩阵，每个q、k、v有多个头
        # q   :[B, num_heads, N, head_dim]
        # k^T :[B, num_heads, head_dim, N]
        q,k,v = qkv[0],qkv[1],qkv[2]
        # 对每个头进行注意力计算
        # [B, num_heads, N, N],进行注意力点积缩放
        attn = q @ k.transpose(-2,-1) * self.scare
        # 对k维度作softmax归一化
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # [B, num_heads, N, head_dim] -> [B, N, num_heads, head_dim] -> [B, N, C] eg: C = N * head_dim
        attn = (attn @ v).transpose(1,2).reshape(B,N,C)
        return self.proj_drop(self.proj(attn))

# transformer里面的全连接层
class MLP(nn.Module):
    def __init__(self, in_features,out_features,hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(0.4)
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# transformer块
class Transformer_Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 drop_path_ratio=0.,):
        super().__init__()
        self.num_heads = num_heads
        self.attn = Attention(dim,heads=num_heads,qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio,proj_drop_ratio=proj_drop_ratio)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.mlp = MLP(dim,dim,dim*mlp_ratio)
        self.drop = nn.Dropout(drop_ratio)
    def forward(self,x):
        x = x + self.drop(self.attn(self.norm(x)))
        x = x + self.drop(self.mlp(self.norm(x)))
        return x



class Vit(nn.Module):
    def __init__(self,img_size=224,patch_size=16,in_chans=3,embed_dim=768,num_classes=10,num_heads=12,mlp_ratio=4,
                 depth=12,qkv_bias=False,qk_scale=None,attn_drop_ratio=0.,proj_drop_ratio=0.,drop_path_ratio=0.):
        super(Vit, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size

        # patch层
        self.patched = PatchEmbed(img_size=img_size,patch_size=patch_size,in_chans=in_chans,embed_dim=embed_dim)
        # 分割的总patches
        num_patches = self.patched.num_patches
        # 归一化
        self.norm = nn.LayerNorm(embed_dim)

        # 全连接输出层
        self.head = nn.Linear(embed_dim,num_classes)

        # transformer块
        self.tran = nn.Sequential(*[
                Transformer_Block(
                dim = embed_dim,
                num_heads = num_heads,
                mlp_ratio = mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop_ratio=attn_drop_ratio,
                proj_drop_ratio=proj_drop_ratio,
                drop_path_ratio=drop_path_ratio,
            )
        for _ in range(depth)]
        )

        # CLS Token
        self.cls = nn.Parameter(torch.zeros(1,1,embed_dim))

        # 位置编码
        self.pos = nn.Parameter(torch.zeros(1,num_patches + 1,embed_dim))

        # 丢弃层
        self.drop = nn.Dropout(drop_path_ratio)


    def forward_features(self,x):
        # 获取批次维度
        B = x.shape[0]
        # 经过分割层
        x = self.patched(x)
        # 扩展cls维度
        cls_token = self.cls.expand(B,-1,-1)
        # 进行拼接
        x = torch.cat((cls_token, x),dim=1)
        # 加上位置编码
        x = x + self.pos
        # 丢弃
        x = self.drop(x)
        # 进入encoder层
        x = self.tran(x)
        # 拿出cls用于分类 [batch,num_tokens,embed_dim]
        x = self.norm(x[:,0,:])
        return x


    def forward(self,x):
        x = self.forward_features(x)
        # 进入全连接层
        x = self.head(x)
        return x