# %%
import math
import typing as ty
from pathlib import Path
from typing import Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor

import lib


# %%
class Tokenizer(nn.Module):
    # Tokenizer类继承自nn.Module，是PyTorch中的基础类，用于构建模型。

    # 类变量，用于存储类别特征的偏移量，如果不使用类别特征，则为None。
    category_offsets: ty.Optional[Tensor]

    def __init__(
            self,
            d_numerical: int,  # 数值特征的维度
            categories: ty.Optional[ty.List[int]],  # 各类别特征的类别数目列表
            d_token: int,  # 令牌嵌入的维度
            bias: bool,  # 是否添加偏置项
    ) -> None:
        super().__init__()  # 调用父类的构造函数
        if categories is None:
            d_bias = d_numerical  # 如果没有类别特征，偏置项的维度等于数值特征的维度
            self.category_offsets = None  # 类别偏移量为None
            self.category_embeddings = None  # 类别嵌入为None
        else:
            d_bias = d_numerical + len(categories)  # 偏置项的维度为数值特征维度加上类别特征的数量
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)  # 计算类别特征的偏移量
            self.register_buffer('category_offsets', category_offsets)  # 将偏移量注册为模型的缓冲区
            self.category_embeddings = nn.Embedding(sum(categories), d_token)  # 创建类别特征的嵌入
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))  # 使用Kaiming初始化嵌入权重
            print(f'{self.category_embeddings.weight.shape}')  # 打印嵌入权重的形状

        # 考虑到[Cross-level Readout Node]，创建一个权重参数
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        # 根据是否添加偏置来创建偏置参数
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # 使用Kaiming初始化权重
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # 如果有偏置，也使用Kaiming初始化偏置
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        # 计算总的令牌数，即权重的长度加上类别偏移量的长度（如果有的话）
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]) -> Tensor:
        # 前向传播函数，接收数值特征x_num和类别特征x_cat
        x_some = x_num if x_cat is None else x_cat  # 确保至少有一种特征是可用的
        assert x_some is not None  # 断言确保x_some不为空
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # 创建一个全1的列向量作为[CLS]令牌
            + ([] if x_num is None else [x_num]),  # 如果x_num不为空，将其添加到列表中
            dim=1,  # 沿着列的方向拼接
        )
        x = self.weight[None] * x_num[:, :, None]  # 将权重应用到数值特征上
        if x_cat is not None:
            # 如果有类别特征，将类别嵌入添加到x中
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            # 如果有偏置，将偏置添加到x中
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),  # 创建一个全0的偏置项
                    self.bias,
                ]
            )
            x = x + bias[None]  # 将偏置应用到x上
        return x


class MultiheadGEAttention(nn.Module):
    # MultiheadGEAttention类继承自nn.Module，用于实现多头图注意力机制。

    def __init__(
            # 正常注意力机制需要的参数
            self, d: int, n_heads: int, dropout: float, initialization: str,
            # FR-Graph（特征关系图）需要的参数
            n: int, sym_weight: bool = True, sym_topology: bool = False, nsi: bool = True,
    ) -> None:
        # 检查是否可以均匀分配维度到每个头上
        if n_heads > 1:
            assert d % n_heads == 0
        # 初始化方法必须是'xavier'或'kaiming'
        assert initialization in ['xavier', 'kaiming']

        super().__init__()  # 调用父类的构造函数
        # 初始化线性层用于值的变换
        self.W_v = nn.Linear(d, d)
        # 如果有多头，则需要一个输出线性层来合并各个头的输出
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads  # 头的数量
        self.dropout = nn.Dropout(dropout) if dropout else None  # dropout层，如果dropout为0，则不使用

        # FR-Graph的参数: 边的权重
        # 头部和尾部的转换
        self.W_head = nn.Linear(d, d)
        # 如果使用对称权重，则头部和尾部共享权重
        if sym_weight:
            self.W_tail = self.W_head
        else:
            self.W_tail = nn.Linear(d, d)  # 非对称权重
        # 关系嵌入：学习可变换的对角矩阵
        self.rel_emb = nn.Parameter(torch.ones(n_heads, d // self.n_heads))

        # 初始化权重和偏置
        for m in [self.W_head, self.W_tail, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

        # FR-Graph的参数: 图的拓扑结构（列 = 节点 = 特征）
        self.n_cols = n + 1  # 节点数量：输入特征节点 + [跨层读出]
        self.nsi = nsi  # 不包含自身交互

        # 列嵌入：为每个列（特征）定义语义
        d_col = math.ceil(2 * math.log2(self.n_cols))  # 列头嵌入的维度
        self.col_head = nn.Parameter(Tensor(self.n_heads, self.n_cols, d_col))
        if not sym_topology:
            self.col_tail = nn.Parameter(Tensor(self.n_heads, self.n_cols, d_col))
        else:
            self.col_tail = self.col_head  # 共享参数
        for W in [self.col_head, self.col_tail]:
            if W is not None:
                nn_init.kaiming_uniform_(W, a=math.sqrt(5))

        # 学习偏置和固定阈值用于拓扑结构
        self.bias = nn.Parameter(torch.zeros(1))
        self.threshold = 0.5

        # 冻结拓扑结构
        # 对于一些敏感数据集设置为`True`
        # 在训练几个epoch后，有助于
        # 稳定性和更好的性能
        self.frozen = False

    def _reshape(self, x: Tensor) -> Tensor:
        # 重塑张量以适应多头视图
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
        )

    def _no_self_interaction(self, x):
        # 如果只有[读出节点]，则不进行自身交互
        if x.shape[-2] == 1:
            return x
        assert x.shape[-1] == x.shape[-2] == self.n_cols
        # 掩盖对角线上的交互
        nsi_mask = 1.0 - torch.diag_embed(torch.ones(self.n_cols, device=x.device))
        return x * nsi_mask

    def _prune_to_readout(self, x):
        # 从任何特征到[读出节点]的边进行剪枝
        assert x.shape[-1] == self.n_cols
        mask = torch.ones(self.n_cols, device=x.device)
        mask[0] = 0  # 从特征到[读出]的交互置零
        return x * mask

    def _get_topology(self, top_score, elewise_func=torch.sigmoid):
        # 学习静态知识拓扑结构（邻接矩阵）
        adj_probs = elewise_func(top_score + self.bias)  # 使用sigmoid函数作为元素级激活函数
        if self.nsi:
            adj_probs = self._no_self_interaction(adj_probs)  # 应用nsi函数
        adj_probs = self._prune_to_readout(adj_probs)  # 从特征到[读出]的边进行剪枝

        if not self.frozen:
            # 使用'Straight-through'技巧进行不可微操作
            adj = (adj_probs > 0.5).float() - adj_probs.detach() + adj_probs
        else:
            # 冻结图拓扑结构：无梯度
            adj = (adj_probs > 0.5).float()
        return adj

    def forward(
            self,
            x_head: Tensor,
            x_tail: Tensor,
            key_compression: ty.Optional[nn.Linear],
            value_compression: ty.Optional[nn.Linear],
            elewise_func=torch.sigmoid,
            comp_func=torch.softmax,
    ):
        # 前向传播函数
        f_head, f_tail, f_v = self.W_head(x_head), self.W_tail(x_tail), self.W_v(x_tail)
        for tensor in [f_head, f_tail, f_v]:
            # 检查是否符合多头设置
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            # 如果有压缩层，则应用压缩
            assert value_compression is not None
            f_tail = key_compression(f_tail.transpose(1, 2)).transpose(1, 2)
            f_v = value_compression(f_v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(f_head)
        d_head_tail = f_tail.shape[-1] // self.n_heads
        d_value = f_v.shape[-1] // self.n_heads
        n_head_nodes = f_head.shape[1]

        # 重塑为多头视图
        f_head = self._reshape(f_head)
        f_tail = self._reshape(f_tail)

        # 边权重分数（Gw）
        weight_score = f_head @ torch.diag_embed(self.rel_emb) @ f_tail.transpose(-1, -2) / math.sqrt(d_head_tail)

        col_emb_head = F.normalize(self.col_head, p=2, dim=-1)  # L2标准化列嵌入
        col_emb_tail = F.normalize(self.col_tail, p=2, dim=-1)  # 使用 L2 归一化将嵌入转换为相似的规模并提高训练稳定性
        # 拓扑分数（Gt）
        top_score = col_emb_head @ col_emb_tail.transpose(-1, -2)
        # 图拓扑结构（A）
        adj = self._get_topology(top_score, elewise_func)
        if n_head_nodes == 1:  # 只有[跨层读出]
            adj = adj[:, :1]

        # 组装图：将FR-Graph应用于交互，类似于注意力掩码
        adj_mask = (1.0 - adj) * -10000  # 类似于注意力掩码
        # 此层的FR-Graph
        # 可用于可视化特征关系和读出收集
        fr_graph = comp_func(weight_score + adj_mask, dim=-1)  # 使用softmax作为竞争函数

        if self.dropout is not None:
            fr_graph = self.dropout(fr_graph)
        x = fr_graph @ self._reshape(f_v)
        x = (
            x.transpose(1, 2)
            .reshape(batch_size, n_head_nodes, self.n_heads * d_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x, fr_graph.detach()  # 返回输出和FR-Graph



class T2GFormer(nn.Module):
    """T2G-Former

    References:
    - FT-Transformer: https://github.com/Yura52/tabular-dl-revisiting-models/blob/main/bin/ft_transformer.py#L151
    """

    def __init__(
            self,
            *,
            # tokenizer 参数
            d_numerical: int,  # 数值特征的维度
            categories: ty.Optional[ty.List[int]],  # 类别特征的数量列表，每个元素对应一个特征的类别数
            token_bias: bool,  # 是否在token中使用偏置
            # transformer 参数
            n_layers: int,  # Transformer层数
            d_token: int,  # token的维度
            n_heads: int,  # 注意力头的数量
            d_ffn_factor: float,  # FFN隐藏层的倍数因子
            attention_dropout: float,  # 注意力层的dropout比率
            ffn_dropout: float,  # FFN层的dropout比率
            residual_dropout: float,  # 残差连接的dropout比率
            activation: str,  # 激活函数类型
            prenormalization: bool,  # 是否使用预归一化
            initialization: str,  # 参数初始化策略
            # linformer 参数
            kv_compression: ty.Optional[float],  # KV压缩比率
            kv_compression_sharing: ty.Optional[str],  # KV压缩共享策略
            # graph estimator 参数
            sym_weight: bool = True,  # 权重是否对称
            sym_topology: bool = False,  # 拓扑结构是否对称
            nsi: bool = True,  # 是否不包含自身交互
            #
            d_out: int,  # 输出维度
    ) -> None:
        assert (kv_compression is None) ^ (
                    kv_compression_sharing is not None)  # 确保kv_compression和kv_compression_sharing参数逻辑正确

        super().__init__()  # 调用父类构造函数
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)  # 初始化Tokenizer
        n_tokens = self.tokenizer.n_tokens  # 获取token的数量

        # 定义KV压缩函数
        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            if initialization == 'xavier':
                nn_init.xavier_uniform_(compression.weight)
            return compression

        # 根据参数决定是否使用共享的KV压缩
        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        # 定义归一化函数
        def make_normalization():
            return nn.LayerNorm(d_token)

        n_tokens = d_numerical if categories is None else d_numerical + len(categories)  # 计算token的总数
        d_hidden = int(d_token * d_ffn_factor)  # 计算FFN隐藏层的维度
        self.layers = nn.ModuleList([])  # 初始化层列表
        for layer_idx in range(n_layers):  # 对于每一层
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadGEAttention(
                        d_token, n_heads, attention_dropout, initialization,
                        n_tokens, sym_weight=sym_weight, sym_topology=sym_topology, nsi=nsi,
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)  # 将层添加到层列表

        self.activation = lib.get_activation_fn(activation)  # 获取激活函数
        self.last_activation = lib.get_nonglu_activation_fn(activation)  # 获取最后一层的激活函数（如果不是GLU）
        self.prenormalization = prenormalization  # 设置是否使用预归一化
        self.last_normalization = make_normalization() if prenormalization else None  # 设置最后一层是否使用归一化
        self.ffn_dropout = ffn_dropout  # 设置FFN的dropout比率
        self.residual_dropout = residual_dropout  # 设置残差连接的dropout比率
        self.head = nn.Linear(d_token, d_out)  # 初始化最后的线性层

    # 获取KV压缩层，如果设置了共享压缩层，则使用共享的，否则使用单独的压缩层
    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    # 开始残差连接前的准备，如果使用预归一化，则先进行归一化处理
    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    # 结束残差连接，将残差连接的输出与输入相加，如果不使用预归一化，则在此步骤后进行归一化处理
    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    # 前向传播函数
    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor], return_fr: bool = False) -> Tensor:
        fr_graphs = []  # 存储每层的FR-Graph
        x = self.tokenizer(x_num, x_cat)  # 对输入数据进行token化处理

        for layer_idx, layer in enumerate(self.layers):  # 遍历每一层
            is_last_layer = layer_idx + 1 == len(self.layers)  # 是否为最后一层
            layer = ty.cast(ty.Dict[str, nn.Module], layer)  # 类型转换，确保layer是字典类型

            x_residual = self._start_residual(x, layer, 0)  # 开始残差连接
            x_residual, fr_graph = layer['attention'](
                # 对于最后一层的注意力机制，只处理[CLS]标记
                (x_residual[:, :1] if is_last_layer else x_residual),
                x_residual,
                *self._get_kv_compressions(layer),
            )
            fr_graphs.append(fr_graph)  # 添加FR-Graph
            if is_last_layer:
                x = x[:, : x_residual.shape[1]]  # 如果是最后一层，调整x的形状以匹配残差
            x = self._end_residual(x, x_residual, layer, 0)  # 结束残差连接

            x_residual = self._start_residual(x, layer, 1)  # 开始第二个残差连接
            x_residual = layer['linear0'](x_residual)  # 应用第一个线性变换
            x_residual = self.activation(x_residual)  # 应用激活函数
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)  # 应用dropout
            x_residual = layer['linear1'](x_residual)  # 应用第二个线性变换
            x = self._end_residual(x, x_residual, layer, 1)  # 结束第二个残差连接

        assert x.shape[1] == 1  # 确保x的第二维度为1
        x = x[:, 0]  # 移除多余的维度
        if self.last_normalization is not None:
            x = self.last_normalization(x)  # 如果设置了最后的归一化，则进行归一化处理
        x = self.last_activation(x)  # 应用最后的激活函数
        x = self.head(x)  # 应用最后的线性层
        x = x.squeeze(-1)  # 移除最后的维度
        return x if not return_fr else (x, fr_graphs)  # 返回输出，如果需要则返回FR-Graphs

    # 冻结FR-Graph拓扑的API，用于训练时固定FR-Graph的拓扑结构
    def froze_topology(self):
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            layer['attention'].frozen = True  # 将attention层的frozen属性设置为True

