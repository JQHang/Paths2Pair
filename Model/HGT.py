import math

import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax


class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, node_dict, edge_dict, n_heads, dropout=0.2, use_norm=False):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm
        
        # 每种类型的边对应一个qkv映射加注意力权重
        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(
            torch.ones(self.num_relations, self.n_heads)
        )
        self.relation_att = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.relation_msg = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        # 保证不会影响图里的原始数据
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            
            # 依次处理每种类型的边
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)
                
                # 获取边对应的序号
                e_id = self.edge_dict[etype]
                
                # 获取对应的关系转移参数
                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata["k"] = k
                sub_graph.dstdata["q"] = q
                sub_graph.srcdata["v_%d" % e_id] = v
                
                # q点乘k，把结果存到t上
                sub_graph.apply_edges(fn.v_dot_u("q", "k", "t"))
                
                # 计算注意力权重
                # pop是移除边的表征，sum
                attn_score = (
                    sub_graph.edata.pop("t").sum(-1)
                    * relation_pri
                    / self.sqrt_dk
                )
                attn_score = edge_softmax(sub_graph, attn_score, norm_by="dst")

                sub_graph.edata["t"] = attn_score.unsqueeze(-1)

            G.multi_update_all(
                {
                    etype: (
                        fn.u_mul_e("v_%d" % e_id, "t", "m"),
                        fn.sum("m", "t"),
                    )
                    for etype, e_id in edge_dict.items()
                },
                cross_reducer="mean",
            )

            new_h = {}
            for ntype in G.ntypes:
                """
                Step 3: Target-specific Aggregation
                x = norm( W[node_type] * gelu( Agg(x) ) + x )
                """
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data["t"].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h

# 全部节点的类型，全部节点的特征，全部的关系数据
# node_dict 节点名称到序号
# edge_dict 边的名称到序号
class HGT(nn.Module):
    def __init__(self, node_type_to_feature_len_dict, all_relation_list, n_hid, n_out, n_layers, n_heads, use_norm=True):
        super(HGT, self).__init__()
        
        # 对节点的原始特征进行映射
        self.adapt_ws = nn.ModuleDict()
        
        for tmp_node_type in node_type_to_feature_len_dict:
            self.adapt_ws.append(nn.Linear(node_type_to_feature_len_dict[tmp_node_type], n_hid))
        
        # 有多少层就有多少模块，每个模块内针对各个关系生成不同的权重
        self.n_layers = n_layers
        self.gcs = nn.ModuleList()
        
        for _ in range(n_layers):
            self.gcs.append(
                HGTLayer(
                    n_hid,
                    n_hid,
                    node_dict,
                    edge_dict,
                    n_heads,
                    use_norm=use_norm,
                )
            )
        
        # 最终的输出层
        self.out = nn.Linear(n_hid, n_out)
        
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out

    def forward(self, G, out_key):
        h = {}
        
        # 转化节点的原始特征
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](G.nodes[ntype].data["inp"]))
        
        # 进行节点的转化
        for i in range(self.n_layers):
            h = self.gcs[i](G, h)
            
        return self.out(h[out_key])