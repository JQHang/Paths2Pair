import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import dgl.function as fn
from dgl.nn.functional import edge_softmax


    
    
class SimpleHGNLayer(nn.Module):
    """
    implementation of Simple-HGN layer
    source code comes from:  
        https://github.com/THUDM/CogDL/blob/master/examples/simple_hgn/conv.py#L72
    or
        https://github.com/THUDM/HGB/blob/master/NC/benchmark/methods/baseline/conv.py
    """

    def __init__(
        self,
        edge_feats_len,
        in_features_len,
        out_features_len,
        nhead,
        edge_dict,
        feat_drop=0.5, 
        attn_drop=0.5,
        negative_slope=0.2, # 0.05
        node_residual=False,
        edge_residual_alpha=0.05,
        activation=None,
        
    ):
        super(SimpleHGNLayer, self).__init__()
        self.edge_feats_len = edge_feats_len
        self.in_features_len = in_features_len
        self.out_features_len = out_features_len
        self.nhead = nhead
        self.edge_emb = nn.Parameter(torch.zeros(size=(len(edge_dict), edge_feats_len)))  # nn.Embedding(num_etypes, edge_feats)

        self.W = nn.Parameter(torch.FloatTensor(in_features_len, out_features_len * nhead))
        self.W_e = nn.Parameter(torch.FloatTensor(edge_feats_len, edge_feats_len * nhead))

        self.a_l = nn.Parameter(torch.zeros(size=(1, nhead, out_features_len)))
        self.a_r = nn.Parameter(torch.zeros(size=(1, nhead, out_features_len)))
        self.a_e = nn.Parameter(torch.zeros(size=(1, nhead, edge_feats_len)))

        self.feat_drop = nn.Dropout(feat_drop)
        self.dropout = nn.Dropout(attn_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        
        self.act = activation
        #self.act = None if activation is None else get_activation(activation)

        if node_residual:
            self.node_residual = nn.Linear(in_features_len, out_features_len * nhead)
        else:
            self.register_buffer("node_residual", None)
        self.reset_parameters()
        self.edge_residual_alpha = edge_residual_alpha  # edge residual weight

    def reset_parameters(self):
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

        reset(self.a_l)
        reset(self.a_r)
        reset(self.a_e)
        reset(self.W)
        reset(self.W_e)
        reset(self.edge_emb)
    
    def forward(self, all_type_edge_src_node_feature_adj_dict, res_attn):
        

        head_node_feature = all_type_edge_src_node_feature_adj_dict['head_feature']
        all_tail_node_feature = all_type_edge_src_node_feature_adj_dict['tail_feature']  # [n1+n2+.., d]
        all_tail_node_feature = torch.cat(all_tail_node_feature, 0)
        edge_list = all_type_edge_src_node_feature_adj_dict['Adj']  # [2, n1+n2+..]
        edge_list = torch.cat(edge_list, 1)
        tmp_edge = all_type_edge_src_node_feature_adj_dict['tmp_edge_index']
        tmp_edge = torch.cat(tmp_edge)
        all_tail_node_num = all_tail_node_feature.shape[0]
        
        # d:in_features_len  D:out_features_len, de:out_features_len
        x_head = self.feat_drop(head_node_feature)  # [N, d]  
        x_tail = self.feat_drop(all_tail_node_feature)
        #  [N, d]*[d, D*head] -> [N, head, D]
        h_head = torch.matmul(x_head, self.W).view(-1, self.nhead, self.out_features_len) 
        h_tail = torch.matmul(x_tail, self.W).view(-1, self.nhead, self.out_features_len) 
        # [edge_num, de]*[de, de*head] -> [edge_num, head, de]
        e = torch.matmul(self.edge_emb, self.W_e).view(-1, self.nhead, self.edge_feats_len) 
        
        head_ind, tail_ind = edge_list
        # Self-attention on the nodes - Shared attention mechanism
        # [1, head, D]*[N, head, D] -> [N, head] -> [sub_n, head]
        h_l = (self.a_l * h_head).sum(dim=-1)[head_ind]
        h_r = (self.a_r * h_tail).sum(dim=-1)[tail_ind]
        h_e = (self.a_e * e).sum(dim=-1)[tmp_edge]
        edge_attention = self.leakyrelu(h_l + h_r + h_e) # [sub_n, head]
        # Cannot use dropout on sparse tensor , put dropout operation before sparse
        edge_attention = self.dropout(edge_attention)

        # get aggregatin result by sparse matrix
        out = []
        edge_attention_weight = []
        for n in range(self.nhead):
            # [sub_n] -> [N_head, N_tail]
            edge_attention_n = torch.sparse.FloatTensor(edge_list, edge_attention[..., n], (x_head.shape[0], x_tail.shape[0]))
            edge_attention_n = torch.sparse.softmax(edge_attention_n, dim=1)
            # edge residual
            if res_attn is not None: 
                edge_attention_n = edge_attention_n * (1 - self.edge_residual_alpha) + res_attn[n] * self.edge_residual_alpha

            # [N_head, N_tail]*[N_tail, d] -> [N_head, d]
            out.append(torch.sparse.mm(edge_attention_n, h_tail[:,n,:]))
            edge_attention_weight.append(edge_attention_n)
        
        out = torch.stack(out, 1) #  [N_head, head, D]
        out = out.view(out.shape[0], -1) #  [N_head, head*D]
        
        # node residual
        if self.node_residual is not None:
            # [N, d]*[d, D*head] -> [N,D*head]
            res = self.node_residual(x_head)
            out += res
        # use activation or not
        if self.act is not None:
            out = self.act(out)
            

        return out, edge_attention_weight

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features_len) + " -> " + str(self.out_features_len) + ")"

    
    
class SimpleHGN(nn.Module):
    def __init__(self, node_dict, edge_dict, node_type_to_feature_len_dict, Model_Config_dict, Target_Node_Type):
        super().__init__()
        
        self.Target_Node_Type = Target_Node_Type
        if type(self.Target_Node_Type) != list:
            self.Target_Node_Type = [self.Target_Node_Type]
        self.all_relation_list = list(edge_dict.keys())
#         self.all_relation_list.append('self_loop')
        self.layer_num = Model_Config_dict['layer_num']
        self.head_num = Model_Config_dict['nhead']
        self.args_cuda = Model_Config_dict['args_cuda']
        self.class_num = Model_Config_dict['class_num']
        node_feature_hid_len = Model_Config_dict['node_feature_hid_len']
        GAT_hid_len = Model_Config_dict['GAT_hid_len']
        dropout = Model_Config_dict['dropout']
        
        
        self.h_dropout = nn.Dropout(dropout)
        
        # 特征转化
        self.Node_Transform_list = {}
        for tmp_node_type in node_dict:
            tmp_linear = nn.Linear(node_type_to_feature_len_dict[tmp_node_type], node_feature_hid_len)
            self.Node_Transform_list[tmp_node_type] = tmp_linear
            self.add_module('{}_Node_Transform'.format(tmp_node_type), self.Node_Transform_list[tmp_node_type])
        
        # 有多少种类型的元路径，每种元路径有多少条，就生成多少个GAT
        self.edge_GAT =  nn.ModuleList()
        # input projection
        ## TODO: 第一层加node_redidual
        self.edge_GAT.append(SimpleHGNLayer(edge_feats_len = Model_Config_dict['edge_feats_len'], 
                                       in_features_len = node_feature_hid_len,
                                       out_features_len = GAT_hid_len,
                                       nhead = self.head_num,
                                       edge_dict = self.all_relation_list,
                                       node_residual=False,
                                       edge_residual_alpha=0.05,
                                       activation=nn.functional.elu           
                                      ))
        # middle projection
        self.edge_GAT.extend([SimpleHGNLayer(edge_feats_len = Model_Config_dict['edge_feats_len'], 
                                       in_features_len = GAT_hid_len * self.head_num,
                                       out_features_len = GAT_hid_len,
                                       nhead = Model_Config_dict['nhead'],
                                       edge_dict = self.all_relation_list,
                                       node_residual=True,
                                       edge_residual_alpha=0.05,
                                       activation=nn.functional.elu           
                                      ) for i in range(self.layer_num-1)])
        
        
        # final projection
        self.edge_GAT.append(SimpleHGNLayer(edge_feats_len = Model_Config_dict['edge_feats_len'], 
                                       in_features_len = GAT_hid_len * self.head_num,
                                       out_features_len = self.class_num,
                                       nhead = 1,
                                       edge_dict = self.all_relation_list,
                                       node_residual=True,
                                       edge_residual_alpha=0.05,
                                       activation=None
                                      ))
        
        
        # 输出预测结果
        self.activation = nn.Softmax(1)#nn.Sigmoid()
        
        self.register_buffer("epsilon", torch.FloatTensor([1e-12]))
        
        self.reset_parameters()

    def reset_parameters(self):
        for edge_GAT_i in self.edge_GAT:
            edge_GAT_i.reset_parameters()

    def forward(self, G):
        
        h = {}
        # 先将涉及到的节点对应的特征都进行转化
        for ntype in G.ntypes:
            h[ntype] = self.Node_Transform_list[ntype]((G.nodes[ntype].data['feat']))
        
        all_type_edge_src_node_feature_adj_dict = {}
        adj_begin_index = {}
        # 存储头节点对应的所有类型边的尾节点特征
        for srctype, etype, dsttype in G.canonical_etypes:
                
            # 存储头节点对应的所有类型边的尾节点特征
            if srctype not in all_type_edge_src_node_feature_adj_dict:
                all_type_edge_src_node_feature_adj_dict[srctype] = defaultdict(list)
                all_type_edge_src_node_feature_adj_dict[srctype]['head_feature'] = h[srctype]
                all_type_edge_src_node_feature_adj_dict[srctype]['res_attn'] = None
                adj_begin_index[srctype] = 0
                # 加入self-loop 
#                 all_type_edge_src_node_feature_adj_dict[srctype]['tail_feature'].append(h[srctype])
#                 
#                 adj = torch.stack([torch.arange(1,h[srctype].shape[0]) for i in range(2)], 0)
#                 tmp_edge_index = self.all_relation_list.index('self_loop')
#                 tmp_edge_index = torch.tensor([tmp_edge_index]*adj.shape[1])
#                 if self.args_cuda:
#                     tmp_edge_index = tmp_edge_index.cuda()
#                     adj = adj.cuda()
#                 all_type_edge_src_node_feature_adj_dict[srctype]['tmp_edge_index'].append(tmp_edge_index)
#                 all_type_edge_src_node_feature_adj_dict[srctype]['Adj'].append(adj)
#                 adj_begin_index[srctype] = adj.shape[1]
            # 保存该类边对应的尾节点特征
            tail_feature = h[dsttype]
            adj = G.all_edges(etype=etype)
            adj = torch.stack(list(adj), dim=0)
            all_type_edge_src_node_feature_adj_dict[srctype]['tail_feature'].append(tail_feature)#.append(tail_feature[adj[1,:]])
            # 由于需要stack，对adj的index进行修改
            adj[1,:] = adj[1,:] + adj_begin_index[srctype]
            adj_begin_index[srctype] = adj_begin_index[srctype] + tail_feature.shape[0]
            all_type_edge_src_node_feature_adj_dict[srctype]['Adj'].append(adj)
            # 生成edge_type的对应index
            tmp_edge_index = self.all_relation_list.index(etype)
            tmp_edge_index = torch.tensor([tmp_edge_index]*adj.shape[1])
            if self.args_cuda:
                tmp_edge_index = tmp_edge_index.cuda()
            all_type_edge_src_node_feature_adj_dict[srctype]['tmp_edge_index'].append(tmp_edge_index)
        
        
      
        for i in range(self.layer_num+1):
            inter_srctype_feaure = {}
            # 对每一个节点，进行边聚合

            for srctype in all_type_edge_src_node_feature_adj_dict:
                # 导入GAT获取结果
                inter_srctype_feaure[srctype] = self.edge_GAT[i](all_type_edge_src_node_feature_adj_dict[srctype], all_type_edge_src_node_feature_adj_dict[srctype]['res_attn'])
                all_type_edge_src_node_feature_adj_dict[srctype]['tail_feature'] = []
            # 更新全部节点特征
            for srctype, etype, dsttype in G.canonical_etypes:
                all_type_edge_src_node_feature_adj_dict[srctype]['head_feature'] = inter_srctype_feaure[srctype][0]
                all_type_edge_src_node_feature_adj_dict[srctype]['res_attn'] = inter_srctype_feaure[srctype][1]
                # 保存该类边对应的尾节点特征
                tail_feature = inter_srctype_feaure[dsttype][0]
                all_type_edge_src_node_feature_adj_dict[srctype]['tail_feature'].append(tail_feature)
                


        # 取出最后一跳目标节点输出的结果
        output_all_nodes = {}
        
        for target_node in self.Target_Node_Type:
            h_prime = all_type_edge_src_node_feature_adj_dict[target_node]['head_feature']

            # L2 Normalization
            # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
            h_prime = h_prime / (torch.max(torch.norm(h_prime, dim=1, keepdim=True), self.epsilon))

            # 转化为概率，并返回预测结果
            output = nn.Softmax(1)(h_prime)
            if self.class_num<=2:
                output = output[:,0]

            output_all_nodes[target_node] = output
        
        if len(self.Target_Node_Type) == 1:
            output_all_nodes = output
        
        return output_all_nodes