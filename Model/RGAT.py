import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim

# 残差网络，用于转换特征，以及输出最终预测结果
class DNN(nn.Module):
    def __init__(self, input_size, output_size, dropout = 0.5):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(output_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class Res_DNN_layer(nn.Module):
    def __init__(self, hidden_size, dropout, num_DNN):
        super().__init__()
        self.multi_DNN = nn.ModuleList([DNN(hidden_size, hidden_size, dropout) for _ in range(num_DNN)])
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        
        hidden_states_shortcut = hidden_states
        for i,layer_module in enumerate(self.multi_DNN):
            hidden_states = layer_module(hidden_states)
        hidden_states = hidden_states_shortcut + hidden_states
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        
        return hidden_states

class Res_DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, num_Res_DNN, num_DNN):
        super().__init__()
        # 先将数据降维
        self.prepare = nn.Linear(input_size, hidden_size) 
        
        # 再导入两轮3层Res_DNN
        self.multi_Res_DNN = nn.ModuleList([Res_DNN_layer(hidden_size, dropout, num_DNN) for _ in range(num_Res_DNN)])
        
        # 输出层，简单的一个线性层，从hidden_size映射到num_labels
        self.classifier = nn.Linear(hidden_size, output_size) 
        
    def forward(self, input_ids):
        hidden_states = self.prepare(input_ids)
        
        for i,layer_module in enumerate(self.multi_Res_DNN):
            hidden_states = layer_module(hidden_states)
        
        hidden_states = self.classifier(hidden_states)
    
        return hidden_states
    
# 针对source-target结构的GAT模型
class GATConv(nn.Module):
    def __init__(self, in_features, hid_features, dropout, bias=True):
        super(GATConv, self).__init__()

        self.in_features = in_features
        
        self.h_dropout = nn.Dropout(dropout)
        
        self.feat_linear = nn.Linear(in_features, hid_features)
        
        self.att = nn.Linear(2*hid_features, 1)
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(hid_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data.fill_(0)
        
    # 由source到dst
    def forward(self, dst_h, source_h, edge_list):
        dst_h = self.h_dropout(dst_h)
        source_h = self.h_dropout(source_h)
        
        dst_h = self.feat_linear(dst_h)
        source_h = self.feat_linear(source_h)
        
        dst_idx, source_idx = edge_list
        
        a_input = torch.cat([dst_h[dst_idx], source_h[source_idx]], dim=1)
        
        e = torch.tanh(self.att(a_input))
    
        # 稀疏矩阵
        attention = torch.sparse.FloatTensor(edge_list, e[:, 0], (dst_h.size(0), source_h.size(0)))
    
        attention = torch.sparse.softmax(attention, dim=1)
        
        h_prime = torch.sparse.mm(attention, source_h)
        
        if self.bias is not None:
            h_prime = h_prime + self.bias

        return h_prime
    
class RGAT(nn.Module):
    def __init__(self, node_type_to_feature_len_dict, all_relation_list, node_feature_hid_len, num_Res_DNN, each_Res_DNN_num, dropout=0.5):
        super().__init__()
        
        self.h_dropout = nn.Dropout(dropout)
        
        # 特征转化
        self.Node_Transform_list = {}
        for tmp_node_type in node_type_to_feature_len_dict:
            tmp_linear = nn.Linear(node_type_to_feature_len_dict[tmp_node_type], node_feature_hid_len)
            self.Node_Transform_list[tmp_node_type] = tmp_linear
            self.add_module('{}_Node_Transform'.format(tmp_node_type), self.Node_Transform_list[tmp_node_type])
        
        # 有多少种类型的元路径，每种元路径有多少条，就生成多少个GAT
        self.edge_GAT = {}
        for tmp_relation_name in all_relation_list:
            tmp_attention = GATConv(in_features = node_feature_hid_len, hid_features = node_feature_hid_len, dropout = dropout)
            self.edge_GAT[tmp_relation_name] = tmp_attention
            self.add_module('Edge_GAT_{}'.format(tmp_relation_name), tmp_attention)
        
        # 输出预测结果
        self.output_linear = Res_DNN(node_feature_hid_len, node_feature_hid_len, 1, dropout, num_Res_DNN, each_Res_DNN_num)
        
        self.activation = nn.Sigmoid()
        
        self.reset_parameters()

    def reset_parameters(self):
        for tmp_relation_name in self.edge_GAT:
            self.edge_GAT[tmp_relation_name].reset_parameters()

    def forward(self, sub_graph_data_dict):
        hop_num = len(sub_graph_data_dict['Adj'].keys())
        
        # 先将涉及到的节点对应的特征都进行转化
        for tmp_node_type in sub_graph_data_dict['Feature']:
            transferred_node_feature = self.Node_Transform_list[tmp_node_type](sub_graph_data_dict['Feature'][tmp_node_type])
            sub_graph_data_dict['Feature'][tmp_node_type] = transferred_node_feature
        
        # 从最后一跳往前进行运算
        for tmp_hop in range(hop_num - 1, -1, -1):
            src_node_feature_list_dict = {}
                
            # 再对每个关系运算GAT
            for tmp_relation_name in sub_graph_data_dict['Adj'][tmp_hop]:
                # 获取该关系头节点类型
                tmp_head_node_type = sub_graph_data_dict['Relation_Node_Type'][tmp_relation_name]['Head_type']
                
                # 获取该关系尾节点类型
                tmp_tail_node_type = sub_graph_data_dict['Relation_Node_Type'][tmp_relation_name]['Tail_type']
                
                if sub_graph_data_dict['Adj'][tmp_hop][tmp_relation_name].size()[1] > 0:
                    # 导入GAT获取结果
                    subgraph_based_h = self.edge_GAT[tmp_relation_name](sub_graph_data_dict['Feature'][tmp_head_node_type], 
                                                       sub_graph_data_dict['Feature'][tmp_tail_node_type], 
                                                       sub_graph_data_dict['Adj'][tmp_hop][tmp_relation_name])
                else:
                    subgraph_based_h = torch.zeros(sub_graph_data_dict['Feature'][tmp_hop][tmp_head_node_type].size()[0],
                                         sub_graph_data_dict['Feature'][tmp_hop][tmp_head_node_type].size()[1])
                    if args_cuda:
                        subgraph_based_h = subgraph_based_h.cuda()
                
                if tmp_head_node_type not in src_node_feature_list_dict:
                    src_node_feature_list_dict[tmp_head_node_type] = [sub_graph_data_dict['Feature'][tmp_head_node_type]]
                src_node_feature_list_dict[tmp_head_node_type].append(subgraph_based_h)
                
            # aggreate 各关系的结果
            for tmp_node_type in src_node_feature_list_dict:
                sub_graph_data_dict['Feature'][tmp_node_type] = torch.mean(torch.stack(src_node_feature_list_dict[tmp_node_type], 0), 0)
            
        # 取出最后一跳目标节点输出的结果
        target_node_type = sub_graph_data_dict['Target_Node_Type']
        h_prime = sub_graph_data_dict['Feature'][target_node_type][sub_graph_data_dict['Target_Node_Index']]
        
        # 转化为概率，并返回预测结果
        output = self.activation(self.output_linear(h_prime))
        
        return output