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
#         print("activate",hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
#         print("Norm",hidden_states)
        return hidden_states

class Res_DNN(nn.Module):
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

class base_Res_DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, num_Res_DNN = 1, num_DNN = 2):
        super().__init__()
        # 先将数据降维
        self.prepare = nn.Linear(input_size, hidden_size) 
        
        # 再导入两轮3层Res_DNN
        self.multi_Res_DNN = nn.ModuleList([Res_DNN(hidden_size, dropout, num_DNN) for _ in range(num_Res_DNN)])
        
        # 输出层，简单的一个线性层，从hidden_size映射到num_labels
        self.classifier = nn.Linear(hidden_size, output_size) 
        
    def forward(self, input_ids):
        hidden_states = self.prepare(input_ids)
        
        for i,layer_module in enumerate(self.multi_Res_DNN):
            hidden_states = layer_module(hidden_states)
        
        hidden_states = self.classifier(hidden_states)
        
        hidden_states = hidden_states.squeeze()
    
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
        
    # 由source到target
    def forward(self, source_h, target_h, edge_list):
        source_h = self.h_dropout(source_h)
        target_h = self.h_dropout(target_h)
        
        source_h = self.feat_linear(source_h)
        target_h = self.feat_linear(target_h)
        
        source_idx, target_idx = edge_list
        
        a_input = torch.cat([source_h[source_idx], target_h[target_idx]], dim=1)
        
        e = torch.tanh(self.att(a_input))
    
        # 稀疏矩阵
        attention = torch.sparse.FloatTensor(edge_list, e[:, 0], (source_h.size(0), target_h.size(0)))
    
        attention = torch.sparse.softmax(attention, dim=1)
        
        h_prime = torch.sparse.mm(attention, target_h)
        
        if self.bias is not None:
            h_prime = h_prime + self.bias

        return h_prime
    
# 对同种元路径的各条结果进行拼接，再利用注意力机制，综合各种结果
class HAN(nn.Module):
    def __init__(self, node_type_to_feature_len_dict, all_meta_path_list, node_feature_hid_len, metapath_level_nhid, num_Res_DNN,
                 each_Res_DNN_num, dropout=0.5):
        super(HAN, self).__init__()
        
        self.h_dropout = nn.Dropout(dropout)
        
        self.all_meta_path_list = all_meta_path_list
        
        # 特征转化
        self.Node_Transform_list = {}
        for tmp_node_type in node_type_to_feature_len_dict:
            tmp_linear = nn.Linear(node_type_to_feature_len_dict[tmp_node_type], node_feature_hid_len)
            self.Node_Transform_list[tmp_node_type] = tmp_linear
            self.add_module('{}_Node_Transform'.format(tmp_node_type), self.Node_Transform_list[tmp_node_type])
        
        # 有多少种类型的元路径，每种元路径有多少条，就生成多少个GAT
        self.edge_GAT = {}
        for tmp_meta_path_name in all_meta_path_list:
            tmp_attention = GATConv(in_features = node_feature_hid_len, hid_features = metapath_level_nhid, dropout = dropout)
            self.edge_GAT[tmp_meta_path_name] = tmp_attention
            self.add_module('intra_metapath_attention_{}'.format(tmp_meta_path_name), tmp_attention)
        
        # 综合各组元路径的结果
        self.semantic_att = nn.Linear(metapath_level_nhid, 1)

        # 输出预测结果
        self.output_linear = base_Res_DNN(metapath_level_nhid, metapath_level_nhid, 1, dropout, num_Res_DNN, each_Res_DNN_num)
        
        self.activation = nn.Sigmoid()
        
        self.reset_parameters()

    def reset_parameters(self):
        for tmp_meta_path_name in self.all_meta_path_list:
            self.edge_GAT[tmp_meta_path_name].reset_parameters()

    def forward(self, sub_graph_data_dict):
        # 先将特征进行映射
        for tmp_meta_path_name in sub_graph_data_dict['Feature']:
            tmp_node_type = sub_graph_data_dict['Feature_Node_Type'][tmp_meta_path_name]
            sub_graph_data_dict['Feature'][tmp_meta_path_name] = self.Node_Transform_list[tmp_node_type](sub_graph_data_dict['Feature'][tmp_meta_path_name])
        
        ####################################################################################################################
        # node level attention
        
        # 获取各元路径对应的结果
        tmp_metapath_output_h_list = [sub_graph_data_dict['Feature']['src_feat']]
        for tmp_meta_path_name in self.all_meta_path_list:
            if sub_graph_data_dict['Adj'][tmp_meta_path_name].size()[1] > 0:
                # 导入GAT获取结果
                meta_path_based_h = self.edge_GAT[tmp_meta_path_name](sub_graph_data_dict['Feature']['src_feat'], 
                                                    sub_graph_data_dict['Feature'][tmp_meta_path_name], 
                                                    sub_graph_data_dict['Adj'][tmp_meta_path_name])
            else:
                meta_path_based_h = torch.zeros(sub_graph_data_dict['Feature']['src_feat'].size()[0],
                                                sub_graph_data_dict['Feature']['src_feat'].size()[1])
                if args_cuda:
                    meta_path_based_h = meta_path_based_h.cuda()
                
            tmp_metapath_output_h_list.append(meta_path_based_h)
        
        ####################################################################################################################
        # semantic level attention
        
        # 先合并各元路径的结果
        semantic_embeddings = torch.stack(tmp_metapath_output_h_list, dim=1)
        semantic_embeddings = self.h_dropout(semantic_embeddings)
        
        semantic_attention = self.semantic_att(semantic_embeddings).squeeze()
        semantic_attention = F.leaky_relu(semantic_attention)
        semantic_attention = torch.softmax(semantic_attention, dim=1)
        
        semantic_attention = semantic_attention.unsqueeze(2)
        semantic_attention = semantic_attention.expand(semantic_attention.shape[0],semantic_attention.shape[1],
                                                       semantic_embeddings.shape[-1])
        
        h_prime = semantic_attention * semantic_embeddings
        
        # 待检查
        h_prime = torch.sum(h_prime, dim=1)
        
        ####################################################################################################################
        # 返回预测结果
        output = self.activation(self.output_linear(h_prime))
        
        return output