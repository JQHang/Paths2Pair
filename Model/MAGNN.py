import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
from kg_model.Basic_NN import Res_DNN
    
# 针对source-target结构的GAT模型
class Intra_Metapath_Conv(nn.Module):
    def __init__(self, in_features, hid_features, dropout, bias=True):
        super().__init__()

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
    def forward(self, edge_list, feature_dict, node_type_list):
        # 获得路径上各列对应的点经过转化后的特征
        feature_h_list = []
        for tmp_feature_i in range(len(edge_list)):
            tmp_feature_index = edge_list[tmp_feature_i]
            
            # 先对特征进行转化
            tmp_feature_h = self.feat_linear(feature_dict[tmp_feature_i])
            
            # 再取出目标数据
            tmp_feature_h = tmp_feature_h[tmp_feature_index]
            
            feature_h_list.append(tmp_feature_h)
            
        # 算对路径上的点的encoder(目前只有均值的方案)
        feature_h_encoder = torch.mean(torch.stack(feature_h_list, 0), 0)
        
        # 和起始节点的特征合并
        a_input = torch.cat([feature_h_list[0], feature_h_encoder], dim=1)
        
        # 计算每条边对应的注意力权重
        e = torch.tanh(self.att(a_input))
        
        # 获得头结点到转化后的embedding间的关系
        head_tail_edge_list = torch.stack([edge_list[0], torch.arange(0, tmp_feature_h.size(0), dtype = torch.long).to(tmp_feature_h.device)], 0)
        
        # 稀疏矩阵
        attention = torch.sparse.FloatTensor(head_tail_edge_list, e[:, 0], 
                                 (feature_dict[0].size(0), feature_h_encoder.size(0)))
    
        attention = torch.sparse.softmax(attention, dim=1)
        
        h_prime = torch.sparse.mm(attention, feature_h_encoder)
        
        if self.bias is not None:
            h_prime = h_prime + self.bias

        return h_prime
    
    
# 对同种元路径的各条结果进行拼接，再利用注意力机制，综合各种结果
class MAGNN(nn.Module):
    def __init__(self, node_type_to_feature_len_dict, all_meta_path_list, node_feature_hid_len, metapath_level_nhid, num_Res_DNN,
                 each_Res_DNN_num, dropout=0.5):
        super().__init__()
        
        self.h_dropout = nn.Dropout(dropout)
        
        self.all_meta_path_list = all_meta_path_list
        
        # 特征转化
        self.Node_Transform_list = {}
        for tmp_node_type in node_type_to_feature_len_dict:
            tmp_linear = nn.Linear(node_type_to_feature_len_dict[tmp_node_type], node_feature_hid_len)
            self.Node_Transform_list[tmp_node_type] = tmp_linear
            self.add_module('{}_Node_Transform'.format(tmp_node_type), self.Node_Transform_list[tmp_node_type])
        
        # 有多少种类型的元路径，每种元路径有多少条，就生成多少个GAT
        self.Intra_Metapath = {}
        for tmp_meta_path_name in all_meta_path_list:
            tmp_attention = Intra_Metapath_Conv(in_features = node_feature_hid_len, hid_features = metapath_level_nhid, dropout = dropout)
            self.Intra_Metapath[tmp_meta_path_name] = tmp_attention
            self.add_module('intra_metapath_attention_{}'.format(tmp_meta_path_name), tmp_attention)
        
        # 综合各组元路径的结果
        self.semantic_att = nn.Linear(metapath_level_nhid, 1)

        # 输出预测结果
        self.output_linear = Res_DNN(metapath_level_nhid, metapath_level_nhid, 1, dropout, num_Res_DNN, each_Res_DNN_num)
        
        self.activation = nn.Sigmoid()
        
        self.reset_parameters()

    def reset_parameters(self):
        for tmp_meta_path_name in self.all_meta_path_list:
            self.Intra_Metapath[tmp_meta_path_name].reset_parameters()

    def forward(self, sub_graph_data_dict):
        # 先将特征进行映射
        for tmp_meta_path_name in sub_graph_data_dict['Feature']:
            if tmp_meta_path_name == 'src_feat':
                tmp_node_type = sub_graph_data_dict['Feature_Node_Type'][tmp_meta_path_name]
                sub_graph_data_dict['Feature'][tmp_meta_path_name] = self.Node_Transform_list[tmp_node_type](sub_graph_data_dict['Feature'][tmp_meta_path_name])
            else:
                for tmp_path_i in sub_graph_data_dict['Feature'][tmp_meta_path_name].keys():
                    tmp_node_type = sub_graph_data_dict['Feature_Node_Type'][tmp_meta_path_name][tmp_path_i]
                    sub_graph_data_dict['Feature'][tmp_meta_path_name][tmp_path_i] = self.Node_Transform_list[tmp_node_type](sub_graph_data_dict['Feature'][tmp_meta_path_name][tmp_path_i])
        
        ####################################################################################################################
        # Intra_Metapath 获取各元路径对应的结果
        tmp_metapath_output_h_list = [sub_graph_data_dict['Feature']['src_feat']]
        for tmp_meta_path_name in self.all_meta_path_list:
            if sub_graph_data_dict['Adj'][tmp_meta_path_name][0].size()[0] > 0:
                # 添加上起始点的特征
                sub_graph_data_dict['Feature'][tmp_meta_path_name][0] = sub_graph_data_dict['Feature']['src_feat']
                
                # 导入GAT获取结果
                meta_path_based_h = self.Intra_Metapath[tmp_meta_path_name](sub_graph_data_dict['Adj'][tmp_meta_path_name],
                                                        sub_graph_data_dict['Feature'][tmp_meta_path_name], 
                                                        sub_graph_data_dict['Feature_Node_Type'][tmp_meta_path_name])
            else:
                meta_path_based_h = torch.zeros(sub_graph_data_dict['Feature']['src_feat'].size()[0],
                                                sub_graph_data_dict['Feature']['src_feat'].size()[1])
                if args_cuda:
                    meta_path_based_h = meta_path_based_h.cuda()
                
            tmp_metapath_output_h_list.append(meta_path_based_h)
        
        ####################################################################################################################
        # Inter_Metapath 合并各元路径的结果
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