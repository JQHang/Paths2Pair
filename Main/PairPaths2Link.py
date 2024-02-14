#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import io
import copy
import re
import gc
import json
import os
import sys
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from random import sample
from collections import defaultdict
from tqdm import tqdm
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from py4j.protocol import Py4JError, Py4JJavaError, Py4JNetworkError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim

# Self packages path
Src_Dir_path = "../../../.."
sys.path.append(Src_Dir_path)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Self packages
from Utils.Pyspark_utils import ResilientSparkRunner
from Utils.Pyspark_utils import DEFAULT_STORAGE_LEVEL
from Utils.utils import read_json_config_file
from Utils.utils import mkdir
from Utils.utils import Log_save


# In[2]:


#显示所有列
pd.set_option('display.max_columns', None)

#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

# 固定随机值
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
setup_seed(42)


# # 数据配置信息

# In[3]:


from Utils.Pyspark_utils import Data_In_HDFS_Path
from Utils.utils import read_json_config_file

Data_Dir_Name = '/paths2pairs_0111'

# 数据存储位置
Data_Output_path = Src_Dir_path + "/.."

# 具体的图相关数据存储位置
Graph_Output_path = Data_Output_path + '/Graph'
mkdir(Graph_Output_path)

# 具体的本次数据的存储位置
Feature_Data_From_Online_Store_dir = Graph_Output_path + Data_Dir_Name
mkdir(Feature_Data_From_Online_Store_dir)

# 具体的时间区间
data_time_range_str = '2022-10-01_2023-10-01'
Feature_Data_From_Online_Time_Store_dir = (Feature_Data_From_Online_Store_dir + '/' + data_time_range_str)
mkdir(Feature_Data_From_Online_Time_Store_dir)
print('目标时间数据本地存储文件夹', Feature_Data_From_Online_Time_Store_dir)


# # 读取并转化本地数据

# In[4]:


# 直接从内存读取数据的代码
def read_local_dataset():
    dataset_in_memory = {}
    
    dataset_local_dir = Feature_Data_From_Online_Time_Store_dir + '/Data_Prepare'

    # 依次处理各数据集
    for data_type in ['train', 'val', 'test']:
        # 本地结果保存位置
        result_dir = dataset_local_dir + f'/{data_type}'

        dataset_in_memory[data_type] = {}
        
        sample_pd = pd.read_pickle(result_dir + '/sample_pairs.pkl')
        sample_pd['Feature_Time'] = sample_pd['Feature_Time'].apply(lambda x:str(x))
        
        # 删除没有label等于1的公司
        valid_companies = sample_pd[sample_pd['Label'] == 1][['Company_Node_UID', 'Feature_Time']].drop_duplicates()
        sample_pd = sample_pd.merge(valid_companies, on = ['Company_Node_UID', 'Feature_Time'], how = 'inner')
        
        dataset_in_memory[data_type]['sample_pairs'] = sample_pd
        
        # 将各个path的数据转化为pd
        dataset_in_memory[data_type]['main_paths'] = []
        dataset_in_memory[data_type]['aux_paths'] = []
        main_path_names = ['0_0', '1_3', '1_5']
        aux_path_names = ['1_12', '1_7', '2_0', '2_30', '2_37', '2_40', '2_5', '2_7', '3_0', '3_1', '3_20', '3_21', '3_22',
                          '3_28', '3_30', '3_32', '3_36', '3_37', '3_7']
        for path_index, path_name in enumerate(main_path_names + aux_path_names):
            print(path_name)
            
            # 确定是主路径还是辅助路径
            if path_index < len(main_path_names):
                path_type = 'main_paths'
            else:
                path_type = 'aux_paths'
                
            with open(result_dir + f"/{path_type}/{path_name}.json", "r", encoding="utf-8") as file:
                path_config = json.load(file)

            path_pd = pd.read_pickle(result_dir + f"/{path_type}/{path_name}.pkl")
            
            path_pd['Feature_Time'] = path_pd['Feature_Time'].apply(lambda x:str(x))
            
            path = {}
            path['data'] = path_pd
            path['path_config'] = path_config
            path['path_name'] = path_name
            
            # 保存结果
            if path_type == 'main_paths':
                dataset_in_memory[data_type]['main_paths'].append(path)
            else:
                dataset_in_memory[data_type]['aux_paths'].append(path)
            
    # 将点的特征读取为pd
    dataset_in_memory['node_feature'] = {}
    mkdir(dataset_local_dir + "/node_feature")
    for node_type in ['Company_Node', 'Mobile_Node', 'Traceid_Node', 'User_Pin_Node']:
        print(node_type)

        node_pd = pd.read_pickle(dataset_local_dir + f"/node_feature/{node_type}.pkl")

        dataset_in_memory['node_feature'][node_type] = node_pd
            
    return dataset_in_memory

dataset = read_local_dataset()


# In[ ]:


transferred_dataset = {}

# 记录各个路径上各点和边的类型
transferred_dataset['paths_types'] = []

# 记录各类型的边和点对应的特征长度
transferred_dataset['typ_feat_len'] = {}

##############################################################################################################################
# 依次转化各集合内的数据
data_types = ['train', 'val', 'test']
for data_type in data_types:
    print(data_type)
    transferred_dataset[data_type] = {}
    
    sample_pairs = dataset[data_type]['sample_pairs']

    # 过滤掉有null值的行
    sample_pairs = sample_pairs[sample_pairs['Mobile_Node_UID'].notnull()].reset_index(drop = True)
    sample_pairs['pair_id'] = sample_pairs.index

    # 获得pair_id的映射表
    pair_id_map_pd = sample_pairs[['Company_Node_UID', 'Mobile_Node_UID', 'Feature_Time', 'pair_id']]
    
    # 获得各pair的标签
    pair_label_pd = sample_pairs[['pair_id', 'Label']]
    
    # 统计正负样本数
    print('正样本数:', pair_label_pd[pair_label_pd['Label'] == 1].shape[0])
    print('负样本数:', pair_label_pd[pair_label_pd['Label'] == 0].shape[0])
    
    # 保存numpy格式的标签
    transferred_dataset[data_type]['label'] = pair_label_pd['Label'].values

    # 获取正样本的序号
    transferred_dataset[data_type]['pos_label_index'] = np.argwhere(transferred_dataset[data_type]['label'] == 1).T[0]

    # 获取负样本的序号
    transferred_dataset[data_type]['neg_label_index'] = np.argwhere(transferred_dataset[data_type]['label'] == 0).T[0]
    
    # 为每个公司生成id
    company_name_to_id = sample_pairs[['Company_Node_UID', 'Feature_Time']].drop_duplicates(ignore_index = True)
    company_name_to_id = company_name_to_id.reset_index(drop = True)
    company_name_to_id['company_id'] = company_name_to_id.index
    
    # 获得公司id和pair_id的对应表
    company_id_to_pair_id = pair_id_map_pd.merge(company_name_to_id, on = ['Company_Node_UID', 'Feature_Time'], how = 'left')
    transferred_dataset[data_type]['company_id'] = company_id_to_pair_id['company_id'].values
    
    # 获得各行pair的公司特征
    company_feature_pd = sample_pairs[['Company_Node_UID', 'Feature_Time']].merge(dataset['node_feature']['Company_Node'],
                                                                                   left_on = ['Company_Node_UID', 'Feature_Time'],
                                                                                   right_on = ['Company_Node', 'Feature_Time'],
                                                                                   how = 'left')
    company_feature_pd = company_feature_pd.drop(['Company_Node_UID', 'Company_Node', 'Feature_Time'], axis = 1)
    company_feature_pd = company_feature_pd.fillna(0)
    transferred_dataset[data_type]['Company_Node'] = company_feature_pd.to_numpy()
                
    # 获得各行pair的手机号特征
    phone_feature_pd = sample_pairs[['Mobile_Node_UID', 'Feature_Time']].merge(dataset['node_feature']['Mobile_Node'],
                                                                               left_on = ['Mobile_Node_UID', 'Feature_Time'],
                                                                               right_on = ['Mobile_Node', 'Feature_Time'],
                                                                               how = 'left')
    phone_feature_pd = phone_feature_pd.drop(['Mobile_Node_UID', 'Mobile_Node', 'Feature_Time'], axis = 1)
    phone_feature_pd = phone_feature_pd.fillna(0)
    transferred_dataset[data_type]['Mobile_Node'] = phone_feature_pd.to_numpy()
    
    ##############################################################################################################################
    # 转化各个路径
    transferred_dataset[data_type]['paths'] = []
    for path_type in ['main_paths', 'aux_paths']:
        print(path_type)
        
        for path in dataset[data_type][path_type]:
            print(path['path_name'])
            
            path_config = path['path_config']
            
            # 获取节点列列名
            node_columns = [path_config[0]['head_node_std_name']] + [x['tail_node_std_name'] for x in path_config]
            print(node_columns)
            
            # 先过滤掉null值的节点列
            path_pd = path['data'].dropna(subset = node_columns)
            
            # 给特征列中的null值补0
            path_pd = path_pd.fillna(0)
            
            # 显示path数目
            print(path_pd.shape)
            
            # 补全各行数据对应的pair_id(没merge上的就不要了)
            path_to_pair_id = path_pd.merge(pair_id_map_pd, left_on = [path_config[0]['head_node_std_name'], path_config[-1]['tail_node_std_name'], 'Feature_Time'],
                                            right_on = ['Company_Node_UID', 'Mobile_Node_UID', 'Feature_Time'])
            
            # 给保留下的path生成path_id
            path_to_pair_id = path_to_pair_id.reset_index(drop = True)
            path_to_pair_id['path_id'] = path_to_pair_id.index
            
            # 获取pair_id到path_id的映射表
            path_id_to_pair_id = path_to_pair_id[['path_id', 'pair_id']]
            
            path_to_pair = path_id_to_pair_id.to_numpy().T
            path_to_pair_dict = dict(zip(path_to_pair[0], path_to_pair[1]))
            
            # 将path上的点和边按顺序拆成list，并将节点信息转化为特征
            path_feats = []
            path_types = []
            
            # 先处理起始点(一定得是left join防止path id的对应关系出现偏差)
            start_node_name = path_config[0]['head_node_std_name']
            start_node_type = path_config[0]['head_node_type']
            
            path_node_feat_pd = path_to_pair_id[[start_node_name, 'Feature_Time']].merge(dataset['node_feature'][start_node_type],
                                                                                         left_on = [start_node_name, 'Feature_Time'],
                                                                                         right_on = [start_node_type, 'Feature_Time'],
                                                                                         how = 'left')
            path_node_feat_pd = path_node_feat_pd.drop([start_node_name, start_node_type, 'Feature_Time'], axis = 1)
            path_node_feat_pd = path_node_feat_pd.fillna(0)
            
            print(start_node_name, '点特征长度:', path_node_feat_pd.shape)
            
            path_feats.append(path_node_feat_pd.to_numpy())
            path_types.append(start_node_type)
            
            # 记录该节点类型对应的特征长度
            transferred_dataset['typ_feat_len'][start_node_type] = path_node_feat_pd.shape[1]
            
            # 将其余各点对应的特征及各点本身的id号存入list
            for hop_k, node_column in enumerate(node_columns[1:]):
                # 获得该点对应的边类型，以及节点类型
                edge_type = path_config[hop_k]['edge_table_name']
                node_type = path_config[hop_k]['tail_node_type']
                
                # 通过边的名称找到各跳对应的边特征
                if data_type == 'train':
                    pattern = f"Edge_Feat_.*_to_{node_column}"
                else:
                    pattern = f"Hop_{hop_k}_Edge_Feat_.*"
                edge_feature_pd = path_to_pair_id.filter(regex = pattern)
                edge_feature_pd = edge_feature_pd.fillna(0)
                
                print(node_column, '边特征长度:', edge_feature_pd.shape)
                
                path_feats.append(edge_feature_pd.to_numpy())
                path_types.append(edge_type)

                # 记录该节点类型对应的特征长度
                transferred_dataset['typ_feat_len'][edge_type] = edge_feature_pd.shape[1]
            
                path_node_feat_pd = path_to_pair_id[[node_column, 'Feature_Time']].merge(dataset['node_feature'][node_type],
                                                                                         left_on = [node_column, 'Feature_Time'],
                                                                                         right_on = [node_type, 'Feature_Time'],
                                                                                         how = 'left')
                path_node_feat_pd = path_node_feat_pd.drop([node_column, node_type, 'Feature_Time'], axis = 1)
                path_node_feat_pd = path_node_feat_pd.fillna(0)
                
                print(node_column, '点特征长度:', path_node_feat_pd.shape)
            
                path_feats.append(path_node_feat_pd.to_numpy())
                path_types.append(node_type)
                
                # 记录该节点类型对应的特征长度
                transferred_dataset['typ_feat_len'][node_type] = path_node_feat_pd.shape[1]
            
            # 保存所使用的的路径配置
            if data_type == data_types[0]:
                transferred_dataset['paths_types'].append(path_types)
            
            # 保存路径具体数据
            path = {'data': path_feats, 'path_to_pair': path_to_pair, 'path_to_pair_dict': path_to_pair_dict}
            transferred_dataset[data_type]['paths'].append(path)


# # 随机采样部分

# In[ ]:


def paths2pairs_random_sample(dataset, sample_size, positive_percent):
    sampled_dataset = {}
    
    if sample_size > 0:
        if positive_percent > 0:
            pos_sample_size = math.ceil(sample_size * positive_percent)
            neg_sample_size = (sample_size - pos_sample_size)

            # 随机采样指定数量指定比例的正负样本对应的index号
            pos_sample_index = np.random.choice(dataset['pos_label_index'], size = pos_sample_size, replace = False)
            neg_sample_index = np.random.choice(dataset['neg_label_index'], size = neg_sample_size, replace = False)

            # 获得全部目标pair的id号
            sample_index = np.concatenate((pos_sample_index, neg_sample_index))
        else:
            sample_index = np.random.choice(np.arange(0, dataset['label'].size), size = sample_size, replace = False)
            
        # 获得这些pair对应的标签
        sampled_dataset['label'] = dataset['label'][sample_index]

        # 获得这些pair的新序号与pair_id间的对应关系
        sample_index_trans_dict = dict(zip(sample_index, range(len(sample_index))))

        # 获取涉及到的公司id
        sampled_dataset['company_id'] = dataset['company_id'][sample_index]
        
        # 获得对应的公司和手机特征
        sampled_dataset['Company_Node'] = dataset['Company_Node'][sample_index]
        sampled_dataset['Mobile_Node'] = dataset['Mobile_Node'][sample_index]
        
        # 依次处理各个路径
        sampled_dataset['paths'] = []
        for path in dataset['paths']:
            # 获取对应样本所对应的全部路径对应的序号
            sample_path_index = path['path_to_pair'][0, np.isin(path['path_to_pair'][1], sample_index)]

            # 若该类型路径没对应到任何样本
            if len(sample_path_index) == 0:
                # 返回空值
                sampled_dataset['paths'].append({'data': [], 'path_to_pair': np.array([[],[]])})
                continue
            
            # 获得保留的路径对应的pair id号
            sample_path_pair_index = np.vectorize(path['path_to_pair_dict'].get)(sample_path_index)

            # 获得保留的路径对应的新的pair id号
            sample_path_sample_pair_index = np.vectorize(sample_index_trans_dict.get)(sample_path_pair_index)

            # 记录保留的路径和新的pair id号之间的对应关系
            sample_path_to_sample_pair = np.array([range(len(sample_path_index)), sample_path_sample_pair_index])

            # 基于路径id号，只保留指定的路径
            sample_path_feats = []
            for path_feats in path['data']:
                sample_path_feats.append(path_feats[sample_path_index])

            sampled_dataset['paths'].append({'data': sample_path_feats, 'path_to_pair': sample_path_to_sample_pair})
    else:
        sampled_dataset = dataset
        
    # 将数据都转tensor，并放入指定环境
    sampled_dataset['label'] = torch.FloatTensor(sampled_dataset['label']).to(device_type)
    sampled_dataset['Company_Node'] = torch.FloatTensor(sampled_dataset['Company_Node']).to(device_type)
    sampled_dataset['Mobile_Node'] = torch.FloatTensor(sampled_dataset['Mobile_Node']).to(device_type)
    for path_index in range(len(sampled_dataset['paths'])):
        for node_edge_index in range(len(sampled_dataset['paths'][path_index]['data'])):
            sampled_dataset['paths'][path_index]['data'][node_edge_index] = torch.FloatTensor(sampled_dataset['paths'][path_index]['data'][node_edge_index]).to(device_type)
        sampled_dataset['paths'][path_index]['path_to_pair'] = torch.LongTensor(sampled_dataset['paths'][path_index]['path_to_pair']).to(device_type)
        
    return sampled_dataset

# sampled_dataset = paths2pairs_random_sample(transferred_dataset['train'], 2000, 0.1)


# In[ ]:


def paths2pairs_sample_with_index(dataset, sample_index):
    sampled_dataset = {}
    
    if len(sample_index) > 0:
        # 获得这些pair对应的标签
        sampled_dataset['label'] = dataset['label'][sample_index]

        # 获得这些pair的新序号与pair_id间的对应关系
        sample_index_trans_dict = dict(zip(sample_index, range(len(sample_index))))

        # 获取涉及到的公司id
        sampled_dataset['company_id'] = dataset['company_id'][sample_index]
        
        # 获得对应的公司和手机特征
        sampled_dataset['Company_Node'] = dataset['Company_Node'][sample_index]
        sampled_dataset['Mobile_Node'] = dataset['Mobile_Node'][sample_index]
        
        # 依次处理各个路径
        sampled_dataset['paths'] = []
        for path in dataset['paths']:
            # 获取对应样本所对应的全部路径对应的序号
            sample_path_index = path['path_to_pair'][0, np.isin(path['path_to_pair'][1], sample_index)]

            # 若该类型路径没对应到任何样本
            if len(sample_path_index) == 0:
                # 返回空值
                sampled_dataset['paths'].append({'data': [], 'path_to_pair': np.array([[],[]])})
                continue
            
            # 获得保留的路径对应的pair id号
            sample_path_pair_index = np.vectorize(path['path_to_pair_dict'].get)(sample_path_index)

            # 获得保留的路径对应的新的pair id号
            sample_path_sample_pair_index = np.vectorize(sample_index_trans_dict.get)(sample_path_pair_index)

            # 记录保留的路径和新的pair id号之间的对应关系
            sample_path_to_sample_pair = np.array([range(len(sample_path_index)), sample_path_sample_pair_index])

            # 基于路径id号，只保留指定的路径
            sample_path_feats = []
            for path_feats in path['data']:
                sample_path_feats.append(path_feats[sample_path_index])

            sampled_dataset['paths'].append({'data': sample_path_feats, 'path_to_pair': sample_path_to_sample_pair})
    else:
        sampled_dataset = dataset
        
    # 将数据都转tensor，并放入指定环境
    sampled_dataset['label'] = torch.FloatTensor(sampled_dataset['label']).to(device_type)
    sampled_dataset['Company_Node'] = torch.FloatTensor(sampled_dataset['Company_Node']).to(device_type)
    sampled_dataset['Mobile_Node'] = torch.FloatTensor(sampled_dataset['Mobile_Node']).to(device_type)
    for path_index in range(len(sampled_dataset['paths'])):
        for node_edge_index in range(len(sampled_dataset['paths'][path_index]['data'])):
            sampled_dataset['paths'][path_index]['data'][node_edge_index] = torch.FloatTensor(sampled_dataset['paths'][path_index]['data'][node_edge_index]).to(device_type)
        sampled_dataset['paths'][path_index]['path_to_pair'] = torch.LongTensor(sampled_dataset['paths'][path_index]['path_to_pair']).to(device_type)
        
    return sampled_dataset

# sampled_dataset = paths2pairs_random_sample(transferred_dataset['train'], 2000, 0.1)


# # 模型部分

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PairPaths2Embed(nn.Module):
    def __init__(self, paths_types_list, types_feat_len_dict, hid_len, dropout):
        super().__init__()
        
        # 记录各path的各位数据对应的特征类型，方便进行映射
        self.paths_types_list = paths_types_list
        
        self.hid_len = hid_len
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hid_len)
        
        # 对原始特征进行映射
        self.feature_project = {}
        for feat_type in types_feat_len_dict:
            self.feature_project[feat_type] = nn.Linear(types_feat_len_dict[feat_type], hid_len)
            self.add_module('Feature_Projection_{}'.format(feat_type), self.feature_project[feat_type])
            
        # 生成各元路径的细分transformer
        self.path_level_embeds = []
        self.path_level_projects = []
        self.path_level_atts = []
        for path_index in range(len(paths_types_list)):
            # Path level embed
            encoder_layers = TransformerEncoderLayer(hid_len, 1, hid_len, dropout)
            path_level_embed = TransformerEncoder(encoder_layers, 1)
            self.path_level_embeds.append(path_level_embed)
            self.add_module('Path_Level_Embed_{}'.format(path_index), path_level_embed)
        
            # Path level project
            path_level_project = nn.Linear(len(paths_types_list[path_index])*hid_len, hid_len)
            self.path_level_projects.append(path_level_project)
            self.add_module('Path_Level_Project_{}'.format(path_index), path_level_project)
        
            # Path level attention
            path_level_att = nn.Linear(3*hid_len, 1)
            self.path_level_atts.append(path_level_att)
            self.add_module('Path_Level_Att_{}'.format(path_index), path_level_att)
        
        # Mutual Influence
        encoder_layers = TransformerEncoderLayer(hid_len, 1, hid_len, dropout)
        self.pair_level_embed = TransformerEncoder(encoder_layers, 1)
        
        # 最后的输出函数
        self.predict_layer = nn.Linear(hid_len*(2 + len(paths_types_list)), 1)
        self.predict_activation = nn.Sigmoid()
        
    def forward(self, sampled_dataset): 
        pair_count = sampled_dataset['label'].shape[0]
        
        start_node_embed = self.feature_project['Company_Node'](sampled_dataset['Company_Node'])
        start_node_embed = self.LayerNorm(start_node_embed)
        start_node_embed = self.activation(start_node_embed)
        start_node_embed = self.dropout(start_node_embed)
        
        end_node_embed = self.feature_project['Mobile_Node'](sampled_dataset['Mobile_Node'])
        end_node_embed = self.LayerNorm(end_node_embed)
        end_node_embed = self.activation(end_node_embed)
        end_node_embed = self.dropout(end_node_embed)
        
        # 依次处理各个路径的结果
        pair_paths_embed = []
        for path_index, path in enumerate(sampled_dataset['paths']):
            path_nodes_edges_embed = []
            
            # 检查该路径再本轮采样中是否有数据
            if len(path['data']) == 0:
                # 没有数据则直接创建对应维度的全零矩阵
                pair_paths_embed.append(torch.zeros(pair_count, self.hid_len).to(device_type))
                continue
                
            path_node_edge_types = self.paths_types_list[path_index]
            
            # 依次处理路径上的各点和边
            for node_edge_index in range(len(path_node_edge_types)):
                feat_type = path_node_edge_types[node_edge_index]
                
                node_edge_embed = self.feature_project[feat_type](path['data'][node_edge_index])
                node_edge_embed = self.LayerNorm(node_edge_embed)
                node_edge_embed = self.activation(node_edge_embed)
                node_edge_embed = self.dropout(node_edge_embed)
                
                path_nodes_edges_embed.append(node_edge_embed)
            
            path_nodes_edges_embed = torch.stack(path_nodes_edges_embed, 0)
            
            # path representation
            path_embed = self.path_level_embeds[path_index](path_nodes_edges_embed)
            batch_num = path_embed.size(1)
            path_embed = path_embed.permute(1, 0, 2).reshape(batch_num, -1)
            path_embed = self.path_level_projects[path_index](path_embed)
            path_embed = self.LayerNorm(path_embed)
            path_embed = self.activation(path_embed)
            path_embed = self.dropout(path_embed)
            
            # 基于path的起止点，加path本身的表征，给出重要性
            path_att = torch.cat([path_nodes_edges_embed[0], path_embed, path_nodes_edges_embed[-1]], 1)
            path_att = self.path_level_atts[path_index](path_att).squeeze()
            path_att = torch.tanh(path_att)
            
            # 获得pair对应的各个path
            pair_to_paths = path['path_to_pair'].clone()
            pair_to_paths[[0, 1]] = pair_to_paths[[1, 0]]

            # 将个path的重要性转化为稀疏矩阵
            attention = torch.sparse.FloatTensor(pair_to_paths, path_att, (pair_count, path_embed.size(0))).to(device_type)
            attention = torch.sparse.softmax(attention, dim=1)
            
            # 获得各pair对应embed
            pair_path_embed = torch.sparse.mm(attention, path_embed)
            
            pair_paths_embed.append(pair_path_embed)
        
        # Mutual Influence
        pair_paths_embed = torch.stack(pair_paths_embed, 0)
        pair_paths_embed = self.pair_level_embed(pair_paths_embed)
        pair_paths_embed = self.LayerNorm(pair_paths_embed)
        pair_paths_embed = self.activation(pair_paths_embed)
        pair_paths_embed = self.dropout(pair_paths_embed)
        
        # Prediction
        pair_num = pair_paths_embed.size(1)
        pair_paths_embed = pair_paths_embed.permute(1, 0, 2).reshape(pair_num, -1)
        pair_embed = torch.cat([start_node_embed, end_node_embed, pair_paths_embed], 1)
        
        prediction = self.predict_layer(pair_embed)
        prediction = self.predict_activation(prediction)
        
        return prediction


# # 模型配置信息

# In[ ]:


if torch.cuda.is_available():
    device_type = 'cuda'
else:
    device_type = 'cpu'
    
print('device_type:', device_type)

model_config = {}
model_config['epoch'] = 1000
model_config['round'] = 20
model_config['lr'] = 1e-5
model_config['weight_decay'] = 0
model_config['train_sample'] = 4000
model_config['positive_percent'] = 0
model_config['max_eval_sample'] = 30000
model_config['nhid'] = 512
model_config['dropout'] = 0.6
model_config['patience'] = 50

print(model_config)


# # 模型初始化

# In[ ]:


# 建立模型
model = PairPaths2Embed(transferred_dataset['paths_types'], transferred_dataset['typ_feat_len'], model_config['nhid'], 
                        model_config['dropout'])

model.to(device_type)

# 优化器
optimizer = optim.Adam(model.parameters(), lr = model_config['lr'], weight_decay = model_config['weight_decay'])

# 损失函数
BCE_loss = torch.nn.BCELoss()


# # 训练模型

# In[ ]:


from sklearn.metrics import roc_auc_score, average_precision_score

def perfomance_check(dataset):
    model.eval()
    with torch.no_grad(): 
        
        label_list = []
        predict_list = []
        
        for sample_start_index in range(0, len(dataset['label']), model_config['max_eval_sample']):
            sample_end_index = sample_start_index + model_config['max_eval_sample']
            if sample_end_index > len(dataset['label']):
                sample_end_index = len(dataset['label'])
        
            # 获得全部目标pair的id号
            sample_index = np.arange(sample_start_index, sample_end_index)
            
            sampled_dataset = paths2pairs_sample_with_index(dataset, sample_index)
            
            predict = model(sampled_dataset)
            predict = torch.squeeze(predict)
            
            label_list.append(sampled_dataset['label'])
            predict_list.append(predict)
            
        label = torch.cat(label_list)
        predict = torch.cat(predict_list)
        
        label_np = label.data.cpu().numpy()
        predict_np = predict.data.cpu().numpy()
        
        roc_auc = roc_auc_score(label_np, predict_np)
        average_precision = average_precision_score(label_np, predict_np)
        
        hit_k_rates = []
        for hit_k in [1, 3, 5]:
            # 将ID和概率及标签配对
            paired = np.column_stack((dataset['company_id'], predict_np, label_np))

            # 排序并提取前X个概率
            result = []
            hit_sum = 0
            for id_value in np.unique(dataset['company_id']):
                # 筛选特定ID
                id_group = paired[paired[:, 0] == id_value]
                # 按概率降序排序，并取前X个
                top_k_label = id_group[id_group[:, 1].argsort()][::-1][:hit_k]
                
                if np.sum(top_k_label[:, 2]) > 0:
                    hit_sum = hit_sum + 1
                
                result.append(top_k_label[:, 2])

            # 提取并展示结果
            result = np.concatenate(result)
            hit_k_rate = hit_sum / np.unique(dataset['company_id']).size
            hit_k_rates.append({hit_k: hit_k_rate})
            
        metric_result = {'roc_auc': roc_auc, 'avg_precision': average_precision, 'hit_k_rate': hit_k_rates}
        
        print(metric_result)
        
    return metric_result


# In[ ]:


from Utils.metrics import binary_problem_evaluate
from sklearn.metrics import roc_auc_score, average_precision_score
import time

patience = model_config['patience']
counter = 0
best_auc = 0

# 记录评价指标的变化
metric_results = {'train':[], 'val':[], 'test':[]}

# 模型参数的输出文件夹
localtime = time.strftime("%m-%d-%H:%M", time.localtime())
model_parameter_output_dir = Data_Output_path + '/Model_Parameter/PairPaths2Embed' + Data_Dir_Name + f'/{localtime}'
mkdir(Data_Output_path + '/Model_Parameter')
mkdir(Data_Output_path + '/Model_Parameter/PairPaths2Embed')
mkdir(Data_Output_path + '/Model_Parameter/PairPaths2Embed' + Data_Dir_Name)
mkdir(model_parameter_output_dir)

for epoch in range(model_config['epoch']):
    # 多少轮查看一次效果
    pbar = tqdm(range(model_config['round']))
    for sample_index in pbar:
        # 先采样
        sampled_dataset = paths2pairs_random_sample(transferred_dataset['train'], model_config['train_sample'], 
                                                    model_config['positive_percent'])
        
        # 再训练模型
        model.train()

        predict = model(sampled_dataset)
        predict = torch.squeeze(predict)
        
        loss = BCE_loss(predict, sampled_dataset['label'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 把数据转numpy，来算评价指标
        label_np = sampled_dataset['label'].data.cpu().numpy()
        predict_np = predict.data.cpu().numpy()
        
        roc_auc = roc_auc_score(label_np, predict_np)
        average_precision = average_precision_score(label_np, predict_np)
        pbar.set_postfix({'loss':loss.item(), 'roc_auc': roc_auc, 'avg_precision': average_precision})
    
    # 查看验证和测试集效果
    print('Epoch:', epoch)
    print('train')
    train_metric_result = perfomance_check(transferred_dataset['train'])
    metric_results['train'].append(train_metric_result)
    
    print('val')
    metric_result = perfomance_check(transferred_dataset['val'])
    metric_results['val'].append(metric_result)
    
    print('test')
    test_metric_result = perfomance_check(transferred_dataset['test'])
    metric_results['test'].append(test_metric_result)
    
    if best_auc < metric_result['roc_auc']:
        best_auc = metric_result['roc_auc']
        best_epoch = epoch
        counter = 0
        
        best_model_state_dict = model.state_dict()
        torch.save(best_model_state_dict, model_parameter_output_dir + f'/model_auc_{best_auc}.pt')
    else:
        counter += 1
    
    if counter >= patience:
        print('Early stopping')
        print('Best metric:')
        print(metric_results['train'][best_epoch])
        print(metric_results['val'][best_epoch])
        print(metric_results['test'][best_epoch])
        break


# # 显示效果变化图

# In[ ]:


metric_list = [x['roc_auc'] for x in metric_results['train']]
plt.plot(range(len(metric_list)), metric_list)

metric_list = [x['roc_auc'] for x in metric_results['val']]
plt.plot(range(len(metric_list)), metric_list)

metric_list = [x['roc_auc'] for x in metric_results['test']]
plt.plot(range(len(metric_list)), metric_list)

plt.show()


# In[ ]:




