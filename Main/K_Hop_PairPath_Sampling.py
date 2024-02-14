#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
Src_Dir_path = "../../.."
sys.path.append(Src_Dir_path)

# Self packages
from Utils.Pyspark_utils import ResilientSparkRunner
from Utils.Pyspark_utils import DEFAULT_STORAGE_LEVEL
from Utils.utils import read_json_config_file
from Utils.utils import mkdir
from Utils.utils import Log_save


# In[ ]:


#显示所有列
pd.set_option('display.max_columns', None)

#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

# 固定随机值
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
setup_seed(42)


# # 数据配置信息

# In[3]:


# 目标时间及月份(左闭右开)
Train_time_range = ["2022-10-01", "2023-06-01"]
Val_time_range = ["2023-06-01", "2023-08-01"]
Test_time_range = ["2023-08-01", "2023-10-01"]

All_aim_time_range_dict = {'Train': Train_time_range, 'Val': Val_time_range, 'Test': Test_time_range}


# In[5]:


from Utils.Pyspark_utils import Data_In_HDFS_Path
from Utils.utils import read_json_config_file

Data_Dir_Name = '/paths2pairs_0111'

# 数据存储位置
Data_Output_path = Src_Dir_path + "/.."
mkdir(Data_Output_path)

# 具体的图相关数据存储位置
Graph_Output_path = Data_Output_path + '/Graph'
mkdir(Graph_Output_path)

# 具体的本次数据的存储位置
Feature_Data_From_Online_Store_dir = Graph_Output_path + Data_Dir_Name
mkdir(Feature_Data_From_Online_Store_dir)

# 具体的时间区间
data_time_range_str = Train_time_range[0] + '_'+ str(Test_time_range[1])
Feature_Data_From_Online_Time_Store_dir = (Feature_Data_From_Online_Store_dir + '/' + data_time_range_str)
mkdir(Feature_Data_From_Online_Time_Store_dir)
print('目标时间数据本地存储文件夹', Feature_Data_From_Online_Time_Store_dir)

# HDFS Store dir
HDFS_Store_dir = Data_In_HDFS_Path + Data_Dir_Name + '/' + data_time_range_str
print('目标时间数据HDFS存储文件夹', HDFS_Store_dir)

Label_Data_Config_file = '../Config/Target_Node_Dataset_Config/KP_Task_2023_12_18.json'
Subgraph_Config_file = '../Config/K_Hop_Config/k_hop_config_for_paths2pairs_2023_12_30.json'
Node_Config_file = '../Config/Node_Config/Node_Config_Paths2Pairs_12_29.json'

Label_Data_Config_dict = read_json_config_file(Label_Data_Config_file)
Subgraph_Config_dict = read_json_config_file(Subgraph_Config_file)
Node_Config_dict = read_json_config_file(Node_Config_file)


# # 模型配置信息

# In[6]:


if torch.cuda.is_available():
    device_type = 'cuda'
else:
    device_type = 'cpu'
    
print('device_type:', device_type)


# # 读取数据

# In[7]:


# import importlib
# import Utils
# importlib.reload(Utils.K_hop_Path_Sampling)

from Utils.Target_Node_Dataloader import Read_Target_Node_with_Label
from Utils.K_hop_Path_Sampling import Pyspark_K_Hop_Path_Sampling, k_hop_path_pairing_to_sample
from Utils.Complex_Path_Basic_PySpark import complex_path_sampling

def get_train_sample_data(spark, Target_Node_Info_Dict, subgraph_hop_k = 4, max_main_path = 5):
    # 训练集对应信息存储位置
    Train_Data_Store_dir = HDFS_Store_dir + '/Train_K_Hop_Path'
    
    # 只取出训练集对应的信息
    tmp_time_range_start = All_aim_time_range_dict["Train"][0]
    tmp_time_range_end = All_aim_time_range_dict["Train"][1]
    tmp_time_range_limit = f"Source_Time >= '{tmp_time_range_start}' AND Source_Time < '{tmp_time_range_end}'"
    Pairs_for_train_df = Target_Node_Info_Dict['Data'].filter(tmp_time_range_limit)
    Pairs_for_train_df.persist(DEFAULT_STORAGE_LEVEL)
    
    # 获取训练集涉及的全部特征时间
    train_df_feature_times = [row['Feature_Time'] for row in Pairs_for_train_df.select("Feature_Time").distinct().collect()]
    train_df_feature_times.sort()
    
    # 将训练集中的公司节点为采样目标点
    K_Hop_Target_Info = {}
    K_Hop_Target_Info['Target_Node_df'] = Pairs_for_train_df.select(['Company_Node_UID', "Feature_Time"]).distinct()
    K_Hop_Target_Info['Target_Node_Type'] = "Company_Node"
    K_Hop_Target_Info['End_Node_Type'] = 'Mobile_Node'
    K_Hop_Target_Info['Target_Node_UID_name'] = "Company_Node_UID"
    K_Hop_Target_Info['Feature_Times'] = train_df_feature_times

    # 取出训练集中包含的公司-手机号组合对作为目标组合对
    Pair_Nodes_Info = {}
    Pair_Nodes_Info['Pair_Data'] = Pairs_for_train_df.select(["Company_Node_UID", "Mobile_Node_UID", "Feature_Time"]).distinct()
    Pair_Nodes_Info['Start_Node_Type'] = 'Company_Node'
    Pair_Nodes_Info['End_Node_Type'] = 'Mobile_Node'
    Pair_Nodes_Info['Start_Node_name'] = "Company_Node_UID"
    Pair_Nodes_Info['End_Node_name'] = "Mobile_Node_UID"
    Pair_Nodes_Info['Feature_Times'] = train_df_feature_times
    
    # 针对目标节点做k-hop采样
    Pyspark_K_Hop_Path_Sampling(spark, K_Hop_Target_Info, Subgraph_Config_dict, subgraph_hop_k, Train_Data_Store_dir, 
                                Pair_Nodes_Info)
    
    # 针对采样结果检测覆盖情况，给出主要路径、辅助路径，以及对应的样本
    train_data = k_hop_path_pairing_to_sample(spark, Train_Data_Store_dir, subgraph_hop_k, Pair_Nodes_Info, max_main_path)
    
    return train_data


# In[8]:


from pyspark.sql import DataFrame
from functools import reduce
from Utils.Pyspark_utils import hdfs_file_exists, hdfs_list_files, hdfs_read_txt_file, hdfs_read_marker_file
from Utils.K_hop_Path_Sampling import Pyspark_K_Hop_Path_Pairing
from pyspark.sql.functions import col

def get_val_test_sample_data(spark, Target_Node_Info_Dict, tain_main_paths, train_aux_paths):
    # 验证+测试集对应信息存储位置
    Val_Test_Data_Store_dir = HDFS_Store_dir + '/Val_Test'
    
    # 取出测试集+验证集对应的信息
    tmp_time_range_start = All_aim_time_range_dict["Val"][0]
    tmp_time_range_end = All_aim_time_range_dict["Test"][1]
    tmp_time_range_limit = f"Source_Time >= '{tmp_time_range_start}' AND Source_Time < '{tmp_time_range_end}'"
    Pairs_for_val_test_df = Target_Node_Info_Dict['Data'].filter(tmp_time_range_limit)
    Pairs_for_val_test_df.persist(DEFAULT_STORAGE_LEVEL)
    
    # 获取测试+验证集涉及的全部特征时间
    val_test_df_feature_times = [row['Feature_Time'] for row in Pairs_for_val_test_df.select("Feature_Time").distinct().collect()]
    val_test_df_feature_times.sort()
    
    # 获取测试+验证集涉及的全部目标公司点
    Path_Target_Nodes_df = Pairs_for_val_test_df.select(['Company_Node_UID', "Feature_Time"]).distinct()
    Path_Target_Nodes_df = Path_Target_Nodes_df.repartition('Company_Node_UID', "Feature_Time")
    Path_Target_Nodes_df = Path_Target_Nodes_df.withColumnRenamed('Company_Node_UID', 'Node_0')
    
    path_target_node = {}
    path_target_node['data'] = Path_Target_Nodes_df
    path_target_node['node_column'] = 'Node_0'
    
    # 取出训练集中包含的公司-手机号组合对作为目标组合对
    Pair_Nodes_Info = {}
    Pair_Nodes_Info['Pair_Data'] = Pairs_for_val_test_df.select(["Company_Node_UID", "Mobile_Node_UID", "Feature_Time"]).distinct()
    Pair_Nodes_Info['Start_Node_Type'] = 'Company_Node'
    Pair_Nodes_Info['End_Node_Type'] = 'Mobile_Node'
    Pair_Nodes_Info['Start_Node_name'] = "Company_Node_UID"
    Pair_Nodes_Info['End_Node_name'] = "Mobile_Node_UID"
    Pair_Nodes_Info['Feature_Times'] = val_test_df_feature_times
    
    ##########################################################################################################################
    # 基于主要路径采样样本
    test_val_main_paths = []
    for tain_main_path in tain_main_paths:
        main_path_name = tain_main_path['path_name']
        main_path_config = tain_main_path['path_config']
        main_path_result_dir = Val_Test_Data_Store_dir + f"/Sample/Main_Paths/{main_path_name}"
        
        # 结果记录字典
        test_val_main_path = {}
        test_val_main_path['path_name'] = main_path_name
        test_val_main_path['path_config'] = tain_main_path['path_config']
        
        # 读取该路径采样结果
        if not hdfs_file_exists(main_path_result_dir + "/Path_Data/_SUCCESS"):
            # 进行采样
            sample_result = complex_path_sampling(spark, main_path_name, main_path_config, val_test_df_feature_times, 
                                                         path_target_node)
            
            sample_result['data'].persist(DEFAULT_STORAGE_LEVEL)
            
            # 保存采样结果
            sample_result['data'].write.mode("overwrite").parquet(main_path_result_dir + "/Path_Data")
            
            test_val_main_path['data'] = sample_result['data']
        else:
            # 直接读取采样结果
            test_val_main_path['data'] = spark.read.parquet(main_path_result_dir + "/Path_Data")
        
        # 查看覆盖情况
        Pyspark_K_Hop_Path_Pairing(test_val_main_path, Pair_Nodes_Info, main_path_result_dir)
        
        test_val_main_paths.append(test_val_main_path)
    
    ##########################################################################################################################
    # 每个pair的全部标识列
    pair_id_columns = [Pair_Nodes_Info['Start_Node_name'], Pair_Nodes_Info['End_Node_name'], 'Feature_Time']
    
    # 合并各个main_path中的起止列获得全量样本
    test_val_sample_pairs_list = []
    for test_val_main_path in test_val_main_paths:
        path_id_columns = [test_val_main_path['path_config'][0]['head_node_std_name'], test_val_main_path['path_config'][-1]['tail_node_std_name'], 'Feature_Time']
        
        # 只保留该path中对应的id列，并去重
        test_val_sub_sample_pairs = test_val_main_path['data'].select(path_id_columns).distinct()
        
        # 修正id列列名
        test_val_sub_sample_pairs = test_val_sub_sample_pairs.withColumnRenamed(path_id_columns[0], pair_id_columns[0])
        test_val_sub_sample_pairs = test_val_sub_sample_pairs.withColumnRenamed(path_id_columns[1], pair_id_columns[1])
        
        test_val_sample_pairs_list.append(test_val_sub_sample_pairs)

    # 合并各个样本，并去重
    test_val_sample_pairs = reduce(DataFrame.union, test_val_sample_pairs_list).distinct()
    
    # 将目标pair能对应上的pair标记为正样本，其余标记为负样本
    test_val_sample_pairs = test_val_sample_pairs.alias("test_val_sample_pairs")
    target_pair_df = Pair_Nodes_Info['Pair_Data'].alias("target_pair_df")
    join_conditions = [col(f"test_val_sample_pairs.{pair_id_columns[i]}") == col(f"target_pair_df.{pair_id_columns[i]}") for i in range(len(pair_id_columns))]
    selected_columns = [col(f"test_val_sample_pairs.{col_name}") for col_name in test_val_sample_pairs.columns]
    
    test_val_sample_pairs = test_val_sample_pairs.join(target_pair_df, join_conditions, "left")
    join_check_column = pair_id_columns[0]
    test_val_sample_pairs = test_val_sample_pairs.withColumn(f"Label", col(f"target_pair_df.{join_check_column}").isNotNull())
    test_val_sample_pairs = test_val_sample_pairs.select(selected_columns + ["Label"])
    test_val_sample_pairs.persist(DEFAULT_STORAGE_LEVEL)
    
    # 输出正负样本数目
    positive_sample_count = test_val_sample_pairs.filter(test_val_sample_pairs["Label"] == 1).count()
    negative_sample_count = test_val_sample_pairs.filter(test_val_sample_pairs["Label"] == 0).count()
    print(f'正样本数目:{positive_sample_count}, 负样本数目:{negative_sample_count}')
    ##########################################################################################################################
    # 基于辅助路径采样符合样本要求的路径
    test_val_aux_paths = []
    for train_aux_path in train_aux_paths:
        aux_path_name = train_aux_path['path_name']
        aux_path_config = train_aux_path['path_config']
        aux_path_result_dir = Val_Test_Data_Store_dir + f"/Sample/Auxiliary_Paths/{aux_path_name}"
        
        # 结果记录字典
        test_val_aux_path = {}
        test_val_aux_path['path_name'] = aux_path_name
        test_val_aux_path['path_config'] = aux_path_config
        
        # 读取该路径采样结果
        if not hdfs_file_exists(aux_path_result_dir + "/Path_Data/_SUCCESS"):
            print('开始采样复杂路:', aux_path_config)
            
            # 进行采样
            sample_result = complex_path_sampling(spark, aux_path_name, aux_path_config, val_test_df_feature_times, 
                                                         path_target_node)
            sample_path_df = sample_result['data']
            
            # 该路径的目标pair的标识列
            path_id_columns = [train_aux_path['path_config'][0]['head_node_std_name'], train_aux_path['path_config'][-1]['tail_node_std_name'], 'Feature_Time']
            
            sample_path_df = sample_path_df.alias("sample_path_df")
            test_val_sample_pairs = test_val_sample_pairs.alias("test_val_sample_pairs")
            
            # 只保留能和样本匹配上的路径，以及路径本身的列
            join_conditions = [col(f"sample_path_df.{path_id_columns[i]}") == col(f"test_val_sample_pairs.{pair_id_columns[i]}") for i in range(len(path_id_columns))]
            selected_columns = [col(f"sample_path_df.{col_name}") for col_name in sample_path_df.columns]
            sample_path_df = sample_path_df.join(test_val_sample_pairs, join_conditions, "inner")
            sample_path_df = sample_path_df.select(selected_columns)
            
            sample_path_df.persist(DEFAULT_STORAGE_LEVEL)
            
            # 保存采样结果
            sample_path_df.write.mode("overwrite").parquet(aux_path_result_dir + "/Path_Data")
            
            test_val_aux_path['data'] = sample_path_df
        else:
            # 直接读取采样结果
            test_val_aux_path['data'] = spark.read.parquet(aux_path_result_dir + "/Path_Data")
            print(f'已处理好复杂路{aux_path_name}')
        
        # 查看覆盖情况
        Pyspark_K_Hop_Path_Pairing(test_val_aux_path, Pair_Nodes_Info, aux_path_result_dir)
        
        test_val_aux_paths.append(test_val_aux_path)
    
    ##########################################################################################################################
    # 基于样本的source_time分割验证集和测试集(因为公司的source_time是一定的)
    # test_val_sample_pairs, test_val_main_paths, test_val_aux_paths
    val_data = {}
    test_data = {}
    
    # 时间限制
    val_time_range_start = All_aim_time_range_dict["Val"][0]
    val_time_range_end = All_aim_time_range_dict["Val"][1]
    val_time_range_limit = f"Source_Time >= '{val_time_range_start}' AND Source_Time < '{val_time_range_end}'"
    
    test_time_range_start = All_aim_time_range_dict["Test"][0]
    test_time_range_end = All_aim_time_range_dict["Test"][1]
    test_time_range_limit = f"Source_Time >= '{test_time_range_start}' AND Source_Time < '{test_time_range_end}'"
    
    # 获得各个pair通过公司名和Feature_Time区分来源时间的映射表
    pairs_src_time = Pairs_for_val_test_df.select(["Company_Node_UID", "Feature_Time", "Source_Time"]).distinct()
    pair_src_time_id_columns = ["Company_Node_UID", "Feature_Time"]
    pairs_src_time = pairs_src_time.alias("pairs_src_time")
    
    # 基于company的source时间给全部样本标记上source_time
    test_val_sample_pairs = test_val_sample_pairs.join(pairs_src_time, on = ["Company_Node_UID", "Feature_Time"], how = 'inner')
    
    # 分割为验证集和测试集的样本
    val_data['sample_pairs'] = test_val_sample_pairs.filter(val_time_range_limit)
    test_data['sample_pairs'] = test_val_sample_pairs.filter(test_time_range_limit)
    
    # 分割各个path到test和val
    val_data['main_paths'] = []
    test_data['main_paths'] = []
    val_data['aux_paths'] = []
    test_data['aux_paths'] = []
    for path_index, test_val_path in enumerate(test_val_main_paths + test_val_aux_paths):
        # 该路径的目标pair的标识列
        path_id_columns = [test_val_path['path_config'][0]['head_node_std_name'], 'Feature_Time']

        # 为path数据添加src_time
        test_val_path_df = test_val_path['data'].alias("test_val_path_df")
        join_conditions = [col(f"test_val_path_df.{path_id_columns[i]}") == col(f"pairs_src_time.{pair_src_time_id_columns[i]}") for i in range(len(path_id_columns))]
        selected_columns = [col(f"test_val_path_df.{col_name}") for col_name in test_val_path_df.columns]
        test_val_path_df = test_val_path_df.join(pairs_src_time, join_conditions, "inner")
        
        # 基于src_time分割path
        val_path_df = test_val_path_df.filter(val_time_range_limit)
        test_path_df = test_val_path_df.filter(test_time_range_limit)
    
        # 只保留path的原本列
        val_path_df = val_path_df.select(selected_columns)
        test_path_df = test_path_df.select(selected_columns)
        
        val_path = test_val_path.copy()
        val_path['data'] = val_path_df
        
        test_path = test_val_path.copy()
        test_path['data'] = test_path_df
        
        # 保存结果
        if path_index < len(test_val_main_paths):
            val_data['main_paths'].append(val_path)
            test_data['main_paths'].append(test_path)
        else:
            val_data['aux_paths'].append(val_path)
            test_data['aux_paths'].append(test_path)
            
    return val_data, test_data


# In[9]:


# import importlib
# import Utils
# importlib.reload(Utils.Complex_Path_Basic_PySpark)

from Utils.Complex_Path_Basic_PySpark import read_list_node_tables

def get_nodes_feature(spark, dataset, node_times, regenerate = False):
    """
    合并各path的各种类型的节点，取出对应的特征
    """
    
    # 节点存储位置
    node_id_result_dir = HDFS_Store_dir + '/Node_ID'
    
    # 节点特征存储位置
    node_feature_result_dir = HDFS_Store_dir + '/Node_Feature_debug'
    
    # 合并各类型节点
    node_ids = {}
    for data_type in dataset:
        for path in (dataset[data_type]['main_paths'] + dataset[data_type]['aux_paths']):
            path_df = path['data']
            print(data_type, '处理路径中的点:', path['path_config'])
            
            ################################################################################################
            # 先处理起点
            node_type = path['path_config'][0]["head_node_type"]
            node_column_name = path['path_config'][0]["head_node_std_name"]

            # 取出对应列，去重并修改列名
            node_id_df = path_df.select([node_column_name, 'Feature_Time']).distinct().withColumnRenamed(node_column_name, node_type)

            if node_type not in node_ids:
                node_ids[node_type] = [node_id_df]
            else:
                node_ids[node_type].append(node_id_df)
            
            ################################################################################################
            # 再依次处理其余点
            for column_i in range(len(path['path_config'])):
                node_type = path['path_config'][column_i]["tail_node_type"]
                node_column_name = path['path_config'][column_i]["tail_node_std_name"]
                
                # 取出对应列，去重并修改列名
                node_id_df = path_df.select([node_column_name, 'Feature_Time']).distinct().withColumnRenamed(node_column_name, node_type)
                
                if node_type not in node_ids:
                    node_ids[node_type] = [node_id_df]
                else:
                    node_ids[node_type].append(node_id_df)
    
    # 获取各节点对应的特征
    node_features = {}
    for node_type in node_ids:
        # *注意这里强制进行了重新生成，记得修改
        if not hdfs_file_exists(node_feature_result_dir + f"/{node_type}/_SUCCESS"):
            # 先合并各类型的node id并去重
            node_ids_df = reduce(DataFrame.union, node_ids[node_type]).distinct().persist(DEFAULT_STORAGE_LEVEL)
            
            node_ids_df.write.mode("overwrite").parquet(node_id_result_dir + f"/{node_type}")
            
            target_node = {}
            target_node['data'] = node_ids_df
            target_node['node_column'] = node_type
            
            print(Node_Config_dict[node_type])
            
            # 再抽取对应的特征
            node_table = read_list_node_tables(spark, node_type, Node_Config_dict[node_type], node_times, target_node = target_node)

            # 保存结果
            node_table['data'].write.mode("overwrite").parquet(node_feature_result_dir + f"/{node_type}")
            
            node_features[node_type] = node_table['data']
        else:
            node_features[node_type] = spark.read.parquet(node_feature_result_dir + f"/{node_type}")
            
    return node_features


# In[10]:


def read_dataset_into_memory(dataset):
    dataset_in_memory = {}
    
    dataset_local_dir = Feature_Data_From_Online_Time_Store_dir + '/Data_Prepare'
    mkdir(dataset_local_dir)
    
    # 依次处理各数据集
    for data_type in ['train', 'val', 'test']:
        # 本地结果保存位置
        result_dir = dataset_local_dir + f'/{data_type}'
        mkdir(result_dir)
        
        dataset_in_memory[data_type] = {}
        
        if not os.path.exists(result_dir + '/sample_pairs.pkl'):
            # 将样本读取为pd
            sample_pd = dataset[data_type]['sample_pairs'].toPandas()

            sample_pd.to_pickle(result_dir + '/sample_pairs.pkl')
        
        # 将各个path的数据转化为pd
        dataset_in_memory[data_type]['main_paths'] = []
        dataset_in_memory[data_type]['aux_paths'] = []
        mkdir(result_dir + "/main_paths")
        mkdir(result_dir + "/aux_paths")
        for path_index, path in enumerate(dataset[data_type]['main_paths'] + dataset[data_type]['aux_paths']):
            print(path['path_name'])
            
            # 确定是主路径还是辅助路径
            if path_index < len(dataset[data_type]['main_paths']):
                path_type = 'main_paths'
            else:
                path_type = 'aux_paths'
                
            if not os.path.exists(result_dir + f"/{path_type}/{path['path_name']}.pkl"):
                # 把路径的配置信息也保存到本地
                with open(result_dir + f"/{path_type}/{path['path_name']}.json", "w", encoding="utf-8") as file:
                    json.dump(path['path_config'], file, ensure_ascii=False, indent=4)
                
                # 将各路径读取为pd
                path_pd = path['data'].toPandas()
                
                path_pd.to_pickle(result_dir + f"/{path_type}/{path['path_name']}.pkl")

            
    # 将点的特征读取为pd
    dataset_in_memory['node_feature'] = {}
    mkdir(dataset_local_dir + "/node_feature_new")
    for node_type in dataset['node_feature']:
        print(node_type)
        
        if not os.path.exists(dataset_local_dir + f"/node_feature_new/{node_type}.pkl"):
            node_pd = dataset['node_feature'][node_type].toPandas()

            node_pd.to_pickle(dataset_local_dir + f"/node_feature_new/{node_type}.pkl")

        dataset_in_memory['node_feature'][node_type] = node_pd
            
    return dataset_in_memory


# In[11]:


def paths2pairs_data_prepare(spark, subgraph_hop_k = 4, max_main_path = 3):
    # 要获取的数据集，每组数据分别存储正负样本对应的节点对，以及每种路径下样本能匹配上的路径，以及各路径上涉及到的全部点对应的特征
    dataset = {}
    
    # 读取目标节点信息
    Target_Node_Info_Dict = Read_Target_Node_with_Label(spark, Label_Data_Config_dict, All_aim_time_range_dict, HDFS_Store_dir)
    
    ##########################################################################################################################
    # 获取训练数据
    train_data = get_train_sample_data(spark, Target_Node_Info_Dict, subgraph_hop_k, max_main_path)
    dataset['train'] = train_data
    
    ##########################################################################################################################
    # 基于训练数据的采样结果获得验证及测试数据
    val_data, test_data = get_val_test_sample_data(spark, Target_Node_Info_Dict, train_data['main_paths'], train_data['aux_paths'])
    dataset['val'] = val_data
    dataset['test'] = test_data

    ##########################################################################################################################
    # 获取训练、验证、测试的各路径上涉及的全部点对应的特征
    dataset['node_feature'] = get_nodes_feature(spark, dataset, Target_Node_Info_Dict['Feature_Times'])
    
    ##########################################################################################################################
    # 将全部数据都导入本地内存
    read_dataset_into_memory(dataset)
    
    return


# In[12]:


Spark_Runner = ResilientSparkRunner()
Spark_Runner.run(paths2pairs_data_prepare)


# In[ ]:




