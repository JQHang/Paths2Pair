# 加入Aim_Metapath_list的选项（如果有值，且不为空就以此为准）
# 分别保存各组groupby后的结果
# 加入只保留部分有效特征的功能（用config）

import numpy as np
import math
import pandas as pd
import os
import gc
import torch
import json

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

from Utils.utils import mkdir

from pyspark.sql.types import *
from pyspark.sql.functions import broadcast

from .Get_Aggregated_Data_From_Online import Get_Proc_Feature_Data_From_Online

"""
作用：
    从线上读取聚合后的pandas数据（包含标签列 + 处理后的元路径特征列）。

输入：

    
返回值：
    输出结果直接存储在对应文件夹中
"""
def Get_ComplexPath_Data_From_Online(Spark_Session, Aim_UID_info_dict, Feature_Data_From_Online_Config_dict, 
                         Feature_Data_From_Online_Time_Store_dir, regenerate = False, Meta_path_drop_list = [], 
                         Meta_path_Column_drop_dict = {}, Preprocess_Type_List = ['Norm', 'Std'], sample_start = None, 
                         sample_end = None, Data_Output_Type = 'Pandas', Return_Data = True):
    tmp_aggregate_data_start_time = datetime.now()
    
    # 要输出的字典
    Result_Data_dict = {}
    
    # 各类型节点对应的特征长度
    Result_Data_dict['Node_Type_to_Feature_len'] = {}
    
    # 起始点本身特征
    Result_Data_dict['Start_Node_Feature_List'] = []
    
    # 各元路径的特征
    Result_Data_dict['Meta_Path_Feature'] = {}
    
    Result_Data_dict['Meta_Path_Column_Type'] = {}
    
    Result_Data_dict['Meta_Path_Feature_Column_List'] = {}
    
    Processed_Feature_List = []
    Processed_Feature_Summary_List = []
    
    ###########################################################################################################################
    # 读取目标点相关信息
    tmp_aim_entity_rdd = Aim_UID_info_dict['Data']
    Aim_Node_type = Aim_UID_info_dict['Node_Type']
    Aim_Node_UID_Name = Aim_Node_type + '_UID'

    # 只保留rdd中的目标节点列，去缩减特征表
    tmp_aim_node_UID_rdd = tmp_aim_entity_rdd.select([Aim_Node_UID_Name, 'Feature_Time'])
    
    tmp_aim_node_UID_rdd = tmp_aim_node_UID_rdd.persist()
    
#     # broadcast目标列，加快运算速度
#     tmp_aim_node_UID_rdd_Broadcast = broadcast(tmp_aim_node_UID_rdd)
    
    # 一定得有对应的pandas文件
    if 'Data_pd' not in Aim_UID_info_dict:
        print('Error:没有目标点的pandas格式文件')
        return
    
    # 保存目标点的顺序
    tmp_aim_entity_pd = Aim_UID_info_dict['Data_pd']

    # 如果有设定的范围，则只保留指定范围行内的标签
    if sample_start != None:
        print('只保留', sample_start, sample_end, '范围内的点和特征')
        tmp_aim_entity_pd = tmp_aim_entity_pd.iloc[sample_start:sample_end, :]

    # 保留UID列和特征时间列，用于拼接特征
    tmp_aim_node_UID_order_pd = tmp_aim_entity_pd[[Aim_Node_UID_Name, 'Feature_Time']].copy()
    
    # 如果有标签列，则保存标签列
    if 'Label' in tmp_aim_entity_pd.columns:
        if Data_Output_Type == 'Pandas':
            Result_Data_dict['Label'] = tmp_aim_entity_pd['Label']
        elif Data_Output_Type == 'Torch':
            Result_Data_dict['Label'] = torch.FloatTensor(tmp_aim_entity_pd['Label'].values)
            
        print('Label Shape:', tmp_aim_entity_pd['Label'].shape)
    
    # 获取配置中的全部元路径
    tmp_meta_path_name_list = list(Feature_Data_From_Online_Config_dict['Meta_Path_Feature_Table_List'].keys())

    # 保证处理顺序一致
    tmp_meta_path_name_list.sort()

    print('要处理的元路径为:', tmp_meta_path_name_list)
    
    tmp_Aim_Feature_Table_dt_list = list(set(tmp_aim_node_UID_order_pd['Feature_Time']))
    tmp_Aim_Feature_Table_dt_list.sort()
    
    print('全部目标特征时间为:', tmp_Aim_Feature_Table_dt_list)
    ###########################################################################################################################
    print('----------------------------------------------------------------------------')
    print('先处理起始点本身特征')

    tmp_start_node_start_time = datetime.now()
    
    tmp_aim_node_feature_pandas_all = Get_Proc_Feature_Data_From_Online(Spark_Session, 
                                                  Feature_Data_From_Online_Config_dict['Start_Node_Feature_Table_List'], 
                                                  regenerate, tmp_Aim_Feature_Table_dt_list, Aim_Node_type, 
                                                  tmp_aim_node_UID_rdd, tmp_aim_node_UID_order_pd, 
                                                  Feature_Data_From_Online_Time_Store_dir, 
                                                  Preprocess_Type_List, 'Start_Node', Return_Data = Return_Data)
    
    tmp_start_node_end_time = datetime.now()
   
    print('处理起始点总共花费时间:', (tmp_start_node_end_time - tmp_start_node_start_time))
    
    # 根据需求保存结果
    if Return_Data:
        print(Aim_Node_type + '目标节点涉及的全部特征数:', tmp_aim_node_feature_pandas_all.shape[1])
            
        if Data_Output_Type == 'Pandas':
            Result_Data_dict['Start_Node_Feature_List'].append(tmp_aim_node_feature_pandas_all)
        elif Data_Output_Type == 'Torch':
            Result_Data_dict['Start_Node_Feature_List'].append(torch.FloatTensor(tmp_aim_node_feature_pandas_all.values))

        # 记录特征长度
        Result_Data_dict['Node_Type_to_Feature_len']['Start_Node'] = tmp_aim_node_feature_pandas_all.shape[1]
    ###########################################################################################################################
    print('----------------------------------------------------------------------------')
    print('再按如下顺序开始处理元路径:', tmp_meta_path_name_list)

    # 依次读取各元路径对应的数据
    for tmp_meta_path_name in tmp_meta_path_name_list:
        tmp_meta_path_start_time = datetime.now()
        
        # 跳过不需要的元路径
        if tmp_meta_path_name in Meta_path_drop_list:
            print('跳过元路径:', tmp_meta_path_name)
            continue

        print('处理元路径:', tmp_meta_path_name)
        
        # 初始化各类信息
        Result_Data_dict['Meta_Path_Feature'][tmp_meta_path_name] = {}
        Result_Data_dict['Meta_Path_Column_Type'][tmp_meta_path_name] = {}
        Result_Data_dict['Meta_Path_Feature_Column_List'][tmp_meta_path_name] = {}
        
        # 依次处理同一元路径对应的复杂路
        tmp_complex_path_info_list = Feature_Data_From_Online_Config_dict['Meta_Path_Feature_Table_List'][tmp_meta_path_name]
        for tmp_complex_path_name in tmp_complex_path_info_list:
            print('处理复杂路:', tmp_complex_path_name)
            
            # 初始化复杂路对应的特征
            Result_Data_dict['Meta_Path_Feature'][tmp_meta_path_name][tmp_complex_path_name] = {}
            
            # 获得对应元路径的配置信息
            tmp_complex_path_info = tmp_complex_path_info_list[tmp_complex_path_name]
            
            # 依次处理元路径中各列的信息
            for tmp_column_i in range(len(tmp_complex_path_info)):
                if tmp_meta_path_name in Meta_path_Column_drop_dict and tmp_column_i in Meta_path_Column_drop_dict[tmp_meta_path_name]:
                    print('跳过复杂路:', tmp_complex_path_name, '的第', tmp_column_i, '列')
                    continue

                tmp_meta_path_column_start_time = datetime.now()

                # 获取对应列节点类型
                tmp_column_node_class = tmp_complex_path_info[tmp_column_i]["Node_class"]

#                 # 初始化该列的特征的存储
#                 Result_Data_dict['Meta_Path_Feature'][tmp_meta_path_name][tmp_column_i] = []

                # 保留节点类型信息
                Result_Data_dict['Meta_Path_Column_Type'][tmp_meta_path_name][tmp_column_i] = tmp_column_node_class

                # 获取对这类型特征的说明
                tmp_node_comment = tmp_complex_path_name + '_' + str(tmp_column_i) + '_' + tmp_column_node_class

                # 将特征表信息转化为指定格式
                tmp_Feature_Table_Config_list = []
                for tmp_aim_feature_table_i in range(len(tmp_complex_path_info[tmp_column_i]["Feature_Table_List"])):
                    tmp_feature_table_name = tmp_complex_path_info[tmp_column_i]["Feature_Table_List"][tmp_aim_feature_table_i]
                    tmp_feature_summary_table_name = tmp_complex_path_info[tmp_column_i]["Feature_Table_Summary_List"][tmp_aim_feature_table_i]
                    tmp_aim_column_name = 'Start_Column'

                    tmp_Feature_Table_Config_list.append({"Table_Name":tmp_feature_table_name, 
                                              "Summary_Table_Name": tmp_feature_summary_table_name,
                                              "UID": tmp_aim_column_name})

                # 获取目标数据
                tmp_aim_node_feature_pandas_all = Get_Proc_Feature_Data_From_Online(Spark_Session, tmp_Feature_Table_Config_list, regenerate, 
                                                              tmp_Aim_Feature_Table_dt_list, Aim_Node_type,
                                                              tmp_aim_node_UID_rdd,
                                                              tmp_aim_node_UID_order_pd, 
                                                              Feature_Data_From_Online_Time_Store_dir, 
                                                              Preprocess_Type_List, tmp_node_comment, 
                                                              Return_Data = Return_Data)

                # 根据需求保存结果
                if Return_Data:
                    if Data_Output_Type == 'Pandas':
                        Result_Data_dict['Meta_Path_Feature'][tmp_meta_path_name][tmp_complex_path_name][tmp_column_i] = tmp_aim_node_feature_pandas_all
                    elif Data_Output_Type == 'Torch':
                        tmp_aim_node_feature_torch_all = torch.FloatTensor(tmp_aim_node_feature_pandas_all.values)
                        Result_Data_dict['Meta_Path_Feature'][tmp_meta_path_name][tmp_complex_path_name][tmp_column_i] = tmp_aim_node_feature_torch_all

                    # 记录节点对应的特征长度
                    Result_Data_dict['Node_Type_to_Feature_len'][tmp_column_node_class] = tmp_aim_node_feature_pandas_all.shape[1]

                    Result_Data_dict['Meta_Path_Feature_Column_List'][tmp_meta_path_name][tmp_column_i] = list(tmp_aim_node_feature_pandas_all.columns)
                
                
                tmp_meta_path_column_end_time = datetime.now()

                print('完成对元路径', tmp_meta_path_name, '的第', tmp_column_i, '列的结果的生成，共花费时间:', 
                     (tmp_meta_path_column_end_time - tmp_meta_path_column_start_time))
                print('----------------------------------------------------------------------------')
        
        tmp_meta_path_end_time = datetime.now()
        print('完成对元路径', tmp_meta_path_name, '的全部列的结果的生成，共花费时间:', (tmp_meta_path_end_time - tmp_meta_path_start_time))
        print('----------------------------------------------------------------------------')
    
    tmp_aggregate_data_end_time = datetime.now()
    
    print('完成全部数据生成，总共花费时间:', (tmp_aggregate_data_end_time - tmp_aggregate_data_start_time))
    print('----------------------------------------------------------------------------')
    
    return Result_Data_dict

"""
作用：
    为机器学习模型按指定格式合并特征矩阵

输入：

    
返回值：
    输出结果直接存储在对应文件夹中
"""
def Concat_ComplexPath_Feature_For_ML_Model(Result_Data_dict, delete_original_data = True):
    tmp_feature_to_concat_list = []
    
    tmp_add_feature_pd = Result_Data_dict['Start_Node_Feature_List'][0]

    # 给特征列名加上注释
    tmp_add_feature_pd.columns = tmp_add_feature_pd.columns + '__Start_Node'

    tmp_feature_to_concat_list.append(tmp_add_feature_pd)

    if delete_original_data == True:
        del Result_Data_dict['Start_Node_Feature_List']
        gc.collect()
    
    for tmp_meta_path_name in Result_Data_dict['Meta_Path_Feature']:
        for tmp_complex_path_name in Result_Data_dict['Meta_Path_Feature'][tmp_meta_path_name]:
            tmp_column_i_list = list(Result_Data_dict['Meta_Path_Feature'][tmp_meta_path_name][tmp_complex_path_name].keys())
            for tmp_column_i in tmp_column_i_list:
                # 只读取最近一个月的数据
                tmp_add_feature_pd = Result_Data_dict['Meta_Path_Feature'][tmp_meta_path_name][tmp_complex_path_name][tmp_column_i]

                # 给特征名加上元路径及列序号
                tmp_add_feature_pd.columns = tmp_add_feature_pd.columns + '__' + tmp_complex_path_name + '__' + str(tmp_column_i)

                tmp_feature_to_concat_list.append(tmp_add_feature_pd)

                if delete_original_data == True:
                    del Result_Data_dict['Meta_Path_Feature'][tmp_meta_path_name][tmp_complex_path_name][tmp_column_i]
                    gc.collect()
    
    Result_Data_dict['Feature'] = pd.concat(tmp_feature_to_concat_list, axis = 1)
    
    return Result_Data_dict