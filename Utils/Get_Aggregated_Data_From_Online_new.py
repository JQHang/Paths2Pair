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


"""
作用：
    从线上读取指定目标点对应的特征信息和各特征的总结信息，并进行预处理后返回指定结果

输入：
    Feature_Table_Config_list:目标特征表的配置信息
    tmp_Aim_Feature_Table_dt_list：目标时间范围
    
返回值：
    目标节点的特征值
"""
def Get_Node_Feature_Data_From_Online(Spark_Session, Feature_Table_Config_list, regenerate, tmp_Aim_Feature_Table_dt_list, Aim_Node_type, 
                          tmp_aim_node_UID_rdd_Broadcast, tmp_aim_node_UID_order_pd, Feature_Data_From_Online_Time_Store_dir, 
                          Preprocess_Type_List, Node_Comment, eps = 1e-05, Return_Data = True):
    # 设置各类型特征的保存位置
    Aim_Node_Raw_Feature_file = (Feature_Data_From_Online_Time_Store_dir + Node_Comment + '_Raw.parquest')
    Aim_Node_Norm_Feature_file = (Feature_Data_From_Online_Time_Store_dir + Node_Comment + '_Norm.parquest')
    Aim_Node_Std_Feature_file = (Feature_Data_From_Online_Time_Store_dir + Node_Comment + '_Std.parquest')
        
    if regenerate or not os.path.exists(Aim_Node_Std_Feature_file):
        print('开始生成目标点本身特征')
        
        # 记录各个时间点聚合后的特征
        tmp_aim_node_feature_rdd_dict = {}

        # 记录各个时间点特征表的统计信息
        tmp_feature_table_summary_pd_dict = {}

        for tmp_feature_table_Info_dict in Feature_Table_Config_list:
            tmp_feature_table_name = tmp_feature_table_Info_dict['Table_Name']
            tmp_aim_column_name = tmp_feature_table_Info_dict['UID']

            print('读取特征表:', tmp_feature_table_name)

            tmp_sql_command = """
                        SELECT
                            *
                        FROM
                            """ + tmp_feature_table_name + """
                        WHERE 
                            """ + tmp_aim_column_name +""" IS NOT NULL AND
                            dt = '""" + tmp_Aim_Feature_Table_dt_list[0] + """'"""

            # Extend Limit if have more possible time
            if len(tmp_Aim_Feature_Table_dt_list) > 0:
                for tmp_Aim_Feature_Table_dt in tmp_Aim_Feature_Table_dt_list[1:]:
                    tmp_sql_command = tmp_sql_command + """\nOR dt = '""" + tmp_Aim_Feature_Table_dt + """'"""
            
            # Get Result DataFrame
            tmp_feature_table_rdd = Spark_Session.sql(tmp_sql_command)
            
            # Change aim node ID column name to tmp_node_class + '_UID'
            tmp_feature_table_rdd = tmp_feature_table_rdd.withColumnRenamed(tmp_aim_column_name, Aim_Node_type + '_UID')
            
            # 获取特征表的格式信息
            tmp_feature_table_rdd_json = json.loads(tmp_feature_table_rdd.schema.json())['fields']

            # 记录有效的特征列名
            tmp_useful_feature_cols_list = []

            # 只保留其中有效的列（entity_id加数值类型的列，*待优化，数据格式标准化后改为entity_id及其之后的列）
            for tmp_col_info in tmp_feature_table_rdd_json:
                col = tmp_col_info['name']
                col_type = tmp_col_info['type']

                if col == (Aim_Node_type + '_UID'):
                    continue

                if 'entity_id' in col:
                    continue

                if col_type in ['int', 'integer', 'float', 'bigint','double', 'long']:

                    tmp_useful_feature_cols_list.append(col)

                elif col_type != 'string':
                    print('-----------------------------------------------------------')
                    print('WARNING:stange_type:', col, col_type)
                    print('-----------------------------------------------------------')

            # 只保留目标节点、目标特征和时间
            tmp_feature_table_rdd = tmp_feature_table_rdd.select([Aim_Node_type + '_UID'] + tmp_useful_feature_cols_list + ['dt'])

            print('Useful feature number in Table' , tmp_feature_table_name, 'is:', len(tmp_useful_feature_cols_list))
            ############################################################################################################
            # Preprocessing: Normalize and Standardize
            tmp_time_processed_feature_table_list = []
            
            for tmp_Aim_Feature_Table_dt in tmp_Aim_Feature_Table_dt_list:
                print('处理时间:', tmp_Aim_Feature_Table_dt)

                # 只保留对应时间的信息
                tmp_time_feature_table_rdd = tmp_feature_table_rdd.where("dt = '" + tmp_Aim_Feature_Table_dt + "'")

                # 删除dt列
                tmp_time_feature_table_rdd = tmp_time_feature_table_rdd.drop('dt')

                # 通过persist保留计算结果
                tmp_time_feature_table_rdd = tmp_time_feature_table_rdd.persist()

                # 查看对应时间内的数据是否有重复
                tmp_time_feature_table_rdd_raw_count = tmp_time_feature_table_rdd.count()

                if tmp_time_feature_table_rdd_raw_count == 0:
                    print('Error: 特征表', tmp_feature_table_name, '在时间', tmp_Aim_Feature_Table_dt, '的数据为空，得及时处理')
                else:
                    # 对特征表去重（理论上不需要，但防止有不符合规范的表）
                    tmp_time_feature_table_rdd = tmp_time_feature_table_rdd.dropDuplicates([Aim_Node_type + '_UID'])

                    # 通过persist保留计算结果
                    tmp_time_feature_table_rdd = tmp_time_feature_table_rdd.persist()

                    tmp_time_feature_table_rdd_count = tmp_time_feature_table_rdd.count()

                    # 检查数据量是否保持一致
                    if tmp_time_feature_table_rdd_raw_count > tmp_time_feature_table_rdd_count:
                        print('Error: 特征表', tmp_feature_table_name, '在时间', tmp_Aim_Feature_Table_dt, 
                            '的数据中有重复UID，得及时修改, 目前先保留第一条信息，原始行数为:', 
                             tmp_time_feature_table_rdd_raw_count,
                             '去重后为:', tmp_time_feature_table_rdd_count)
            
            
            ############################################################################################################
            # Union Table
            
            ############################################################################################################
            # Keep data based on UID and dt
            tmp_sub_feature_table_rdd = tmp_feature_table_rdd.join(tmp_aim_node_UID_rdd_Broadcast, 
                                                 (tmp_feature_table_rdd[Aim_Node_type + '_UID'] ==  
                                                  tmp_aim_node_UID_rdd_Broadcast[Aim_Node_type + '_UID']) &
                                                 (tmp_feature_table_rdd['dt'] ==  
                                                  tmp_aim_node_UID_rdd_Broadcast['Feature_Time']), 'inner').\
                                                  drop(tmp_aim_node_UID_rdd_Broadcast[Aim_Node_type + '_UID'])

            # 删除dt列(后面只使用Feature_Time这一列)
            tmp_sub_feature_table_rdd = tmp_sub_feature_table_rdd.drop('dt')

            # 通过persist保留计算结果
            tmp_sub_feature_table_rdd = tmp_sub_feature_table_rdd.persist()

            *

            # 依次对各个目标时间对应的特征进行检测，并计算特征总结
            tmp_time_feature_table_summary_pd_list = []
            for tmp_Aim_Feature_Table_dt in tmp_Aim_Feature_Table_dt_list:
                print('处理时间:', tmp_Aim_Feature_Table_dt)

                # 只保留对应时间的信息
                tmp_time_feature_table_rdd = tmp_feature_table_rdd.where("dt = '" + tmp_Aim_Feature_Table_dt + "'")

                # 删除dt列
                tmp_time_feature_table_rdd = tmp_time_feature_table_rdd.drop('dt')

                # 通过persist保留计算结果
                tmp_time_feature_table_rdd = tmp_time_feature_table_rdd.persist()

                # 查看对应时间内的数据是否有重复
                tmp_time_feature_table_rdd_raw_count = tmp_time_feature_table_rdd.count()

                if tmp_time_feature_table_rdd_raw_count == 0:
                    print('Error: 特征表', tmp_feature_table_name, '在时间', tmp_Aim_Feature_Table_dt, '的数据为空，得及时处理')
                else:
                    # 对特征表去重（理论上不需要，但防止有不符合规范的表）
                    tmp_time_feature_table_rdd = tmp_time_feature_table_rdd.dropDuplicates([Aim_Node_type + '_UID'])

                    # 通过persist保留计算结果
                    tmp_time_feature_table_rdd = tmp_time_feature_table_rdd.persist()

                    tmp_time_feature_table_rdd_count = tmp_time_feature_table_rdd.count()

                    # 检查数据量是否保持一致
                    if tmp_time_feature_table_rdd_raw_count > tmp_time_feature_table_rdd_count:
                        print('Error: 特征表', tmp_feature_table_name, '在时间', tmp_Aim_Feature_Table_dt, 
                            '的数据中有重复UID，得及时修改, 目前先保留第一条信息，原始行数为:', 
                             tmp_time_feature_table_rdd_raw_count,
                             '去重后为:', tmp_time_feature_table_rdd_count)

                # 对
                tmp_time_feature_table_summary_pd = tmp_time_feature_table_rdd.select(tmp_useful_feature_cols_list)\
                                                           .summary("min", "max", "mean", "stddev").toPandas()

                tmp_time_feature_table_summary_pd['dt'] = tmp_Aim_Feature_Table_dt

                tmp_time_feature_table_summary_pd_list.append(tmp_time_feature_table_summary_pd)

            tmp_feature_table_summary_pd = pd.concat(tmp_time_feature_table_summary_pd_list, ignore_index = True)


            # 将各时间段的特征分隔开来，从而进行归一化等操作
            print('根据对应时间分隔特征和总结信息')
            for tmp_Aim_Feature_Table_dt in tmp_Aim_Feature_Table_dt_list:
                print('处理时间:', tmp_Aim_Feature_Table_dt)

                # 只保留对应时间的信息
                tmp_time_sub_feature_table_rdd = tmp_sub_feature_table_rdd.where("Feature_Time = '" + tmp_Aim_Feature_Table_dt + "'")

                # 删除Feature_Time列
                tmp_time_sub_feature_table_rdd = tmp_time_sub_feature_table_rdd.drop('Feature_Time')

                # 和之前已获取的特征合并
                if tmp_Aim_Feature_Table_dt not in tmp_aim_node_feature_rdd_dict:
                    tmp_aim_node_UID_time_feature_rdd = tmp_time_sub_feature_table_rdd
                else:
                    tmp_aim_node_UID_time_feature_rdd = tmp_aim_node_feature_rdd_dict[tmp_Aim_Feature_Table_dt]

                    tmp_aim_node_UID_time_feature_rdd = tmp_aim_node_UID_time_feature_rdd.join(tmp_time_sub_feature_table_rdd, 
                                                                       Aim_Node_type + '_UID', 'outer')

                tmp_aim_node_feature_rdd_dict[tmp_Aim_Feature_Table_dt] = tmp_aim_node_UID_time_feature_rdd

                #########################################################################################################
                tmp_time_feature_table_summary_pd = tmp_feature_table_summary_pd[tmp_feature_table_summary_pd['dt'] == 
                                                            tmp_Aim_Feature_Table_dt]

                tmp_time_feature_table_summary_pd = tmp_time_feature_table_summary_pd.drop(columns = ['dt'])

                # 查看是否有无效列(特征都为同一值)，及时提醒
                tmp_summary_min = tmp_time_feature_table_summary_pd[tmp_time_feature_table_summary_pd['summary'] == 'min'].values[0]
                tmp_summary_max = tmp_time_feature_table_summary_pd[tmp_time_feature_table_summary_pd['summary'] == 'max'].values[0]

                tmp_problem_columns = np.array(tmp_time_feature_table_summary_pd.columns)[tmp_summary_min == tmp_summary_max]

                if tmp_problem_columns.shape[0] > 0:
                    print('ERROR: 特征表', tmp_feature_table_name, '在时间', tmp_Aim_Feature_Table_dt, 
                        '存在一些列的全部行都是一个值，具体情况如下，得及时修改')
                    print(dict(tmp_time_feature_table_summary_pd[tmp_problem_columns].iloc[0]))

                # 删除summary列后记录结果
                tmp_time_feature_table_summary_pd = tmp_time_feature_table_summary_pd.set_index('summary')
                if tmp_Aim_Feature_Table_dt not in tmp_feature_table_summary_pd_dict:
                    tmp_feature_table_summary_pd_dict[tmp_Aim_Feature_Table_dt] = tmp_time_feature_table_summary_pd
                else:
                    tmp_old_pd = tmp_feature_table_summary_pd_dict[tmp_Aim_Feature_Table_dt]
                    
                    tmp_feature_table_summary_pd_dict[tmp_Aim_Feature_Table_dt] = pd.concat([tmp_old_pd,
                                                                     tmp_time_feature_table_summary_pd], axis = 1)
                    
            print('完成对该特征表的采样')
            print('----------------------------------------------------------------------------')

        print('开始转pandas及进行归一化等预处理')

        # 依次处理各个目标时间对应的特征的结果
        tmp_result_feature_pd_list_dict = {'Raw': [], 'Norm': [], 'Std': []}
        for tmp_Aim_Feature_Table_dt in tmp_Aim_Feature_Table_dt_list:
            # 将结果转为pandas
            tmp_aim_node_feature_pandas_raw = tmp_aim_node_feature_rdd_dict[tmp_Aim_Feature_Table_dt].toPandas()

            # 将空值补0
            tmp_aim_node_feature_pandas_raw = tmp_aim_node_feature_pandas_raw.fillna(0)
            
            # 先取出UID列
            tmp_aim_node_feature_pandas_UID = tmp_aim_node_feature_pandas_raw[[Aim_Node_type + '_UID']]

            # 删去UID列
            tmp_aim_node_feature_pandas_raw = tmp_aim_node_feature_pandas_raw.drop(columns=[Aim_Node_type + '_UID'])

            # 合并各特征表的统计结果
            tmp_feature_table_summary_pd = tmp_feature_table_summary_pd_dict[tmp_Aim_Feature_Table_dt]

            # 进行标准化和归一化前先确认各列列名一致，否则报错
            if list(tmp_aim_node_feature_pandas_raw.columns) != list(tmp_feature_table_summary_pd.columns):
                print('Error: 特征列和总结列列名不一致')
                print('特征列:', list(tmp_aim_node_feature_pandas_raw.columns))
                print('总结列:', list(tmp_feature_table_summary_pd.columns))
                return

            # 基于统计结果进行归一化和标准化
            tmp_pd_min = tmp_feature_table_summary_pd.loc['min'].astype('float')
            tmp_pd_max = tmp_feature_table_summary_pd.loc['max'].astype('float')
            tmp_pd_mean = tmp_feature_table_summary_pd.loc['max'].astype('float')
            tmp_pd_std = tmp_feature_table_summary_pd.loc['max'].astype('float')
            
#             print(tmp_pd_min.dtypes)
#             print(tmp_pd_max.dtypes)
#             print(tmp_aim_node_feature_pandas_raw.dtypes)
            
            # 进行归一化和标准化
            tmp_aim_node_feature_pandas_norm = ((tmp_aim_node_feature_pandas_raw - tmp_pd_min) / (tmp_pd_max - tmp_pd_min + eps))
            
            tmp_aim_node_feature_pandas_std = ((tmp_aim_node_feature_pandas_raw - tmp_pd_mean) / np.sqrt(tmp_pd_std + eps))
            
            # 拼接上各行对应的UID以及时间
            tmp_aim_node_feature_pandas_UID['Feature_Time'] = tmp_Aim_Feature_Table_dt

            tmp_aim_node_feature_pandas_raw = pd.concat([tmp_aim_node_feature_pandas_UID, tmp_aim_node_feature_pandas_raw], axis = 1)
            tmp_aim_node_feature_pandas_norm = pd.concat([tmp_aim_node_feature_pandas_UID, tmp_aim_node_feature_pandas_norm], axis = 1)
            tmp_aim_node_feature_pandas_std = pd.concat([tmp_aim_node_feature_pandas_UID, tmp_aim_node_feature_pandas_std], axis = 1)

            tmp_result_feature_pd_list_dict['Raw'].append(tmp_aim_node_feature_pandas_raw)
            tmp_result_feature_pd_list_dict['Norm'].append(tmp_aim_node_feature_pandas_norm)
            tmp_result_feature_pd_list_dict['Std'].append(tmp_aim_node_feature_pandas_std)

        # 汇总各时间的结果
        tmp_aim_node_feature_pandas_raw = pd.concat(tmp_result_feature_pd_list_dict['Raw'], ignore_index = True)
        tmp_aim_node_feature_pandas_norm = pd.concat(tmp_result_feature_pd_list_dict['Norm'], ignore_index = True)
        tmp_aim_node_feature_pandas_std = pd.concat(tmp_result_feature_pd_list_dict['Std'], ignore_index = True)

        # 按目标节点顺序重新排序
        tmp_aim_node_feature_pandas_raw = tmp_aim_node_UID_order_pd.merge(tmp_aim_node_feature_pandas_raw, how = 'left', 
                                                    on = [Aim_Node_type + '_UID', 'Feature_Time'])
        tmp_aim_node_feature_pandas_norm = tmp_aim_node_UID_order_pd.merge(tmp_aim_node_feature_pandas_norm, how = 'left', 
                                                     on = [Aim_Node_type + '_UID', 'Feature_Time'])
        tmp_aim_node_feature_pandas_std = tmp_aim_node_UID_order_pd.merge(tmp_aim_node_feature_pandas_std, how = 'left', 
                                                    on = [Aim_Node_type + '_UID', 'Feature_Time'])

        # 保留各类型的特征
        tmp_aim_node_feature_pandas_raw.to_pickle(Aim_Node_Raw_Feature_file)
        tmp_aim_node_feature_pandas_norm.to_pickle(Aim_Node_Norm_Feature_file)
        tmp_aim_node_feature_pandas_std.to_pickle(Aim_Node_Std_Feature_file)
        
        if not Return_Data:
            return None
    
    else:
        print('目标点特征表已存在，可直接读取')
        
        if not Return_Data:
            return None
        
    tmp_aim_node_feature_pandas_raw = pd.read_pickle(Aim_Node_Raw_Feature_file)
    tmp_aim_node_feature_pandas_norm = pd.read_pickle(Aim_Node_Norm_Feature_file)
    tmp_aim_node_feature_pandas_std = pd.read_pickle(Aim_Node_Std_Feature_file)

    # 保存需要的格式的数据
    tmp_Start_Node_All_Output_Feature_list = []
    if 'Raw' in Preprocess_Type_List:
        tmp_Start_Node_All_Output_Feature_list.append(tmp_aim_node_feature_pandas_raw.drop(columns=[Aim_Node_type + '_UID', 'Feature_Time']))
    if 'Norm' in Preprocess_Type_List:
        tmp_Start_Node_All_Output_Feature_list.append(tmp_aim_node_feature_pandas_norm.drop(columns=[Aim_Node_type + '_UID', 'Feature_Time']))
    if 'Std' in Preprocess_Type_List:
        tmp_Start_Node_All_Output_Feature_list.append(tmp_aim_node_feature_pandas_std.drop(columns=[Aim_Node_type + '_UID', 'Feature_Time']))

    tmp_aim_node_feature_pandas_all = pd.concat(tmp_Start_Node_All_Output_Feature_list, axis = 1)

    # 空值按0处理
    tmp_aim_node_feature_pandas_all = tmp_aim_node_feature_pandas_all.fillna(0)

    # 无穷值也按0处理
    tmp_aim_node_feature_pandas_all.replace([np.inf, -np.inf], 0, inplace=True)

    return tmp_aim_node_feature_pandas_all


"""
作用：
    从线上读取聚合后的pandas数据（包含标签列 + 处理后的元路径特征列）。

输入：

    
返回值：
    输出结果直接存储在对应文件夹中
"""
def Get_Aggregated_Data_From_Online(Spark_Session, Aim_UID_info_dict, Feature_Data_From_Online_Config_dict, Feature_Data_From_Online_Time_Store_dir,
                         regenerate = False, Meta_path_drop_list = [], Meta_path_Column_drop_dict = {}, 
                         Preprocess_Type_List = ['Norm', 'Std'], sample_start = None, sample_end = None, Data_Output_Type = 'Pandas'):
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
    
    # broadcast目标列，加快运算速度
    tmp_aim_node_UID_rdd_Broadcast = broadcast(tmp_aim_node_UID_rdd)
    
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
                                                  tmp_aim_node_UID_rdd_Broadcast, tmp_aim_node_UID_order_pd, 
                                                  Feature_Data_From_Online_Time_Store_dir, 
                                                  Preprocess_Type_List, 'Start_Node')

    # 保存结果
    if Data_Output_Type == 'Pandas':
        Result_Data_dict['Start_Node_Feature_List'].append(tmp_aim_node_feature_pandas_all)
    elif Data_Output_Type == 'Torch':
        Result_Data_dict['Start_Node_Feature_List'].append(torch.FloatTensor(tmp_aim_node_feature_pandas_all.values))

    # 记录特征长度
    Result_Data_dict['Node_Type_to_Feature_len']['Start_Node'] = tmp_aim_node_feature_pandas_all.shape[1]
    
    tmp_start_node_end_time = datetime.now()
    
    print(Aim_Node_type + '目标节点涉及的全部特征数:', tmp_aim_node_feature_pandas_all.shape[1])
    print('处理起始点总共花费时间:', (tmp_start_node_end_time - tmp_start_node_start_time))
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

        # 获得对应元路径的配置信息
        tmp_meta_path_info = Feature_Data_From_Online_Config_dict['Meta_Path_Feature_Table_List'][tmp_meta_path_name]

        # 依次处理元路径中各列的信息
        for tmp_column_i in range(len(tmp_meta_path_info)):
            if tmp_meta_path_name in Meta_path_Column_drop_dict and tmp_column_i in Meta_path_Column_drop_dict[tmp_meta_path_name]:
                print('跳过元路径:', tmp_meta_path_name, '的第', tmp_column_i, '列')
                continue
                
            tmp_meta_path_column_start_time = datetime.now()
            
            # 获取对应列节点类型
            tmp_column_node_class = tmp_meta_path_info[tmp_column_i]["Node_class"]
            
            # 初始化该列的特征的存储
            Result_Data_dict['Meta_Path_Feature'][tmp_meta_path_name][tmp_column_i] = []
            
            # 保留节点类型信息
            Result_Data_dict['Meta_Path_Column_Type'][tmp_meta_path_name][tmp_column_i] = tmp_column_node_class
            
            # 获取对这类型特征的说明
            tmp_node_comment = tmp_meta_path_name + '_' + str(tmp_column_i) + '_' + tmp_column_node_class
            
            # 将特征表信息转化为指定格式
            tmp_Feature_Table_Config_list = []
            for tmp_aim_feature_table_i in range(len(tmp_meta_path_info[tmp_column_i]["Feature_Table_List"])):
                tmp_feature_table_name = tmp_meta_path_info[tmp_column_i]["Feature_Table_List"][tmp_aim_feature_table_i]
                tmp_feature_summary_table_name = tmp_meta_path_info[tmp_column_i]["Feature_Table_Summary_List"][tmp_aim_feature_table_i]
                tmp_aim_column_name = 'Start_Column'
                
                tmp_Feature_Table_Config_list.append({"Table_Name":tmp_feature_table_name, 
                                          "Summary_Table_Name": tmp_feature_summary_table_name,
                                          "UID": tmp_aim_column_name})
            
            # 获取目标数据
            tmp_aim_node_feature_pandas_all = Get_Proc_Feature_Data_From_Online(Spark_Session, tmp_Feature_Table_Config_list, regenerate, 
                                                          tmp_Aim_Feature_Table_dt_list, Aim_Node_type,
                                                          tmp_aim_node_UID_rdd_Broadcast, tmp_aim_node_UID_order_pd, 
                                                          Feature_Data_From_Online_Time_Store_dir, 
                                                          Preprocess_Type_List, tmp_node_comment)

            # 保存结果
            if Data_Output_Type == 'Pandas':
                Result_Data_dict['Meta_Path_Feature'][tmp_meta_path_name][tmp_column_i].append(tmp_aim_node_feature_pandas_all)
            elif Data_Output_Type == 'Torch':
                tmp_aim_node_feature_torch_all = torch.FloatTensor(tmp_aim_node_feature_pandas_all.values)
                Result_Data_dict['Meta_Path_Feature'][tmp_meta_path_name][tmp_column_i].append(tmp_aim_node_feature_torch_all)

            # 记录特征长度
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
def Concat_Feature_For_ML_Model(Result_Data_dict, delete_original_data = True):
    tmp_feature_to_concat_list = []
    
    tmp_add_feature_pd = Result_Data_dict['Start_Node_Feature_List'][0]

    # 给特征列名加上注释
    tmp_add_feature_pd.columns = tmp_add_feature_pd.columns + '__Start_Node'

    tmp_feature_to_concat_list.append(tmp_add_feature_pd)

    if delete_original_data == True:
        del Result_Data_dict['Start_Node_Feature_List']
        gc.collect()
    
    for tmp_meta_path_name in Result_Data_dict['Meta_Path_Feature']:
        tmp_column_i_list = list(Result_Data_dict['Meta_Path_Feature'][tmp_meta_path_name].keys())
        for tmp_column_i in tmp_column_i_list:
            # 只读取最近一个月的数据
            tmp_add_feature_pd = Result_Data_dict['Meta_Path_Feature'][tmp_meta_path_name][tmp_column_i][0]
            
            # 给特征名加上元路径及列序号
            tmp_add_feature_pd.columns = tmp_add_feature_pd.columns + '__' + tmp_meta_path_name + '__' + str(tmp_column_i)
            
            tmp_feature_to_concat_list.append(tmp_add_feature_pd)
            
            if delete_original_data == True:
                del Result_Data_dict['Meta_Path_Feature'][tmp_meta_path_name][tmp_column_i]
                gc.collect()
    
    Result_Data_dict['Feature'] = pd.concat(tmp_feature_to_concat_list, axis = 1)
    
    return Result_Data_dict