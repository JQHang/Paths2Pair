from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.sql.types import *
from pyspark.sql.functions import broadcast
from pyspark.storagelevel import StorageLevel

from Utils.Pyspark_utils import sample_random_n_samples_for_samll_rdd
from Utils.Pyspark_utils import sample_top_n_groupby_samples_for_samll_rdd
from Utils.Pyspark_utils import sample_random_n_groupby_samples_for_samll_rdd
from Utils.Pyspark_utils import sample_rdd_from_aim_row, sample_rdd_from_aim_range
from Utils.Pyspark_utils import Groupby_Feature_Table, Groupby_Pyspark_Table
from Utils.Pyspark_utils import Pyspark_Create_Table, Upload_RDD_Data_to_Database

import pandas as pd
import numpy as np
import math
import os
import json
from tqdm import tqdm
from datetime import datetime

def Upload_Aggregated_Feature_and_Summary(Spark_Session, tmp_aim_path_result_rdd, tmp_all_feature_table_rdd, tmp_add_feature_node_type,
                            tmp_all_useful_feature_cols_list, tmp_all_useful_feature_cols_comments_list, Groupby_Type_List,
                            tmp_add_feature_start_time, Max_Column_Number, Table_Name_Comment, Aim_Node_type, 
                            Feature_Table_Upload_Count_list, tmp_relation_name_key, tmp_add_feature_column_name, Aim_Relation_Table_dt, 
                            Aim_Feature_Table_dt, Output_Summary):
    
    # 为关系拼接特征
    tmp_sub_meta_path_feature_result_rdd = tmp_aim_path_result_rdd.join(tmp_all_feature_table_rdd, 
                                                  tmp_add_feature_node_type + '_UID', 'left')

    # 删去目标列，只保留起始列及目标列对应的特征
    tmp_sub_meta_path_feature_result_rdd = tmp_sub_meta_path_feature_result_rdd.drop(tmp_add_feature_node_type + '_UID')

    # 未拼接上的特征默认是0
    tmp_sub_meta_path_feature_result_rdd = tmp_sub_meta_path_feature_result_rdd.fillna(0)

    tmp_add_feature_end_time = datetime.now()

    print(tmp_add_feature_node_type + '节点本次上传涉及的全部特征数:', len(tmp_all_useful_feature_cols_list) * len(Groupby_Type_List))
    print(tmp_add_feature_end_time, '读取本次上传全部特征表总共花费时间:', (tmp_add_feature_end_time - tmp_add_feature_start_time))
    print("----------------------------------------------------------------------------")

    # 查看是否有数据倾斜问题
#     print('聚合前特征表数据分区数:', tmp_sub_meta_path_feature_result_rdd.rdd.getNumPartitions())

#     tmp_sub_meta_path_feature_result_rdd = tmp_sub_meta_path_feature_result_rdd.persist()
#     tmp_count_item_list = tmp_sub_meta_path_feature_result_rdd.rdd.sample(False, 0.01).countByKey().items()
#     tmp_count_item_list = sorted(tmp_count_item_list, key=lambda x: x[1], reverse=True)
#     print(tmp_count_item_list[0:100])

#     # 保存要聚合的数据
#     tmp_sub_meta_path_feature_result_rdd.write.csv(tmp_add_feature_column_name + '_test_data_skewness.csv', header = True)
    
#     print(tmp_sub_meta_path_feature_result_rdd.explain(True))
    
    # 重新分区，防止出现数据倾斜
    tmp_sub_meta_path_feature_result_rdd.repartition("Start_Column")
    
    # 按起始点进行groupby并计算Groupby_Type_List
    print('进行groupby的计算')
    tmp_feat_groupby_result_rdd = Groupby_Feature_Table(Spark_Session, tmp_sub_meta_path_feature_result_rdd, 
                                       'Start_Column', tmp_all_useful_feature_cols_list,
                                       Groupby_Type_List)

    groupby_upload_start_time = datetime.now()

    # 设置上传对应的表名
    if Max_Column_Number > 0:
        tmp_meta_path_feature_table_name = ('tmp.tmp___JL_MP_N_Feat_Agg___' + Table_Name_Comment + '_' + 
                                 str(Feature_Table_Upload_Count_list[0]) + '___' + 
                                 tmp_relation_name_key + '___' + tmp_add_feature_column_name)
    else:
        tmp_meta_path_feature_table_name = ('tmp.tmp___JL_MP_N_Feat___' + tmp_relation_name_key + '___' + 
                                 tmp_add_feature_column_name + '___Groupby_Result___' + Table_Name_Comment)

    if len(tmp_meta_path_feature_table_name) > 128:
        tmp_meta_path_feature_table_name = tmp_meta_path_feature_table_name[:128]
        print('Error: 只能保留表名的前128位，可能出问题')

    print('输出表名为', tmp_meta_path_feature_table_name)

    # 获取全部特征列列名
    tmp_groupby_feature_name_list = ['Start_Column']
    tmp_groupby_feature_comment_list = [Aim_Node_type + '_UID']
    for tmp_groupby_type in Groupby_Type_List:
        if tmp_groupby_type in ['AVG', 'SUM', 'MAX', 'MIN']:
            # 特征表列名为UID列加特征列
            tmp_groupby_feature_name_list = (tmp_groupby_feature_name_list + 
                                   [tmp_groupby_type + '_' + x for x in tmp_all_useful_feature_cols_list])

            # 特征表注释
            tmp_groupby_feature_comment_list = (tmp_groupby_feature_comment_list + 
                                     [tmp_groupby_type+'_'+x for x in tmp_all_useful_feature_cols_comments_list])
        elif tmp_groupby_type == 'COUNT':
            # 特征表列名为UID列加特征列
            tmp_groupby_feature_name_list = tmp_groupby_feature_name_list + ['Groupby_COUNT']

            # 特征表注释
            tmp_groupby_feature_comment_list = tmp_groupby_feature_comment_list + ["关联节点个数"]

        else:
            continue

    # 查看是否有过长的特征名
    print('查看是否有过长的特征名,并将过长列名转化为feature_column_i的格式')
    for tmp_groupby_feature_name_i in range(1, len(tmp_groupby_feature_name_list)):
        if len(tmp_groupby_feature_name_list[tmp_groupby_feature_name_i]) > 128:
            # 如果存在过长列名，则将对应特征列的列名转化为序号
    #                     print('存在过长列名:' + tmp_groupby_feature_name_list[tmp_groupby_feature_name_i] + '长度为:',
    #                         len(tmp_groupby_feature_name_list[tmp_groupby_feature_name_i]), '将对应特征列名转化为feature_column_i的格式')

            tmp_feat_groupby_result_rdd = tmp_feat_groupby_result_rdd.withColumnRenamed( 
                                                        tmp_groupby_feature_name_list[tmp_groupby_feature_name_i], 
                                                        'Feature_Column_' + str(tmp_groupby_feature_name_i))

            tmp_groupby_feature_name_list[tmp_groupby_feature_name_i] = ('Feature_Column_' + 
                                                     str(tmp_groupby_feature_name_i))

    # 如果有重复的列名，则报错
    if len(set(tmp_groupby_feature_name_list)) != len(tmp_groupby_feature_name_list):
        print('存在重复列名')
        print(tmp_groupby_feature_name_list)
        return

    # 获取对应列的数据
    tmp_meta_path_feature_for_upload = tmp_feat_groupby_result_rdd.select(tmp_groupby_feature_name_list)
    tmp_meta_path_feature_for_upload = tmp_meta_path_feature_for_upload.persist()

    # 获取各列的类型
    tmp_groupby_feature_type_list = []
    for _, col_type in tmp_meta_path_feature_for_upload.dtypes: 
        tmp_groupby_feature_type_list.append(col_type)

    print('开始创建表')

    # 创建表（如果特征表已存在，会自动不进行创建）
    Pyspark_Create_Table(Spark_Session, tmp_meta_path_feature_table_name, tmp_groupby_feature_name_list, 
                  tmp_groupby_feature_type_list, tmp_groupby_feature_comment_list)

    # 设定临时view的名称
    tmp_view_name = 'Groupby_Feature_Data_Upload_' + datetime.now().strftime("%m_%d_%H_%M")

    # 创建临时view
    tmp_meta_path_feature_for_upload.createTempView(tmp_view_name)

    # 生成upload对应的dt
    if Aim_Relation_Table_dt == Aim_Feature_Table_dt:
        tmp_up_load_table_dt = Aim_Relation_Table_dt
    else:
        tmp_up_load_table_dt = Aim_Relation_Table_dt + '___' + Aim_Feature_Table_dt

    print('开始运算并上传数据')

    # 上传特征（如果是第一次上传，则清空对应dt的数据）
    sql_str = """insert overwrite table """ + tmp_meta_path_feature_table_name + """ 
             partition(dt='""" + tmp_up_load_table_dt + """')(
             select * from """ + tmp_view_name + """
            )    
            """

    Spark_Session.sql(sql_str)

    # 删除临时view
    Spark_Session.catalog.dropTempView(tmp_view_name)

    groupby_upload_end_time = datetime.now()
    print('完成节点特征对应的合并Groupby表的上传, 运算时间为:', (groupby_upload_end_time - groupby_upload_start_time))

    if Output_Summary:
        # 统计合并后的表的信息
        tmp_feature_table_summary_rdd = tmp_meta_path_feature_for_upload.drop('Start_Column').summary("min", "max", "mean", "stddev")

        # 设置上传对应的表名
        if Max_Column_Number > 0:
            tmp_meta_path_feature_summary_table_name = ('tmp.tmp___JL_MP_N_Feat_Summary___' + Table_Name_Comment + '_' 
                                          + str(Feature_Table_Upload_Count_list[0]) + '___' + 
                                          tmp_relation_name_key + '___' + tmp_add_feature_column_name)
        else:
            tmp_meta_path_feature_summary_table_name = ('tmp.tmp___JL_MP_N_Feat___' + tmp_relation_name_key 
                                          + '___' + tmp_add_feature_column_name + 
                                          '___Groupby_Summary___' + Table_Name_Comment)

        if len(tmp_meta_path_feature_summary_table_name) > 128:
            tmp_meta_path_feature_summary_table_name = tmp_meta_path_feature_summary_table_name[:128]
            print('Error:只能保留表名的前128位, 可能出问题')

        print('输出总结表名为', tmp_meta_path_feature_summary_table_name)

        # 上传统计信息
        Upload_RDD_Data_to_Database(Spark_Session, tmp_meta_path_feature_summary_table_name, 
                           tmp_feature_table_summary_rdd, tmp_up_load_table_dt, [], [])
        print('完成统计信息的上传,耗时为:', (datetime.now() - groupby_upload_end_time))

    tmp_meta_path_feature_for_upload = tmp_meta_path_feature_for_upload.unpersist()
                    
    return

"""
作用：

运算流程:
    先算元路径
    针对元路径的尾结点拼接特征
    对拼接特征后的表算groupby
    上传结果表

输入：
    Spark_Session：pyspark接口
    Metapath_Config_dict：元路径配置文件
    Feature_Dataset_Config_dict：
    Aim_Table_dt：元路径输出文件夹
    Create_Table:是否要重新生成文件
    Groupby_Type_List

返回值：
    无返回值，运算结果都会保存在tmp_meta_path_output_dir和tmp_node_output_dir这两个文件夹下
"""
def Meta_Path_Feature_Generate_and_Upload(Spark_Session, Metapath_Feature_Config_dict, tmp_relation_name_key, Feature_Dataset_Config_dict, 
                            Aim_Relation_Table_dt, Aim_Feature_Table_dt, Column_Processed_Count_list, Output_Weight_Feature_Table_list,
                            Feature_Table_Processed_Count_list, Feature_Table_Processed_Column_Name_list,
                            Feature_Table_Upload_Count_list, Table_Name_Comment, Output_Columns_Type = "Head_And_Tail", 
                            Output_Split_Groupby_Table = False):
        
    print('生成元路径', tmp_relation_name_key, '对应的特征表')
    print('输出列的选择方案：', Output_Columns_Type)
    
    ##############################################################################################
    # 该元路径的信息
    tmp_meta_path_info_dict = Metapath_Feature_Config_dict[tmp_relation_name_key]
    
    # 获取要执行的groupby操作
    if 'Groupby_Type_List' in tmp_meta_path_info_dict:
        Groupby_Type_List = tmp_meta_path_info_dict['Groupby_Type_List']
    else:
        Groupby_Type_List = ['AVG', 'MAX', 'MIN']
    print('要执行的groupby操作:', Groupby_Type_List)
    
    # 获取要读取的元路径涉及的全部关系
    tmp_aim_meta_path_Relation_List = tmp_meta_path_info_dict['Relation_List']

    # 元路径起始列就是目标节点和目标列
    Aim_Node_type = tmp_aim_meta_path_Relation_List[0]['Head_Node_class']

    Aim_Meta_Path_Start_Column = tmp_aim_meta_path_Relation_List[0]['Head_Column_name_AS']
    
    # 表的注释
    if "Table_Name_Comment" in tmp_meta_path_info_dict:
        Table_Name_Comment = tmp_meta_path_info_dict["Table_Name_Comment"]
    print("Table_Name_Comment:", Table_Name_Comment)
    
    # 输出列的类型
    if "Output_Columns_Type" in tmp_meta_path_info_dict:
        Output_Columns_Type = tmp_meta_path_info_dict["Output_Columns_Type"]
    print("Output_Columns_Type:", Output_Columns_Type)
    
    # 一次性可处理的最大特征列数
    if "Max_Column_Number" in tmp_meta_path_info_dict:
        Max_Column_Number = tmp_meta_path_info_dict["Max_Column_Number"]
    else:
        Max_Column_Number = -1
    
    # 是否要输出Summary表
    if "Output_Summary" in tmp_meta_path_info_dict and tmp_meta_path_info_dict["Output_Summary"] == "True":
        Output_Summary = True
    else:
        Output_Summary = False
    ##############################################################################################
    # 存储输出列对应的节点类型
    tmp_column_name_to_class_dict = {}
    
    # 存储节点列对应的权重列
    tmp_node_column_to_weight_dict = {}
    
    # 存储权重列是否有值
    tmp_weight_column_name_to_exist_dict = {}
    
    # 起始点是一定会输出的
    tmp_column_name_to_class_dict[tmp_aim_meta_path_Relation_List[0]['Head_Column_name_AS']] = tmp_aim_meta_path_Relation_List[0]['Head_Node_class']

    # 存储目标输出列列名
    tmp_all_aim_column_name_list = [tmp_aim_meta_path_Relation_List[0]['Head_Column_name_AS']]

    ##############################################################################################
    for tmp_Relation_index, tmp_aim_meta_path_sub_Relation in enumerate(tmp_aim_meta_path_Relation_List):
        print('处理到关系表:', tmp_aim_meta_path_sub_Relation['Relation_Data'])
        
        # 读取该关系表中的头列
        tmp_sql_command = """SELECT\n    """ + tmp_aim_meta_path_sub_Relation['Head_Column_name'] + """ AS """ 
        tmp_sql_command = tmp_sql_command + tmp_aim_meta_path_sub_Relation['Head_Column_name_AS'] 
        print('起始列为:', tmp_aim_meta_path_sub_Relation['Head_Column_name'])
        
        # 根据设定的情况，添加尾列
        if 'Tail_Column_name' in tmp_aim_meta_path_sub_Relation:
            tmp_sql_command = tmp_sql_command + """,\n    """ + tmp_aim_meta_path_sub_Relation['Tail_Column_name'] + """ AS """ 
            
            tmp_sql_command = tmp_sql_command + tmp_aim_meta_path_sub_Relation['Tail_Column_name_AS']
            
            print('添加尾列:', tmp_aim_meta_path_sub_Relation['Tail_Column_name'])
            
        elif 'Tail_Column_name_list' in tmp_aim_meta_path_sub_Relation:
            for tmp_tail_column_i in range(len(tmp_aim_meta_path_sub_Relation['Tail_Column_name_list'])):
                tmp_sql_command = tmp_sql_command + """,\n    """ + tmp_aim_meta_path_sub_Relation['Tail_Column_name_list'][tmp_tail_column_i]
                tmp_sql_command = tmp_sql_command + """ AS """
                tmp_sql_command = tmp_sql_command + tmp_aim_meta_path_sub_Relation['Tail_Column_name_AS_list'][tmp_tail_column_i]
                
                print('添加尾列:', tmp_aim_meta_path_sub_Relation['Tail_Column_name_list'][tmp_tail_column_i])
        else:
            print('Error: 未指定尾列')
            return
        
        # 如果有权重列的话就加上
        tmp_Weight_Column_AS = "Weight_" + str(tmp_Relation_index)
        if 'Weight_Column' in tmp_aim_meta_path_sub_Relation and tmp_aim_meta_path_sub_Relation['Weight_Column'] != '':
            tmp_sql_command = tmp_sql_command + """,\n    """ + tmp_aim_meta_path_sub_Relation['Weight_Column'] + """ AS """ + tmp_Weight_Column_AS
            
            tmp_weight_column_name_to_exist_dict[tmp_Weight_Column_AS] = True
        else:
            tmp_sql_command = tmp_sql_command + """,\n    1 AS """ + tmp_Weight_Column_AS
            
            tmp_weight_column_name_to_exist_dict[tmp_Weight_Column_AS] = False

        # 设置表名和dt
        tmp_sql_command = tmp_sql_command + """\nFROM\n    """ + tmp_aim_meta_path_sub_Relation['Relation_Data']
        tmp_sql_command = tmp_sql_command + """\nWHERE\n    dt = '""" + Aim_Relation_Table_dt + """'"""
        
        # 保证首列不为空
        tmp_sql_command = tmp_sql_command + """\n    AND """ + tmp_aim_meta_path_sub_Relation['Head_Column_name'] + """ IS NOT NULL""" 
        
        # 保证尾列也不为空
        if 'Tail_Column_name' in tmp_aim_meta_path_sub_Relation:
            tmp_sql_command = tmp_sql_command + """\n    AND """ + tmp_aim_meta_path_sub_Relation['Tail_Column_name'] + """ IS NOT NULL"""
        elif 'Tail_Column_name_list' in tmp_aim_meta_path_sub_Relation:
            for tmp_tail_column_i in range(len(tmp_aim_meta_path_sub_Relation['Tail_Column_name_list'])):
                tmp_sql_command = tmp_sql_command + """\n    AND """ + tmp_aim_meta_path_sub_Relation['Tail_Column_name_list'][tmp_tail_column_i] + """ IS NOT NULL"""
                
        # 如果有限制条件的话就加上
        if 'Limits' in tmp_aim_meta_path_sub_Relation and tmp_aim_meta_path_sub_Relation['Limits'] != '':
            tmp_sql_command = tmp_sql_command + '\n    AND ' + tmp_aim_meta_path_sub_Relation['Limits']
            print('关系表限制条件为', tmp_aim_meta_path_sub_Relation['Limits'])
        
        print('完整sql语句为:\n' + tmp_sql_command)
        
        tmp_aim_relation_rdd = Spark_Session.sql(tmp_sql_command)

#         tmp_aim_relation_rdd = tmp_aim_relation_rdd.persist()

#         tmp_aim_relation_rdd_count = tmp_aim_relation_rdd.count()
#         print('要合并的关系表的关系总数为:', tmp_aim_relation_rdd.count())
        
#         # 检测是否有重复
#         tmp_aim_relation_rdd_distinct_count = tmp_aim_relation_rdd.dropDuplicates().count()
        
#         if tmp_aim_relation_rdd_distinct_count != tmp_aim_relation_rdd_count:
#             print('WARNING:关系有重复，若去重则关系总数为:', tmp_aim_relation_rdd_distinct_count)
        
        # 获取涉及到的点的list
        if 'Tail_Column_name' in tmp_aim_meta_path_sub_Relation:
            tmp_head_and_tail_list = [tmp_aim_meta_path_sub_Relation['Head_Column_name_AS'], tmp_aim_meta_path_sub_Relation['Tail_Column_name_AS']]
        elif 'Tail_Column_name_list' in tmp_aim_meta_path_sub_Relation:
            tmp_head_and_tail_list = ([tmp_aim_meta_path_sub_Relation['Head_Column_name_AS']] + 
                              tmp_aim_meta_path_sub_Relation['Tail_Column_name_AS_list'])
        
        print('合并如下列完全重复的边的权重:', tmp_head_and_tail_list)
        
        # 将起始列和节点列相同的行合并，权重累加
        tmp_aim_relation_rdd = Groupby_Pyspark_Table(Spark_Session, tmp_aim_relation_rdd, tmp_head_and_tail_list, [tmp_Weight_Column_AS], ['SUM'])
        
        # 将权重列列名去除sum
        tmp_aim_relation_rdd = tmp_aim_relation_rdd.withColumnRenamed('SUM_' + tmp_Weight_Column_AS, tmp_Weight_Column_AS)
        
        # 如果要拼接别的特征表的话就拼上指定列
        if 'Limits_Add_Feature_List' in tmp_aim_meta_path_sub_Relation and len(tmp_aim_meta_path_sub_Relation['Limits_Add_Feature_List']) > 0:
            for tmp_add_feature_info_dict in tmp_aim_meta_path_sub_Relation['Limits_Add_Feature_List']:
                # 读取对应表
                tmp_feature_table_name = tmp_add_feature_info_dict['Table_Name']
                tmp_feature_UID_column_name = tmp_add_feature_info_dict['UID']
                tmp_column_for_feature_to_join = tmp_add_feature_info_dict['UID_AS']
                tmp_feature_aim_column_name_list = tmp_add_feature_info_dict['Column_Name_List']

                tmp_sql_command = """
                            SELECT
                                """ + tmp_feature_UID_column_name + """ AS """ + tmp_column_for_feature_to_join + """,
                                """ + ','.join(tmp_feature_aim_column_name_list) + """
                            FROM
                                """ + tmp_feature_table_name + """
                            WHERE 
                                dt = '""" + Aim_Relation_Table_dt + """'
                            """

                tmp_add_feature_table_rdd = Spark_Session.sql(tmp_sql_command)

                # 将其加入 tmp_aim_relation_rdd
                tmp_aim_relation_rdd = tmp_aim_relation_rdd.join(tmp_add_feature_table_rdd, 
                                                 tmp_column_for_feature_to_join,
                                                 'left')

                print('添加来自' + tmp_feature_table_name + '的特征:', tmp_feature_aim_column_name_list)

            # 如果有对特征的限制条件的话，则执行limit
            if 'Limits_For_Feature' in tmp_aim_meta_path_sub_Relation and tmp_aim_meta_path_sub_Relation['Limits_For_Feature'] != '':
                tmp_aim_relation_rdd = tmp_aim_relation_rdd.where(tmp_aim_meta_path_sub_Relation['Limits_For_Feature'])
                print('对特征的限制条件为', tmp_aim_meta_path_sub_Relation['Limits_For_Feature'])

        # 根据要求，保证每个样本对应关系数目不超出范围，计算特征时一般不用设置(目前基于window functions的方案)
        # https://stackoverflow.com/questions/33749965/choosing-random-items-from-a-spark-groupeddata-object
        if 'Max_Sample_Scale' in tmp_aim_meta_path_sub_Relation and tmp_aim_meta_path_sub_Relation['Max_Sample_Scale'] > 0:
            if 'Max_Sample_Type' in tmp_aim_meta_path_sub_Relation:
                if tmp_aim_meta_path_sub_Relation['Max_Sample_Type'] == 'Random':
                    print('随机采样,最多保留', tmp_aim_meta_path_sub_Relation['Max_Sample_Scale'])
                    tmp_aim_relation_rdd = sample_random_n_groupby_samples_for_samll_rdd(Spark_Session, tmp_aim_relation_rdd, 
                                                             tmp_aim_meta_path_sub_Relation['Head_Column_name_AS'], 
                                                             tmp_aim_meta_path_sub_Relation['Max_Sample_Scale'])
                elif tmp_aim_meta_path_sub_Relation['Max_Sample_Type'] == 'TopN':
                    print('TopN采样,最多保留', tmp_aim_meta_path_sub_Relation['Max_Sample_Scale'])
                    tmp_aim_relation_rdd = sample_top_n_groupby_samples_for_samll_rdd(Spark_Session, tmp_aim_relation_rdd, 
                                                                tmp_aim_meta_path_sub_Relation['Head_Column_name_AS'], 
                                                                tmp_aim_meta_path_sub_Relation['Max_Sample_Scale'],
                                                                tmp_Weight_Column_AS)
            else:
                print('TopN采样,最多保留', tmp_aim_meta_path_sub_Relation['Max_Sample_Scale'])
                tmp_aim_relation_rdd = sample_top_n_groupby_samples_for_samll_rdd(Spark_Session, tmp_aim_relation_rdd, 
                                                            tmp_aim_meta_path_sub_Relation['Head_Column_name_AS'], 
                                                            tmp_aim_meta_path_sub_Relation['Max_Sample_Scale'],
                                                            tmp_Weight_Column_AS)
        
        # 如果是第一个关系，则直接保存副本
        if tmp_Relation_index == 0:
            tmp_meta_path_result_rdd = tmp_aim_relation_rdd
        else:
            # 如果是后面的关系，则和过去的结果拼接
            tmp_meta_path_result_rdd = tmp_meta_path_result_rdd.join(tmp_aim_relation_rdd, 
                                                  tmp_aim_meta_path_sub_Relation['Head_Column_name_AS'], 
                                                  "inner")

        # 如果有join后的limit，则按要求执行
        if 'Limits_After_Join' in tmp_aim_meta_path_sub_Relation and tmp_aim_meta_path_sub_Relation['Limits_After_Join'] != '':
            tmp_meta_path_result_rdd = tmp_meta_path_result_rdd.where(tmp_aim_meta_path_sub_Relation['Limits_After_Join'])
            print('join关系表后限制条件为', tmp_aim_meta_path_sub_Relation['Limits_After_Join'])
        
        # 先保存权重列列名
        tmp_all_aim_column_name_list.append(tmp_Weight_Column_AS)
        
        # 再记录各列对应的节点类型，以及涉及到的全部列名
        if 'Tail_Column_name' in tmp_aim_meta_path_sub_Relation:
            tmp_column_name_to_class_dict[tmp_aim_meta_path_sub_Relation['Tail_Column_name_AS']] = tmp_aim_meta_path_sub_Relation['Tail_Node_class']
            
            tmp_all_aim_column_name_list.append(tmp_aim_meta_path_sub_Relation['Tail_Column_name_AS'])
            
            # 保存各节点列对应的是哪个权重列
            tmp_node_column_to_weight_dict[tmp_aim_meta_path_sub_Relation['Tail_Column_name_AS']] = tmp_Weight_Column_AS
            
        elif 'Tail_Column_name_list' in tmp_aim_meta_path_sub_Relation:
            for tmp_tail_column_i in range(len(tmp_aim_meta_path_sub_Relation['Tail_Column_name_AS_list'])):
                tmp_tail_column_i_name = tmp_aim_meta_path_sub_Relation['Tail_Column_name_AS_list'][tmp_tail_column_i]
                
                tmp_column_name_to_class_dict[tmp_tail_column_i_name] = tmp_aim_meta_path_sub_Relation['Tail_Node_class_list'][tmp_tail_column_i]
                
                tmp_all_aim_column_name_list.append(tmp_tail_column_i_name)
                
                # 保存各节点列对应的是哪个权重列
                tmp_node_column_to_weight_dict[tmp_tail_column_i_name] = tmp_Weight_Column_AS
        
#         # 释放关系表的空间
#         tmp_aim_relation_rdd = tmp_aim_relation_rdd.unpersist()
        
    # 如果设置了终止列，则删去终止列之后的列名
    if 'Tail_Column' in tmp_meta_path_info_dict:
        tmp_tail_column_index = tmp_all_aim_column_name_list.index(tmp_meta_path_info_dict['Tail_Column'])
        tmp_all_aim_column_name_list = tmp_all_aim_column_name_list[:(tmp_tail_column_index + 1)]
        
        tmp_meta_path_result_rdd = tmp_meta_path_result_rdd.select(tmp_all_aim_column_name_list)
        
#     # 如果只要头尾节点，则删去中间的列
#     if Output_Columns_Type == 'Head_And_Tail':
#         tmp_all_aim_column_name_list = [tmp_all_aim_column_name_list[0], tmp_all_aim_column_name_list[-1]]
#     elif Output_Columns_Type == 'ALL_Nodes':
#         # 如果只要节点，则删去权重列
#         tmp_all_aim_column_name_list = [x for x in tmp_all_aim_column_name_list if 'Weight_' not in x]
#     elif Output_Columns_Type == 'ALL_Edges':
#         # 如果只要权重列，则删去节点列
#         tmp_all_aim_column_name_list = [tmp_all_aim_column_name_list[0]] + [x for x in tmp_all_aim_column_name_list if 'Weight_' in x]
    
    print('元路径' + tmp_relation_name_key + '最终包含的列为:', tmp_all_aim_column_name_list)
    
    # 获得全部的目标节点列
    tmp_all_aim_node_column_name_list = [x for x in tmp_all_aim_column_name_list if 'Weight_' not in x]
    
    # 如果只要头尾节点，则删去中间的列
    if Output_Columns_Type in ['Head_And_Tail', "Tail_Edges"]:
        tmp_all_aim_node_column_name_list = [tmp_all_aim_node_column_name_list[0], tmp_all_aim_node_column_name_list[-1]]
    
    print('元路径' + tmp_relation_name_key + '最终目标的节点列为:', tmp_all_aim_node_column_name_list)

    # 如果采样完成后还要进行一次压缩
    if 'Max_All_Sample_Scale' in tmp_meta_path_info_dict:
        # 则直接进行随机采样
        tmp_meta_path_start_column = tmp_aim_meta_path_Relation_List[0]['Head_Column_name_AS']
        tmp_result_max_scale = tmp_meta_path_info_dict['Max_All_Sample_Scale']
        tmp_meta_path_result_rdd = sample_random_n_groupby_samples_for_samll_rdd(Spark_Session, tmp_meta_path_result_rdd, 
                                                         tmp_meta_path_start_column, 
                                                         tmp_result_max_scale)
    
    # 上传获取的元路径关系(待优化)
    
    print('开始元路径' + tmp_relation_name_key + '的生成运算，并计算总行数和起始点数')
    ##############################################################################################
    # 通过persist保留元路径计算结果
    tmp_meta_path_result_rdd = tmp_meta_path_result_rdd.persist()
    
    # 元路径总行数
    tmp_meta_path_rows = tmp_meta_path_result_rdd.count()
    print('元路径总行数:', tmp_meta_path_rows)
    
    # 只取出元路径的起始列
    tmp_meta_path_result_start_column_rdd = tmp_meta_path_result_rdd.select(Aim_Meta_Path_Start_Column)

    # 对起始列去重
    tmp_meta_path_result_start_node_rdd = tmp_meta_path_result_start_column_rdd.distinct()
    
    # 通过persist保留元路径计算结果
    tmp_meta_path_result_start_node_rdd = tmp_meta_path_result_start_node_rdd.persist()
    
#     print('目标UID数据分区数:', tmp_meta_path_result_start_node_rdd.rdd.getNumPartitions())
    
    # 元路径总点数
    tmp_start_node_counts = tmp_meta_path_result_start_node_rdd.count()
    print('起始点总点数:', tmp_start_node_counts)
    
    # 将起始列列名改为'Start_Column'，防止重复
    tmp_meta_path_result_rdd = tmp_meta_path_result_rdd.withColumnRenamed(Aim_Meta_Path_Start_Column, 'Start_Column')

    # 依次处理该元路径中保留的非起始列的各列对应的节点
    for tmp_add_feature_column_i in range(Column_Processed_Count_list[0], len(tmp_all_aim_node_column_name_list) - 1):
        Column_Processed_Count_list[0] = tmp_add_feature_column_i

        # 因为第一列不用算，故对序号加一
        tmp_add_feature_column_name = tmp_all_aim_node_column_name_list[tmp_add_feature_column_i + 1]
        
        # 获取该节点列对应的权重列
        tmp_add_feature_weight_column_name = tmp_node_column_to_weight_dict[tmp_add_feature_column_name]
        
        ###############################################################################################################################
        # 查看是否要进行权重列的聚合
        if Output_Columns_Type in ['ALL_Nodes_And_Edges', 'ALL_Edges', "Tail_Edges"] and Output_Weight_Feature_Table_list[0] == True:
            print('计算元路径中的节点列:', tmp_add_feature_column_name, '的权重列', tmp_add_feature_weight_column_name, '的Groupby结果')

            # 只保留起始列,权重列和节点列
            tmp_sub_meta_path_feature_result_rdd = tmp_meta_path_result_rdd.select(['Start_Column', tmp_add_feature_weight_column_name, 
                                                            tmp_add_feature_column_name])
            
            # 部分情况不用再去重，待优化
            print('对目标边关系进行去重，并累加重复边的权重')
            
            # 将起始列和节点列相同的行合并，权重累加
            tmp_sub_meta_path_feature_result_rdd = Groupby_Pyspark_Table(Spark_Session, tmp_sub_meta_path_feature_result_rdd, 
                                                     ['Start_Column', tmp_add_feature_column_name], 
                                                     [tmp_add_feature_weight_column_name], ['SUM'])
            
            # 将权重列列名去除sum
            tmp_sub_meta_path_feature_result_rdd = tmp_sub_meta_path_feature_result_rdd.withColumnRenamed('SUM_' + 
                                                                           tmp_add_feature_weight_column_name, 
                                                                           tmp_add_feature_weight_column_name)
            
            print(tmp_sub_meta_path_feature_result_rdd.drop('Start_Column', tmp_add_feature_column_name).summary("min", "max", "mean", "stddev").show())
            
            print('进行groupby的计算')
            
            # 计算'AVG', 'SUM', 'MAX', 'MIN'和'COUNT'
            tmp_sub_meta_path_feature_groupby_result_rdd = Groupby_Feature_Table(Spark_Session, tmp_sub_meta_path_feature_result_rdd, 
                                                          'Start_Column', [tmp_add_feature_weight_column_name],
                                                           ['AVG', 'SUM', 'MAX', 'MIN', 'COUNT'])

            groupby_upload_start_time = datetime.now()

            # 设置对应表名
            tmp_meta_path_feature_table_name = ('tmp.tmp___JL_MP_W_Feat___' + tmp_relation_name_key + '___' + tmp_add_feature_column_name + 
                                    '___' + tmp_add_feature_weight_column_name + '___Groupby_Result___' + Table_Name_Comment)
            if len(tmp_meta_path_feature_table_name) > 128:
                tmp_meta_path_feature_table_name = tmp_meta_path_feature_table_name[:128]
                print('只能保留表名的前128位')
            print('输出表名为', tmp_meta_path_feature_table_name)

            # 获取全部特征列列名
            tmp_groupby_feature_name_list = (['Start_Column'] + 
                                   [x + '_' + tmp_add_feature_weight_column_name for x in ['AVG', 'SUM', 'MAX', 'MIN']] +
                                   ['Groupby_COUNT'])

            # 特征表注释
            tmp_groupby_feature_comment_list = tmp_groupby_feature_name_list.copy()
            tmp_groupby_feature_comment_list[0] = Aim_Node_type + '_UID'

            # 获取对应列的数据
            tmp_meta_path_feature_for_upload = tmp_sub_meta_path_feature_groupby_result_rdd.select(tmp_groupby_feature_name_list)
            tmp_meta_path_feature_for_upload = tmp_meta_path_feature_for_upload.persist()
            
            # 获取各列的类型
            tmp_groupby_feature_type_list = []
            for _, col_type in tmp_meta_path_feature_for_upload.dtypes: 
                tmp_groupby_feature_type_list.append(col_type)
            
            # 创建表（如果特征表已存在，会自动不进行创建）
            Pyspark_Create_Table(Spark_Session, tmp_meta_path_feature_table_name, tmp_groupby_feature_name_list, 
                          tmp_groupby_feature_type_list, tmp_groupby_feature_comment_list)

            # 设定临时view的名称
            tmp_view_name = tmp_meta_path_feature_table_name.split('tmp.tmp___')[1]

            # 创建临时view
            tmp_meta_path_feature_for_upload.createTempView(tmp_view_name)

            # 生成upload对应的dt
            if Aim_Relation_Table_dt == Aim_Feature_Table_dt:
                tmp_up_load_table_dt = Aim_Relation_Table_dt
            else:
                tmp_up_load_table_dt = Aim_Relation_Table_dt + '___' + Aim_Feature_Table_dt

            # 上传特征（会先清空对应dt的数据）
            sql_str = """insert overwrite table """ + tmp_meta_path_feature_table_name + """ 
                         partition(dt='""" + tmp_up_load_table_dt + """')(
                         select * from """ + tmp_view_name + """
                        )    
                        """

            Spark_Session.sql(sql_str)

            # 删除临时view
            Spark_Session.catalog.dropTempView(tmp_view_name)

            groupby_upload_end_time = datetime.now()

            print('完成权重对应groupby的特征表的上传, 运算时间为:', (groupby_upload_end_time - groupby_upload_start_time))
            
            if Output_Summary:
                # 统计权重表的信息
                tmp_feature_table_summary_rdd = tmp_meta_path_feature_for_upload.drop('Start_Column').summary("min", "max", "mean", "stddev")

                # 设置上传对应的表名
                tmp_meta_path_feature_summary_table_name = ('tmp.tmp___JL_MP_W_Feat___' + tmp_relation_name_key + '___' + 
                                              tmp_add_feature_column_name + '___' + tmp_add_feature_weight_column_name + 
                                              '___Groupby_Summary___' + Table_Name_Comment)
                if len(tmp_meta_path_feature_summary_table_name) > 128:
                    tmp_meta_path_feature_summary_table_name = tmp_meta_path_feature_summary_table_name[:128]
                    print('只能保留表名的前128位')

                print('输出总结表名为', tmp_meta_path_feature_summary_table_name)

                # 上传统计信息
                Upload_RDD_Data_to_Database(Spark_Session, tmp_meta_path_feature_summary_table_name, tmp_feature_table_summary_rdd,
                                   tmp_up_load_table_dt, [], [])
                print('完成统计信息的上传,耗时为:', (datetime.now() - groupby_upload_end_time))
            
            tmp_meta_path_feature_for_upload = tmp_meta_path_feature_for_upload.unpersist()
            
            Output_Weight_Feature_Table_list[0] = False
            print("----------------------------------------------------------------------------------------------")
        
        ###############################################################################################################################
        # 查看是否需要进行节点列特征的聚合
        if Output_Columns_Type in ['ALL_Nodes_And_Edges', 'Head_And_Tail', 'ALL_Nodes']:
            print('为元路径中的节点列:', tmp_add_feature_column_name, '添加特征')

            # 获取对应的节点类型
            tmp_add_feature_node_type = tmp_column_name_to_class_dict[tmp_add_feature_column_name]
            print('该列的节点类型为:', tmp_add_feature_node_type)

            # 只保留起始列和目标列（不需要去重，因为多次出现就应该给更大的权重）
            tmp_sub_meta_path_result_rdd = tmp_meta_path_result_rdd.select(['Start_Column', tmp_add_feature_column_name])

            # 确保特征列的列名为tmp_node_class + '_UID'
            tmp_aim_path_result_rdd = tmp_sub_meta_path_result_rdd.withColumnRenamed(tmp_add_feature_column_name, 
                                                                      tmp_add_feature_node_type + '_UID')
            tmp_aim_path_result_rdd = tmp_aim_path_result_rdd.persist()
            
            # 只保留需要添加特征的点
            tmp_UID_for_add_feature = tmp_aim_path_result_rdd.select([tmp_add_feature_node_type + '_UID']).distinct().persist()
            
            # 要返回的特征
            tmp_all_feature_table_rdd = tmp_UID_for_add_feature
            
            # 查看要添加特征的点的数目
            tmp_UID_for_add_feature_counts = tmp_UID_for_add_feature.count()
            print('要添加特征的点的数目:', tmp_UID_for_add_feature_counts)
            
            # 如果数目少于100万就进行广播
            if tmp_UID_for_add_feature_counts < 1000000:
                print('数据量较小，进行广播，加快运算')
                
                # broadcast目标列，加快运算速度
                tmp_UID_for_add_feature_Broadcast = broadcast(tmp_UID_for_add_feature)
            
            ##############################################################################################
            # 保存该节点涉及的全部特征
            tmp_all_useful_feature_cols_list = []

            # 保存该节点涉及的全部特征的注释
            tmp_all_useful_feature_cols_comments_list = []

            # 基于全部的节点，依次提取全部的特征表中对应的数据，
            for tmp_feature_table_Info_i in range(Feature_Table_Processed_Count_list[0], 
                                      len(Feature_Dataset_Config_dict[tmp_add_feature_node_type]['Feature_Data_List'])):
                tmp_add_feature_start_time = datetime.now()
                
                tmp_feature_table_Info_dict = Feature_Dataset_Config_dict[tmp_add_feature_node_type]['Feature_Data_List'][tmp_feature_table_Info_i]
                
                tmp_feature_table_name = tmp_feature_table_Info_dict['Table_Name']
                print('开始处理特征表:', tmp_feature_table_name)
                
                if "Simple_Table_Name" in tmp_feature_table_Info_dict:
                    tmp_simple_feature_table_name = tmp_feature_table_Info_dict["Simple_Table_Name"]
                else:
                    tmp_simple_feature_table_name = tmp_feature_table_name.split('.')[-1]
                
                tmp_aim_column_name = tmp_feature_table_Info_dict['UID']

                tmp_sql_command = """
                            SELECT
                                *
                            FROM
                                """ + tmp_feature_table_name + """
                            WHERE 
                                """ + tmp_aim_column_name +""" IS NOT NULL AND
                                dt = '""" + Aim_Feature_Table_dt + """'
                            """

                tmp_feature_table_rdd = Spark_Session.sql(tmp_sql_command)

                # 确保添加特征目标列的列名为tmp_node_class + '_UID'
                tmp_feature_table_rdd = tmp_feature_table_rdd.withColumnRenamed(tmp_aim_column_name, 
                                                           tmp_add_feature_node_type + '_UID')

                # 通过persist保留计算结果
                tmp_feature_table_rdd = tmp_feature_table_rdd.persist()

                tmp_feature_table_rdd_raw_count = tmp_feature_table_rdd.count()

                if tmp_feature_table_rdd_raw_count == 0:
                    print('Error: 特征表', tmp_feature_table_name, '为空，得及时处理')
                else:
                    # 对特征表去重（理论上不需要，但防止有不符合规范的表）
                    tmp_feature_table_rdd = tmp_feature_table_rdd.dropDuplicates([tmp_add_feature_node_type + '_UID'])

                    # 通过persist保留计算结果
                    tmp_feature_table_rdd = tmp_feature_table_rdd.persist()

                    tmp_feature_table_rdd_count = tmp_feature_table_rdd.count()

                    if tmp_feature_table_rdd_raw_count != tmp_feature_table_rdd_count:
                        print('Error: 特征表特征表', tmp_feature_table_name, '在时间', Aim_Feature_Table_dt, 
                            '内部有重复UID，得及时修改, 目前先保留第一条信息，原始行数为:', tmp_feature_table_rdd_raw_count,
                             '去重后为:', tmp_feature_table_rdd_count)

                # 获取特征表的格式信息
                tmp_feature_table_rdd_json = json.loads(tmp_feature_table_rdd.schema.json())['fields']

                # 记录有效的特征列名
                tmp_useful_feature_cols_list = []

                # 记录有效的特征注释名
                tmp_useful_feature_cols_comments_list = []

                # 只保留其中有效的列（entity_id加数值类型的列，*待优化，数据格式标准化后改为entity_id及其之后的列）
                for tmp_col_info in tmp_feature_table_rdd_json:
                    col = tmp_col_info['name']
                    col_type = tmp_col_info['type']

                    if col == (tmp_add_feature_node_type + '_UID'):
                        continue

                    if 'entity_id' in col:
                        continue
                    
                    if col_type in ['int', 'integer', 'float', 'bigint','double', 'long']:
                        tmp_transferred_column_name = (col + '___' + tmp_simple_feature_table_name)
                        
                        if 'comment' in tmp_col_info['metadata']:
                            col_comment = (tmp_col_info['metadata']['comment'] + '___' + tmp_simple_feature_table_name)
                        else:
                            col_comment = tmp_transferred_column_name
                        
                        useless_name_str_list = ['tmp_KG_graph_table_feature_of_',
                                         'tmp___jy_KG_Graph_Meta_Path_Feature_Table___', 
                                         'tmp___JingLian_KG_Graph_Meta_Path_Feature_Table___',
                                         'tmp___JingLian_Meta_Path_Feature_Table___',
                                         'tmp___JingLian_Meta_Path_Feature___',
                                         'tmp_JL_Meta_Path_Feature_Table_']
                        
                        # 统一小写
                        tmp_transferred_column_name = tmp_transferred_column_name.lower()
                        col_comment = col_comment.lower()
                        
                        for tmp_useless_name_str in useless_name_str_list:
                            tmp_useless_name_str = tmp_useless_name_str.lower()
                            
                            # 删去列名中的无效字符
                            tmp_transferred_column_name = tmp_transferred_column_name.replace(tmp_useless_name_str, '')

                            # 删去注释名中的无效字符
                            col_comment = col_comment.replace(tmp_useless_name_str, '')
                        
                        if (tmp_feature_table_Info_i == Feature_Table_Processed_Count_list[0] and 
                           Feature_Table_Processed_Column_Name_list[0] != None and 
                           tmp_transferred_column_name == Feature_Table_Processed_Column_Name_list[0]):
                            tmp_useful_feature_cols_list = []

                            tmp_useful_feature_cols_comments_list = []
                        
                        tmp_feature_table_rdd = tmp_feature_table_rdd.withColumnRenamed(col, tmp_transferred_column_name)

                        tmp_useful_feature_cols_list.append(tmp_transferred_column_name)

                        tmp_useful_feature_cols_comments_list.append(col_comment)
                    elif col_type != 'string':
                        print('-----------------------------------------------------------')
                        print('WARNING:stange_type:', col, col_type)
                        print('-----------------------------------------------------------')

                tmp_feature_table_rdd = tmp_feature_table_rdd.select([tmp_add_feature_node_type + '_UID'] + 
                                                    tmp_useful_feature_cols_list)

                tmp_all_useful_feature_cols_list.extend(tmp_useful_feature_cols_list)
                tmp_all_useful_feature_cols_comments_list.extend(tmp_useful_feature_cols_comments_list)

                print('特征表'+ tmp_feature_table_name + '添加特征数:', len(tmp_useful_feature_cols_list))

                # 通过persist保留计算结果
#                 tmp_feature_table_rdd = tmp_feature_table_rdd.persist()
    
                # 查看各个特征列是否有都为同一个值的情况
                if tmp_feature_table_rdd_raw_count != 0:
                    # 计算除了UID列的min, max, mean, std，并转化为pandas
                    tmp_feature_table_summary_pd = tmp_feature_table_rdd.drop(tmp_add_feature_node_type + '_UID').summary("min", "max", "mean", "stddev").toPandas()

                    # 查看是否有无效列(特征都为同一值)，及时提醒
                    tmp_summary_min = tmp_feature_table_summary_pd[tmp_feature_table_summary_pd['summary'] == 'min'].values[0]
                    tmp_summary_max = tmp_feature_table_summary_pd[tmp_feature_table_summary_pd['summary'] == 'max'].values[0]

                    tmp_problem_columns = np.array(tmp_feature_table_summary_pd.columns)[tmp_summary_min == tmp_summary_max]

                    if tmp_problem_columns.shape[0] > 0:
                        print('ERROR: 特征表', tmp_feature_table_name, '在时间', Aim_Feature_Table_dt, 
                        '存在一些列的全部行都是一个值，具体情况如下，得及时修改')
                        print(dict(tmp_feature_table_summary_pd[tmp_problem_columns].iloc[0]))

                # 只保留需要的特征
                if tmp_UID_for_add_feature_counts < 1000000:
                    tmp_sub_feature_table_rdd = tmp_feature_table_rdd.join(tmp_UID_for_add_feature_Broadcast, 
                                                         tmp_add_feature_node_type + '_UID', 'inner')
                    
                    # broadcast目标特征，加快运算速度
                    tmp_sub_feature_table_rdd = broadcast(tmp_sub_feature_table_rdd)
                    
                else:
                    tmp_sub_feature_table_rdd = tmp_feature_table_rdd.join(tmp_UID_for_add_feature, 
                                                         tmp_add_feature_node_type + '_UID', 'inner')
                
                # 合并各个特征表的数据
                tmp_all_feature_table_rdd = tmp_all_feature_table_rdd.join(tmp_sub_feature_table_rdd, 
                                                        tmp_add_feature_node_type + '_UID', 'left')
                
                ##############################################################################################
                # 如果累计的特征列已经超过目标长度，或全部的特征表已经读取完毕，则进行上传
                if ((Max_Column_Number > 0 and len(tmp_all_useful_feature_cols_list) >= Max_Column_Number) or 
                   (tmp_feature_table_Info_i == (len(Feature_Dataset_Config_dict[tmp_add_feature_node_type]['Feature_Data_List']) - 1))):
                    # 如果Max_Column_Number没有值，则上传数设置为len(tmp_all_useful_feature_cols_list)
                    if Max_Column_Number > 0:
                        tmp_upload_column_max_count = Max_Column_Number
                    else:
                        tmp_upload_column_max_count = len(tmp_all_useful_feature_cols_list)
                    
                    # 要上传的次数
                    if tmp_feature_table_Info_i == (len(Feature_Dataset_Config_dict[tmp_add_feature_node_type]['Feature_Data_List']) - 1):
                        tmp_all_upload_range_count = math.ceil(len(tmp_all_useful_feature_cols_list)/tmp_upload_column_max_count)
                    else:
                        tmp_all_upload_range_count = len(tmp_all_useful_feature_cols_list)//tmp_upload_column_max_count
                    
                    # 如果本次要上传的列数超出了最大范围，则对tmp_all_feature_table_rdd进行切割，只保留限定范围内的数据
                    for tmp_upload_count in range(tmp_all_upload_range_count):
                        # 先保留剩下的特征列的相关信息
                        tmp_rest_useful_feature_cols_list = tmp_all_useful_feature_cols_list[tmp_upload_column_max_count:]
                        tmp_rest_useful_feature_cols_comments_list = tmp_all_useful_feature_cols_comments_list[tmp_upload_column_max_count:]
                        tmp_rest_feature_table_rdd = tmp_all_feature_table_rdd.select([tmp_add_feature_node_type + '_UID'] + 
                                                                  tmp_rest_useful_feature_cols_list)
                        
                        # 再对原始信息进行切割
                        tmp_all_useful_feature_cols_list = tmp_all_useful_feature_cols_list[:tmp_upload_column_max_count]
                        tmp_all_useful_feature_cols_comments_list = tmp_all_useful_feature_cols_comments_list[:tmp_upload_column_max_count]
                        tmp_all_feature_table_rdd = tmp_all_feature_table_rdd.select([tmp_add_feature_node_type + '_UID'] + 
                                                                  tmp_all_useful_feature_cols_list)
                        
                        print("要上传特征数", len(tmp_all_useful_feature_cols_list), 
                             "剩余待上传特征数:", len(tmp_rest_useful_feature_cols_list))
                        
                        Upload_Aggregated_Feature_and_Summary(Spark_Session, tmp_aim_path_result_rdd, tmp_all_feature_table_rdd, 
                                                  tmp_add_feature_node_type, tmp_all_useful_feature_cols_list, 
                                                  tmp_all_useful_feature_cols_comments_list, Groupby_Type_List,
                                                  tmp_add_feature_start_time, Max_Column_Number, Table_Name_Comment, 
                                                  Aim_Node_type, Feature_Table_Upload_Count_list, tmp_relation_name_key, 
                                                  tmp_add_feature_column_name, Aim_Relation_Table_dt, Aim_Feature_Table_dt,
                                                  Output_Summary)
                        
                        # 更新剩余要上传的信息
                        tmp_all_useful_feature_cols_list = tmp_rest_useful_feature_cols_list
                        tmp_all_useful_feature_cols_comments_list = tmp_rest_useful_feature_cols_comments_list
                        tmp_all_feature_table_rdd = tmp_rest_feature_table_rdd

                        # 更新要上传的表名
                        Feature_Table_Upload_Count_list[0] = Feature_Table_Upload_Count_list[0] + 1
                        
                        # 上传的表名中下一次上传的起始特征列名
                        if len(tmp_all_useful_feature_cols_list) > 0:
                            Feature_Table_Processed_Column_Name_list[0] = tmp_all_useful_feature_cols_list[0]
                        else:
                            Feature_Table_Processed_Column_Name_list[0] = None
                        
                        print('下次上传的起始列名:', Feature_Table_Processed_Column_Name_list[0])
                            
                    # 完成本次上传后更新处理到的表序号
                    if len(tmp_all_useful_feature_cols_list) > 0:
                        # 如果有剩余特征，如果有中断，就得从当前表重新开始处理
                        Feature_Table_Processed_Count_list[0] = tmp_feature_table_Info_i
                    else:
                        # 如果没有剩余特征，直接从下一个表开始算就行
                        Feature_Table_Processed_Count_list[0] = tmp_feature_table_Info_i + 1

                print("----------------------------------------------------------------------------------------------")
            
            Feature_Table_Processed_Count_list[0] = 0
            Feature_Table_Upload_Count_list[0] = 0
        
        # 恢复输出边相关特征
        Output_Weight_Feature_Table_list[0] = True
        
    ##############################################################################################    
    tmp_meta_path_result_start_node_rdd = tmp_meta_path_result_start_node_rdd.unpersist()
    tmp_meta_path_result_rdd = tmp_meta_path_result_rdd.unpersist()
    
    print('完成元路径特征' + tmp_relation_name_key + '的生成')
    print('##########################################################################################')
    
    return