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

from Utils.utils import mkdir
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
    tmp_relation_name_key：
    Aim_Relation_Table_dt：元路径输出文件夹
    Table_Name_Comment:是否要重新生成文件

返回值：
    无返回值，运算结果都会保存在tmp_meta_path_output_dir和tmp_node_output_dir这两个文件夹下
"""
def ComplexPath_Node_Query(Spark_Session, Metapath_Feature_Config_dict, tmp_relation_name_key, Aim_Relation_Table_dt, Table_Name_Comment, 
                   Upload_to_BDP = False):
    print('生成复杂路', tmp_relation_name_key, '对应的目标点')
    
    # 该元路径的信息
    tmp_meta_path_info_dict = Metapath_Feature_Config_dict[tmp_relation_name_key]
    
    # 获取要读取的元路径涉及的全部关系
    tmp_aim_meta_path_Relation_List = tmp_meta_path_info_dict['Relation_List']

    # 元路径起始列就是目标节点和目标列
    Aim_Node_type = tmp_aim_meta_path_Relation_List[0]['Head_Node_class']

    Aim_Meta_Path_Start_Column = tmp_aim_meta_path_Relation_List[0]['Head_Column_name_AS']
    
    # 表的注释
    if "Table_Name_Comment" in tmp_meta_path_info_dict:
        Table_Name_Comment = tmp_meta_path_info_dict["Table_Name_Comment"]
    print("Table_Name_Comment:", Table_Name_Comment)
    
    # 一次性可处理的最大特征列数
    if "Max_Column_Number" in tmp_meta_path_info_dict:
        Max_Column_Number = tmp_meta_path_info_dict["Max_Column_Number"]
    else:
        Max_Column_Number = -1
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
    
    print('元路径' + tmp_relation_name_key + '最终包含的列为:', tmp_all_aim_column_name_list)

    # 如果采样完成后还要进行一次压缩
    if 'Max_All_Sample_Scale' in tmp_meta_path_info_dict:
        # 则直接进行随机采样
        tmp_meta_path_start_column = tmp_aim_meta_path_Relation_List[0]['Head_Column_name_AS']
        tmp_result_max_scale = tmp_meta_path_info_dict['Max_All_Sample_Scale']
        tmp_meta_path_result_rdd = sample_random_n_groupby_samples_for_samll_rdd(Spark_Session, tmp_meta_path_result_rdd, 
                                                         tmp_meta_path_start_column, 
                                                         tmp_result_max_scale)
    
    # 只取出结果复杂路的起始列
    tmp_meta_path_result_start_column_rdd = tmp_meta_path_result_rdd.select(Aim_Meta_Path_Start_Column)

    # 对起始列去重
    tmp_meta_path_result_start_node_rdd = tmp_meta_path_result_start_column_rdd.distinct()
    
    # 通过persist保留元路径计算结果
    tmp_meta_path_result_start_node_rdd = tmp_meta_path_result_start_node_rdd.persist()
    
#     # 元路径总点数
#     tmp_start_node_counts = tmp_meta_path_result_start_node_rdd.count()
#     print('起始点总点数:', tmp_start_node_counts)
    
    #################################################################################################################################
    if Upload_to_BDP:
        # 设置对应表名
        tmp_target_node_table_name = ('tmp.tmp___JL_Target_Node___' + tmp_relation_name_key + '___' + Table_Name_Comment)
        if len(tmp_target_node_table_name) > 128:
            tmp_target_node_table_name = tmp_target_node_table_name[:128]
            print('只能保留表名的前128位')
        print('输出表名为', tmp_target_node_table_name)

        # 上传结果
        Upload_RDD_Data_to_Database(Spark_Session, tmp_target_node_table_name, tmp_meta_path_result_start_node_rdd, Aim_Relation_Table_dt, 
                           [], [])
    
    #################################################################################################################################
    tmp_meta_path_result_start_node_pd = tmp_meta_path_result_start_node_rdd.toPandas()
    
    print('目标节点数:', tmp_meta_path_result_start_node_pd.shape[0])
    
    mkdir('../../Target_Node/')
    mkdir('../../Target_Node/' + Table_Name_Comment + '/')
    tmp_meta_path_result_start_node_pd.to_pickle('../../Target_Node/' + Table_Name_Comment + '/' + tmp_relation_name_key + '.pkl')
    
    #################################################################################################################################
    tmp_meta_path_result_start_node_rdd = tmp_meta_path_result_start_node_rdd.unpersist()
    
    print('完成召回源' + tmp_relation_name_key + '的生成')
    print('##########################################################################################')
    
    return