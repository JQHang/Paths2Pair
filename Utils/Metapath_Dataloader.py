from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.sql.types import *

from Utils.Pyspark_utils import sample_random_n_samples_for_samll_rdd
from Utils.Pyspark_utils import sample_top_n_groupby_samples_for_samll_rdd
from Utils.Pyspark_utils import sample_random_n_groupby_samples_for_samll_rdd

import pandas as pd
import numpy as np
import os

"""
作用：
    针对目标节点和元路径配置文件，获取元路径，并将运算结果（元路径关系及各列对应的节点类型）保存在指定文件夹下（运算前会先检验文件是否已存在，若已存在则会跳过生成），同时会更新节点表中的节点信息，将新增节点添加到对应的节点文件中

输入：
    Spark_Session：pyspark接口
    tmp_aim_entity_info_dict：目标节点的相关信息（包含全量的目标节点，以及目标节点的类型和对应日期）
    Metapath_Config_dict：元路径配置文件
    tmp_meta_path_output_dir：元路径输出文件夹
    tmp_node_output_dir：涉及到的节点输出文件夹
    check_old_file:是否要重新生成文件

返回值：
    无返回值，运算结果都会保存在tmp_meta_path_output_dir和tmp_node_output_dir这两个文件夹下
"""
def get_meta_path(Spark_Session, tmp_aim_entity_info_dict, Metapath_Config_dict, tmp_meta_path_output_dir, tmp_node_output_dir,
             Output_Columns_Type = "Head_And_Tail"):
    Node_type = tmp_aim_entity_info_dict['Node_Type']
    Aim_entity_column_name = Node_type + '_UID'
    tmp_aim_table_dt = tmp_aim_entity_info_dict['Monthly_dt']
    
    # 只保留UID列
    tmp_aim_entity_UID_rdd = tmp_aim_entity_info_dict['Data'].select(Aim_entity_column_name).alias("tmp_aim_entity_UID_rdd")
    
    # 去个重，以防万一
    tmp_aim_entity_UID_rdd = tmp_aim_entity_UID_rdd.distinct()
    
    ##############################################################################################
    print('将目标节点存储至节点表')
    
    # 设定对应类型节点的文件名
    tmp_output_data_node_file = (tmp_node_output_dir + Node_type + '.pkl')

    # 查询是否已有相关文件
    if os.path.exists(tmp_output_data_node_file):
        # 如果之前已有对应节点表，则肯定已经保存过了，跳过
        print('WARNING: 已有对应节点表，一定已保存过目标点')
    else:
        # 先将目标点转pandas
        tmp_aim_entity_UID_pd = tmp_aim_entity_UID_rdd.toPandas()
        print('目标样本总数:', tmp_aim_entity_UID_pd.shape[0])
        
        # 如果没有，则直接存储
        tmp_aim_entity_UID_pd.to_pickle(tmp_output_data_node_file)
        print('完成目标节点的保存')
    
    ##############################################################################################
    print('输出列的选择方案：', Output_Columns_Type)
    
    # 遍历全部元路径
    for tmp_relation_name_key in Metapath_Config_dict:
        print('生成元路径：', tmp_relation_name_key)
        
        # 该元路径的信息
        tmp_meta_path_info_dict = Metapath_Config_dict[tmp_relation_name_key]
        
        ##############################################################################################
        # 设定对应类型元路径关系的文件名
        tmp_output_meta_path_edges_file = (tmp_meta_path_output_dir + tmp_relation_name_key + '.pkl')
        
        # 设定对应类型元路径各列节点类型的文件名
        tmp_output_meta_path_node_class_file = (tmp_meta_path_output_dir + tmp_relation_name_key + '-column_to_node_class.npy')
        
        # 先检查是否已经生成完对应数据，如果已生成完，则跳过
        if os.path.exists(tmp_output_meta_path_node_class_file):
            # 已存在则不进行存储，只打印相关信息，进行提示
            print('-----------------------------------------------------------')
            print('WARNING: 元路径关系:' + tmp_relation_name_key + '已存在，跳过生成')
            print('-----------------------------------------------------------')
            
            continue
        ##############################################################################################
        
        # 保存目标实体号副本
        tmp_meta_path_result_rdd = tmp_aim_entity_UID_rdd.alias("tmp_meta_path_result_rdd")
        
        # 获取要读取的元路径涉及的全部关系
        tmp_aim_meta_path_Relation_List = tmp_meta_path_info_dict['Relation_List']
        
        # 存储输出列对应的节点类型
        tmp_column_name_to_class_dict = {}
        # 起始点是一定会输出的
        tmp_column_name_to_class_dict[tmp_aim_meta_path_Relation_List[0]['Head_Column_name_AS']] = tmp_aim_meta_path_Relation_List[0]['Head_Node_class']
        
        # 存储目标输出列列名
        tmp_all_aim_column_name_list = [tmp_aim_meta_path_Relation_List[0]['Head_Column_name_AS']]
        
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

            # 设置表名和dt
            tmp_sql_command = tmp_sql_command + """\nFROM\n    """ + tmp_aim_meta_path_sub_Relation['Relation_Data']
            tmp_sql_command = tmp_sql_command + """\nWHERE\n    dt = '""" + tmp_aim_table_dt + """'"""

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
            
            # 只保留涉及到的关系
            tmp_column_for_relation_to_join = tmp_aim_meta_path_sub_Relation['Head_Column_name_AS']
            tmp_aim_relation_rdd = tmp_aim_relation_rdd.join(tmp_meta_path_result_rdd.select([tmp_column_for_relation_to_join]).distinct(),
                                             tmp_column_for_relation_to_join, 'inner')
            
            # 对保留的关系去重（理论上不需要，但防止意外）
            tmp_aim_relation_rdd = tmp_aim_relation_rdd.dropDuplicates()
            
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
                                    dt = '""" + tmp_aim_table_dt + """'
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
            
            # 根据要求，保证每个样本对应关系数目不超出范围(目前基于window functions的方案)
            # https://stackoverflow.com/questions/33749965/choosing-random-items-from-a-spark-groupeddata-object
            if 'Max_Sample_Scale' in tmp_aim_meta_path_sub_Relation and tmp_aim_meta_path_sub_Relation['Max_Sample_Scale'] > 0:
                if 'Max_Sample_Type' in tmp_aim_meta_path_sub_Relation:
                    if tmp_aim_meta_path_sub_Relation['Max_Sample_Type'] == 'Random':
                        print('随机采样,最多保留', tmp_aim_meta_path_sub_Relation['Max_Sample_Scale'])
                        tmp_aim_relation_rdd = sample_random_n_groupby_samples_for_samll_rdd(Spark_Session, 
                                                                      tmp_aim_relation_rdd, 
                                                                 tmp_aim_meta_path_sub_Relation['Head_Column_name_AS'], 
                                                                 tmp_aim_meta_path_sub_Relation['Max_Sample_Scale'])
                    elif tmp_aim_meta_path_sub_Relation['Max_Sample_Type'] == 'TopN':
                        print('TopN采样,最多保留', tmp_aim_meta_path_sub_Relation['Max_Sample_Scale'])
                        tmp_aim_relation_rdd = sample_top_n_groupby_samples_for_samll_rdd(Spark_Session, tmp_aim_relation_rdd, 
                                                                  tmp_aim_meta_path_sub_Relation['Head_Column_name_AS'], 
                                                                    tmp_aim_meta_path_sub_Relation['Max_Sample_Scale'],
                                                                    tmp_Weight_Column_AS)
                else:
                    print('随机采样,最多保留', tmp_aim_meta_path_sub_Relation['Max_Sample_Scale'])
                    tmp_aim_relation_rdd = sample_random_n_groupby_samples_for_samll_rdd(Spark_Session, tmp_aim_relation_rdd, 
                                                                  tmp_aim_meta_path_sub_Relation['Head_Column_name_AS'], 
                                                                  tmp_aim_meta_path_sub_Relation['Max_Sample_Scale'])
            
            # join两个表
            tmp_meta_path_result_rdd = tmp_meta_path_result_rdd.join(tmp_aim_relation_rdd, tmp_aim_meta_path_sub_Relation['Head_Column_name_AS'], 
                                                  "inner")
            
            # 如果有join后的limit，则按要求执行
            if 'Limits_After_Join' in tmp_aim_meta_path_sub_Relation and tmp_aim_meta_path_sub_Relation['Limits_After_Join'] != '':
                tmp_meta_path_result_rdd = tmp_meta_path_result_rdd.where(tmp_aim_meta_path_sub_Relation['Limits_After_Join'])
                print('join关系表后限制条件为', tmp_aim_meta_path_sub_Relation['Limits_After_Join'])
            
            # 记录各列对应的节点类型，以及涉及到的全部列名
            if 'Tail_Column_name' in tmp_aim_meta_path_sub_Relation:
                tmp_column_name_to_class_dict[tmp_aim_meta_path_sub_Relation['Tail_Column_name_AS']] = tmp_aim_meta_path_sub_Relation['Tail_Node_class']

                tmp_all_aim_column_name_list.append(tmp_aim_meta_path_sub_Relation['Tail_Column_name_AS'])

            elif 'Tail_Column_name_list' in tmp_aim_meta_path_sub_Relation:
                for tmp_tail_column_i in range(len(tmp_aim_meta_path_sub_Relation['Tail_Column_name_AS_list'])):
                    tmp_tail_column_i_name = tmp_aim_meta_path_sub_Relation['Tail_Column_name_AS_list'][tmp_tail_column_i]

                    tmp_column_name_to_class_dict[tmp_tail_column_i_name] = tmp_aim_meta_path_sub_Relation['Tail_Node_class_list'][tmp_tail_column_i]

                    tmp_all_aim_column_name_list.append(tmp_tail_column_i_name)

        # 如果设置了终止列，则删去终止列之后的列名
        if 'Tail_Column' in tmp_meta_path_info_dict:
            tmp_tail_column_index = tmp_all_aim_column_name_list.index(tmp_meta_path_info_dict['Tail_Column'])
            tmp_all_aim_column_name_list = tmp_all_aim_column_name_list[:(tmp_tail_column_index + 1)]
        
        # 如果只要头尾节点，则删去中间的列
        if Output_Columns_Type == 'Head_And_Tail':
            tmp_all_aim_column_name_list = [tmp_all_aim_column_name_list[0], tmp_all_aim_column_name_list[-1]]
        elif Output_Columns_Type == 'ALL_Nodes':
            # 如果只要节点，则删去权重列
            tmp_all_aim_column_name_list = [x for x in tmp_all_aim_column_name_list if 'Weight_' not in x]
        elif Output_Columns_Type == 'ALL_Edges':
            # 如果只要权重列，则删去节点列
            tmp_all_aim_column_name_list = [tmp_all_aim_column_name_list[0]] + [x for x in tmp_all_aim_column_name_list if 'Weight_' in x]
        
        print('完成元路径' + tmp_relation_name_key + '的生成，最终包含的列为:', tmp_all_aim_column_name_list)
        
        # 只用保留需要的列
        tmp_meta_path_result_rdd = tmp_meta_path_result_rdd.select(tmp_all_aim_column_name_list)
        
        # 对保留的结果去重
        tmp_meta_path_result_rdd = tmp_meta_path_result_rdd.dropDuplicates()
        
        # 如果采样完成后还要进行一次压缩
        if 'Max_All_Sample_Scale' in tmp_meta_path_info_dict:
            # 则直接进行随机采样
            tmp_meta_path_start_column = tmp_aim_meta_path_Relation_List[0]['Head_Column_name_AS']
            tmp_result_max_scale = tmp_meta_path_info_dict['Max_All_Sample_Scale']
            tmp_meta_path_result_rdd = sample_random_n_groupby_samples_for_samll_rdd(Spark_Session, tmp_meta_path_result_rdd, 
                                                             tmp_meta_path_start_column, 
                                                             tmp_result_max_scale)
        
        # 完成元路径生成，将其转换为pandas
        tmp_meta_path_result_pandas = tmp_meta_path_result_rdd.toPandas()
        print('元路径：' + tmp_relation_name_key + '最终使用总行数为:', tmp_meta_path_result_pandas.shape[0])
        
        ##############################################################################################
        # 保存该元路径涉及的全部节点UID（可以跳过起始点，因为已保存）
        for tmp_column_name in tmp_all_aim_column_name_list[1:]:
            tmp_node_class = tmp_column_name_to_class_dict[tmp_column_name]
            
            # 去重+更新列名
            tmp_node_UID_to_add_pandas = tmp_meta_path_result_pandas[[tmp_column_name]].drop_duplicates([tmp_column_name], ignore_index = True)
            tmp_node_UID_to_add_pandas = tmp_node_UID_to_add_pandas.rename(columns={tmp_column_name: (tmp_node_class + '_UID')})
            
            # 设定对应类型节点的文件名
            tmp_output_data_node_file = (tmp_node_output_dir + tmp_node_class + '.pkl')

            # 查询是否已有相关文件
            if os.path.exists(tmp_output_data_node_file):
                # 如果之前已有对应节点表，则进行合并
                tmp_past_node_pd = pd.read_pickle(tmp_output_data_node_file)

                # 先拼接
                tmp_node_UID_to_add_pandas = pd.concat([tmp_past_node_pd, tmp_node_UID_to_add_pandas])

                # 再去重
                tmp_node_UID_to_add_pandas = tmp_node_UID_to_add_pandas.drop_duplicates([tmp_node_class + '_UID'], ignore_index = True)

            # 进行存储
            tmp_node_UID_to_add_pandas.to_pickle(tmp_output_data_node_file)

            print(tmp_node_class + '节点最新数目:', tmp_node_UID_to_add_pandas.shape[0])
             
            print('完成列' + tmp_column_name + '涉及到的' + tmp_node_class + '类型的节点的保存')
            
        ##############################################################################################
        # 先存储关系表，再存储各节点类型
        tmp_meta_path_result_pandas.to_pickle(tmp_output_meta_path_edges_file)
        np.save(tmp_output_meta_path_node_class_file, tmp_column_name_to_class_dict)
        
        ##############################################################################################
        print('完成元路径' + tmp_relation_name_key + '的生成')
        print('##########################################################')
    
    return
