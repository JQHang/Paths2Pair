from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.sql.types import *

from kg_lib.Pyspark_utils import sample_random_n_samples_for_samll_rdd
from kg_lib.Pyspark_utils import sample_top_n_groupby_samples_for_samll_rdd
from kg_lib.Pyspark_utils import sample_random_n_groupby_samples_for_samll_rdd
from kg_lib.utils import mkdir

import pandas as pd
import numpy as np
import os

"""
作用：
    针对目标节点和subgraph配置文件，获取subgraph，并将运算结果（采样后的关系）保存在指定文件夹下（运算前会先检验文件是否已存在，若已存在则会跳过生成），同时会更新节点表中的节点信息，将新增节点添加到对应的节点文件中

输入：
    Spark_Session：pyspark接口
    tmp_aim_entity_info_dict：目标节点的相关信息（包含全量的目标节点，以及目标节点的类型和对应日期）
    Subgraph_Config_dict：subgraph配置文件
    tmp_subgraph_hop: 采样多少跳的subgraph
    tmp_subgraph_output_dir：subgraph输出文件夹
    tmp_node_output_dir：涉及到的节点输出文件夹

返回值：
    无返回值，运算结果都会保存在tmp_subgraph_output_dir和tmp_node_output_dir这两个文件夹下
"""
def get_sub_graph(Spark_Session, tmp_aim_entity_info_dict, Subgraph_Config_dict, tmp_subgraph_hop, tmp_subgraph_output_dir, tmp_node_output_dir):
    Aim_Node_type = tmp_aim_entity_info_dict['Node_Type']
    Aim_entity_column_name = Aim_Node_type + '_UID'
    tmp_aim_table_dt = tmp_aim_entity_info_dict['Monthly_dt']
    
    # 只保留UID列
    tmp_aim_entity_UID_rdd = tmp_aim_entity_info_dict['Data'].select(Aim_entity_column_name).alias("tmp_aim_entity_UID_rdd")
    
    # 去个重，以防万一
    tmp_aim_entity_UID_rdd = tmp_aim_entity_UID_rdd.distinct()
    
    # seed_node保存文件夹
    tmp_subgraph_seed_node_output_dir = tmp_subgraph_output_dir + 'Seed_Node/'
    mkdir(tmp_subgraph_seed_node_output_dir)
    
    # 边关系保存文件夹
    tmp_subgraph_edges_output_dir = tmp_subgraph_output_dir + 'Edges/'
    mkdir(tmp_subgraph_edges_output_dir)
    
    ##############################################################################################
    print('将目标节点存储至节点表并设为第0跳的seed_nodes')
    
    # 设定对应类型节点的文件名
    tmp_output_data_node_file = (tmp_node_output_dir + Aim_Node_type + '.pkl')

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
        
        tmp_subgraph_seed_node_hop_output_dir = tmp_subgraph_seed_node_output_dir + '0/'
        mkdir(tmp_subgraph_seed_node_hop_output_dir)
        
        tmp_aim_entity_UID_pd.to_pickle(tmp_subgraph_seed_node_hop_output_dir + Aim_Node_type + '.pkl')
        print('完成第0跳的seed_nodes的保存')
    ##############################################################################################    
    # 遍历各层hop
    for tmp_hop in range(tmp_subgraph_hop):
        print('-----------------------------------------------------------')
        print('开始处理第', tmp_hop, '跳的关系')
        
        # 该跳对应的seed_node保存文件夹(该文件夹一定已存在)
        tmp_subgraph_seed_node_current_hop_output_dir = tmp_subgraph_seed_node_output_dir + str(tmp_hop) + '/'
        
        # 该跳对应的边关系保存文件夹
        tmp_subgraph_edges_current_hop_output_dir = tmp_subgraph_edges_output_dir + str(tmp_hop) + '/'
        mkdir(tmp_subgraph_edges_current_hop_output_dir)
        
        ##############################################################################################
        # 读取该跳对应的seed_node，并上传为rdd
        tmp_seed_node_table_dict = {}
        
        # 获取生成好的全部seed_node文件名
        tmp_seed_node_file_list = os.listdir(tmp_subgraph_seed_node_current_hop_output_dir)
        tmp_seed_node_file_list = [tmp_file_name for tmp_file_name in tmp_seed_node_file_list if '.pkl' in tmp_file_name]
        
        for tmp_seed_node_file in tmp_seed_node_file_list:
            tmp_seed_node_type = tmp_seed_node_file.split('.pkl')[0]
            tmp_seed_node_pd = pd.read_pickle(tmp_subgraph_seed_node_current_hop_output_dir + tmp_seed_node_file)
            
            tmp_node_table_schema = StructType([StructField(tmp_seed_node_type + '_UID', StringType(), True)])
            tmp_seed_node_rdd = Spark_Session.createDataFrame(tmp_seed_node_pd, tmp_node_table_schema)
            
            tmp_seed_node_table_dict[tmp_seed_node_type] = tmp_seed_node_rdd
        
        ##############################################################################################
        # 遍历全部关系
        for tmp_relation_name_key in Subgraph_Config_dict:
            print('针对关系', tmp_relation_name_key ,'进行采样')

            # 该关系的采样信息
            tmp_subgraph_info_dict = Subgraph_Config_dict[tmp_relation_name_key]
            
            if ((len(tmp_subgraph_info_dict["Max_Sample_Scale_list"]) <= tmp_hop) or 
               (len(tmp_subgraph_info_dict["Reverse_Max_Sample_Scale_list"]) <= tmp_hop)):
                print('关系', tmp_relation_name_key, '超出最大跳数，跳过')
                continue
            
            # 查看是否需要采样
            tmp_head_Max_Sample_Scale = tmp_subgraph_info_dict["Max_Sample_Scale_list"][tmp_hop]
            tmp_tail_Max_Sample_Scale = tmp_subgraph_info_dict["Reverse_Max_Sample_Scale_list"][tmp_hop]
            if tmp_head_Max_Sample_Scale == 0 and tmp_tail_Max_Sample_Scale == 0:
                print('关系', tmp_relation_name_key, '在本跳中无需采样，跳过')
                continue
            tmp_Max_Sample_Scale_dict = {"Head": tmp_head_Max_Sample_Scale, "Tail": tmp_tail_Max_Sample_Scale}
            
            # 查看是否有可以对应上的列
            tmp_aim_head_or_tail_list = []

            if tmp_subgraph_info_dict['Head_Node_class'] in tmp_seed_node_table_dict and tmp_head_Max_Sample_Scale != 0:
                tmp_aim_head_or_tail_list.append('Head')
            if tmp_subgraph_info_dict['Tail_Node_class'] in tmp_seed_node_table_dict and tmp_tail_Max_Sample_Scale != 0:
                tmp_aim_head_or_tail_list.append('Tail')

            if len(tmp_aim_head_or_tail_list) == 0:
                print('关系', tmp_relation_name_key, '中涉及的节点都不在seed node中，跳过')
                continue

            # 设定各列对应的节点类型
            tmp_output_subgraph_edges_node_class_file = (tmp_subgraph_edges_current_hop_output_dir + tmp_relation_name_key +
                                          '-column_to_node_class.npy')

            # 通过各列对应的节点类型文件检查是否已经生成完对应数据，如果已生成完，则跳过
            if os.path.exists(tmp_output_subgraph_edges_node_class_file):
                # 已存在则不进行存储，只打印相关信息，进行提示
                print('INFO: subgraph关系:' + tmp_relation_name_key + '已存在，跳过生成')

                continue

            # 记录起始点和终止点的节点类型
            tmp_column_name_to_class_dict = {}
            tmp_column_name_to_class_dict['Head'] = tmp_subgraph_info_dict['Head_Node_class']
            tmp_column_name_to_class_dict['Tail'] = tmp_subgraph_info_dict['Tail_Node_class']

            # 读取该关系表中的头尾列
            tmp_sql_command = """
                        SELECT
                        """ + tmp_subgraph_info_dict['Head_Column_name'] + """ AS Head,
                        """ + tmp_subgraph_info_dict['Tail_Column_name'] + """ AS Tail"""

            # 如果有权重列的话就加上
            if 'Weight_Column' in tmp_subgraph_info_dict and tmp_subgraph_info_dict['Weight_Column'] != '':
                tmp_sql_command = tmp_sql_command + """,
                            """ + tmp_subgraph_info_dict['Weight_Column'] + """ AS Weight"""
            else:
                tmp_sql_command = tmp_sql_command + """,
                            1 AS Weight"""

            # 设置表名和dt
            tmp_sql_command = tmp_sql_command + """
                        FROM
                            """ + tmp_subgraph_info_dict['Relation_Data'] + """
                        WHERE 
                            dt = '""" + tmp_aim_table_dt + """'
                        """

            # 如果有限制条件的话就加上
            if 'Limits' in tmp_subgraph_info_dict and tmp_subgraph_info_dict['Limits'] != '':
                tmp_sql_command = tmp_sql_command + '\nAND ' + tmp_subgraph_info_dict['Limits']
                print('关系表限制条件为', tmp_subgraph_info_dict['Limits'])

            # 执行运算，获取目标关系表
            tmp_aim_relation_rdd = Spark_Session.sql(tmp_sql_command)

            # 针对涉及到的目标列进行处理
            for tmp_aim_head_or_tail in tmp_aim_head_or_tail_list:
                # 设定对应类型元路径关系的文件名
                tmp_output_subgraph_edges_file = (tmp_subgraph_edges_current_hop_output_dir + tmp_relation_name_key + '_Aim_' + 
                                       tmp_aim_head_or_tail + '.pkl')

                # 先检查是否已经生成完对应数据，如果已生成完，则跳过
                if os.path.exists(tmp_output_subgraph_edges_file):
                    # 已存在则不进行存储，只打印相关信息，进行提示
                    print('INFO: subgraph关系:' + tmp_output_subgraph_edges_file + '已存在，跳过生成')

                    continue

                # 针对已有的seed_node保留涉及到的关系
                tmp_aim_column_node_type = tmp_column_name_to_class_dict[tmp_aim_head_or_tail]
                tmp_aim_seed_node_rdd = tmp_seed_node_table_dict[tmp_aim_column_node_type].select([tmp_aim_column_node_type + '_UID'])
                tmp_aim_seed_node_rdd = tmp_aim_seed_node_rdd.withColumnRenamed(tmp_aim_column_node_type + '_UID', tmp_aim_head_or_tail)
                tmp_seed_node_aim_relation_rdd = tmp_aim_relation_rdd.join(tmp_aim_seed_node_rdd, tmp_aim_head_or_tail, 'inner')

                # 对保留的关系去重（理论上不需要，但防止意外）
                tmp_seed_node_aim_relation_rdd = tmp_seed_node_aim_relation_rdd.dropDuplicates()

                # 如果要拼接别的特征表的话就拼上指定列
                if 'Limits_Add_Feature_List' in tmp_subgraph_info_dict and len(tmp_subgraph_info_dict['Limits_Add_Feature_List']) > 0:
                    for tmp_add_feature_info_dict in tmp_subgraph_info_dict['Limits_Add_Feature_List']:
                        # 读取对应表
                        tmp_feature_table_name = tmp_add_feature_info_dict['Table_Name']
                        tmp_feature_UID_column_name = tmp_add_feature_info_dict['UID']
                        tmp_column_for_feature_to_join = tmp_add_feature_info_dict['UID_AS'] # 是和head join还是和tail join
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

                        # 对保留的特征去重
                        tmp_add_feature_table_rdd = tmp_add_feature_table_rdd.dropDuplicates()

                        # 将其加入 tmp_seed_node_aim_relation_rdd
                        tmp_seed_node_aim_relation_rdd = tmp_seed_node_aim_relation_rdd.join(tmp_add_feature_table_rdd, 
                                                                      tmp_column_for_feature_to_join, 'left')

                        print('添加来自' + tmp_feature_table_name + '的特征:', tmp_feature_aim_column_name_list)

                    # 根据添加的特征，按要求执行Limit
                    if 'Limits_To_Feature' in tmp_subgraph_info_dict and tmp_subgraph_info_dict['Limits_To_Feature'] != '':
                        tmp_seed_node_aim_relation_rdd = tmp_seed_node_aim_relation_rdd.where(tmp_subgraph_info_dict['Limits_After_Join'])
                        print('join特征表后限制条件为', tmp_subgraph_info_dict['Limits_After_Join'])

                # 根据要求，保证每个样本对应关系数目不超出范围(目前基于window functions的方案)
                tmp_Max_Sample_Scale = tmp_Max_Sample_Scale_dict[tmp_aim_head_or_tail]
                if tmp_Max_Sample_Scale > 0:
                    if 'Max_Sample_Type' in tmp_subgraph_info_dict:
                        if tmp_subgraph_info_dict['Max_Sample_Type'] == 'Random':
                            print('随机采样,最多保留', tmp_Max_Sample_Scale)
                            tmp_seed_node_aim_relation_rdd = sample_random_n_groupby_samples_for_samll_rdd(Spark_Session, 
                                                                                 tmp_seed_node_aim_relation_rdd, 
                                                                                 tmp_aim_head_or_tail, 
                                                                                 tmp_Max_Sample_Scale)
                        elif tmp_subgraph_info_dict['Max_Sample_Type'] == 'TopN':
                            print('TopN采样,最多保留', tmp_Max_Sample_Scale)
                            tmp_seed_node_aim_relation_rdd = sample_top_n_groupby_samples_for_samll_rdd(Spark_Session,
                                                                               tmp_seed_node_aim_relation_rdd, 
                                                                               tmp_aim_head_or_tail, 
                                                                               tmp_Max_Sample_Scale,
                                                                               'Weight')
                    else:
                        print('随机采样,最多保留', tmp_Max_Sample_Scale)
                        tmp_seed_node_aim_relation_rdd = sample_random_n_groupby_samples_for_samll_rdd(Spark_Session,
                                                                             tmp_seed_node_aim_relation_rdd, 
                                                                             tmp_aim_head_or_tail, 
                                                                             tmp_Max_Sample_Scale)

                # 只保留Head、Tail、Weight
                tmp_seed_node_aim_relation_rdd = tmp_seed_node_aim_relation_rdd.select(['Head', 'Tail', 'Weight'])

                # 完成关系的采样，将其转换为pandas
                tmp_seed_node_aim_relation_pandas = tmp_seed_node_aim_relation_rdd.toPandas()
                print('元路径：' + tmp_relation_name_key + '最终使用总行数为:', tmp_seed_node_aim_relation_pandas.shape[0])

                ##############################################################################################
                # 保存非目标列涉及的全部节点UID，并存储为下一跳的seed nodes
                if tmp_aim_head_or_tail == 'Head':
                    tmp_related_column_name = 'Tail'
                else:
                    tmp_related_column_name = 'Head'
                    
                tmp_related_node_class = tmp_column_name_to_class_dict[tmp_related_column_name]

                # 去重+更新列名
                tmp_node_UID_to_add_pandas = tmp_seed_node_aim_relation_pandas[[tmp_related_column_name]]
                tmp_node_UID_to_add_pandas = tmp_node_UID_to_add_pandas.drop_duplicates([tmp_related_column_name], ignore_index = True)
                tmp_node_UID_to_add_pandas = tmp_node_UID_to_add_pandas.rename(columns={tmp_related_column_name: (tmp_related_node_class + '_UID')})
                
                ##############################################################################################
                tmp_node_UID_pandas_for_node_table = tmp_node_UID_to_add_pandas.copy()
                
                # 设定对应类型节点的文件名
                tmp_output_data_node_file = (tmp_node_output_dir + tmp_related_node_class + '.pkl')

                # 查询是否已有相关文件
                if os.path.exists(tmp_output_data_node_file):
                    # 如果之前已有对应节点表，则进行合并
                    tmp_past_node_pd = pd.read_pickle(tmp_output_data_node_file)

                    # 先拼接
                    tmp_node_UID_pandas_for_node_table = pd.concat([tmp_past_node_pd, tmp_node_UID_pandas_for_node_table])

                    # 再去重
                    tmp_node_UID_pandas_for_node_table = tmp_node_UID_pandas_for_node_table.drop_duplicates([tmp_related_node_class + '_UID'], 
                                                                                ignore_index = True)

                # 进行存储
                tmp_node_UID_pandas_for_node_table.to_pickle(tmp_output_data_node_file)

                print(tmp_related_node_class + '节点最新数目:', tmp_node_UID_pandas_for_node_table.shape[0])

                ##############################################################################################
                tmp_node_UID_pandas_for_seed_node = tmp_node_UID_to_add_pandas.copy()
                
                # 添加入下一跳的seed_nodes
                tmp_subgraph_seed_node_next_hop_output_dir = tmp_subgraph_seed_node_output_dir + str(tmp_hop + 1) + '/'
                mkdir(tmp_subgraph_seed_node_next_hop_output_dir)
                
                tmp_seed_node_output_file = tmp_subgraph_seed_node_next_hop_output_dir + tmp_related_node_class + '.pkl'

                # 查询是否已有相关文件
                if os.path.exists(tmp_seed_node_output_file):
                    # 如果之前已有对应节点表，则进行合并
                    tmp_past_seed_node_pd = pd.read_pickle(tmp_seed_node_output_file)

                    # 先拼接
                    tmp_node_UID_pandas_for_seed_node = pd.concat([tmp_past_seed_node_pd, tmp_node_UID_pandas_for_seed_node])

                    # 再去重
                    tmp_node_UID_pandas_for_seed_node = tmp_node_UID_pandas_for_seed_node.drop_duplicates([tmp_related_node_class + '_UID'], 
                                                                              ignore_index = True)

                # 进行存储
                tmp_node_UID_pandas_for_seed_node.to_pickle(tmp_seed_node_output_file)

                print(tmp_related_node_class + '节点最新seed_node数目:', tmp_node_UID_pandas_for_seed_node.shape[0])
                
                ##############################################################################################
                # 存储关系表
                tmp_seed_node_aim_relation_pandas.to_pickle(tmp_output_subgraph_edges_file)
            
            ##############################################################################################
            # 两种关系都处理完后再存储节点类型表
            np.save(tmp_output_subgraph_edges_node_class_file, tmp_column_name_to_class_dict)

            print('完成关系' + tmp_relation_name_key + '的生成')
            print('##########################################################')
    
    return