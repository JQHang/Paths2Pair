import pandas as pd
import numpy as np
import os
from pyspark.sql.types import *

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

"""
作用：
    根据目标点的配置文件和限制时间给出全量的目标点及相关信息

输入：
    Spark_Session：pyspark接口
    Label_Data_Config_dict：目标点配置文件
    tmp_aim_time_start：目标样本的起始时间
    tmp_aim_time_end：目标样本的终止时间
    tmp_store_dir: 标签表存储文件夹

返回值：
    包含全量的目标点及相关信息的字典，键值'Node_Type'对应目标点的节点类型，键值'Data'对应目标点的全量数据，键值'Monthly_dt'对应目标点对应的月份的1号，从而去取对应的关系表和特征表
"""
def get_aim_UID_with_label_rdd(Spark_Session, Label_Data_Config_dict, tmp_aim_time_start, tmp_aim_time_end, tmp_store_dir, 
                     keep_time_column = False):
    Aim_Node_type = Label_Data_Config_dict['Node_Type']
    Aim_Node_Column = Label_Data_Config_dict['Node_Column']
    Aim_Label_Column = Label_Data_Config_dict['Label_Column']
    Aim_Time_Column = Label_Data_Config_dict['Time_Column']
    Aim_table_name = Label_Data_Config_dict['Table_Name']
    Aim_table_dt = Label_Data_Config_dict['dt']
    Aim_Node_UID_Name = Aim_Node_type + '_UID'
    
    # 标签表存储位置+文件名
    tmp_output_data_label_file = tmp_store_dir + 'Target_Node.pkl'
    
    # 查询是否已存在
    if not os.path.exists(tmp_output_data_label_file):
        # 不存在则进行运算
        tmp_sql_command = """SELECT\n    """ + Aim_Node_Column + """ AS """ + Aim_Node_UID_Name
        tmp_sql_command = tmp_sql_command + """,\n    """ + Aim_Label_Column + """ AS Label"""
        
        if keep_time_column:
            tmp_sql_command = tmp_sql_command + """,\n    """ + Aim_Time_Column + """ AS Source_Time"""
            tmp_sql_command = tmp_sql_command + """,\n    CONCAT(SUBSTRING(""" + Aim_Time_Column + """, 0, 8), '01') AS Feature_Time"""
        
        tmp_sql_command = tmp_sql_command + """\nFROM\n    """ + Aim_table_name + """\nWHERE\n     dt = '""" + Aim_table_dt + """'"""
        tmp_sql_command = tmp_sql_command + """\n    AND """ + Aim_Time_Column + """ >= '""" + tmp_aim_time_start + """'"""
        tmp_sql_command = tmp_sql_command + """\n    AND """ + Aim_Time_Column + """ <  '""" + tmp_aim_time_end + """'"""
        tmp_sql_command = tmp_sql_command + """\n    AND """ + Aim_Label_Column + """ IN (0, 1)"""

        if 'Limits' in Label_Data_Config_dict and Label_Data_Config_dict['Limits'] != '':
            tmp_sql_command = tmp_sql_command + '\nAND ' + Label_Data_Config_dict['Limits']
            print('标签表限制条件为', Label_Data_Config_dict['Limits'])
        
        tmp_aim_entity_rdd = Spark_Session.sql(tmp_sql_command)
        print('完整sql语句为:\n' + tmp_sql_command)
        
        # 去除重复，每月优先保留签约那一次的样本
        if keep_time_column:
            tmp_aim_entity_rdd = tmp_aim_entity_rdd.sort(["Label", "Feature_Time"], ascending = False).coalesce(1)
            tmp_aim_entity_rdd = tmp_aim_entity_rdd.dropDuplicates(subset = [Aim_Node_UID_Name, "Feature_Time"])
        else:
            tmp_aim_entity_rdd = tmp_aim_entity_rdd.sort(["Label"], ascending = False).coalesce(1)
            tmp_aim_entity_rdd = tmp_aim_entity_rdd.dropDuplicates(subset = [Aim_Node_UID_Name])
            
        # 存储运算结果
        tmp_aim_entity_pd = tmp_aim_entity_rdd.toPandas()
        
        tmp_aim_entity_info_dict = {'Node_Type': Aim_Node_type,
                           'Data': tmp_aim_entity_rdd,
                           'Data_pd': tmp_aim_entity_pd,
                           'Monthly_dt': tmp_aim_time_start[0:8] + '01'}
        
        tmp_aim_entity_pd.to_pickle(tmp_output_data_label_file)
        print('已完成标签表的读取和保存')
    else:
        print('标签表已存在，直接读取')
        
        # 已存在则直接读取
        tmp_aim_entity_pd = pd.read_pickle(tmp_output_data_label_file)
        
        # 转为rdd格式
        if keep_time_column:
            tmp_target_node_table_schema = StructType([StructField(Aim_Node_UID_Name, StringType(), True),
                                         StructField("Label", IntegerType(), True),
                                         StructField("Source_Time", StringType(), True),
                                         StructField("Feature_Time", StringType(), True)])
        else:
            tmp_target_node_table_schema = StructType([StructField(Aim_Node_UID_Name, StringType(), True),
                                         StructField("Label", IntegerType(), True)])
            
        tmp_aim_entity_rdd = Spark_Session.createDataFrame(tmp_aim_entity_pd, tmp_target_node_table_schema)
        
        tmp_aim_entity_info_dict = {'Node_Type': Aim_Node_type,
                           'Data': tmp_aim_entity_rdd,
                           'Data_pd': tmp_aim_entity_pd,
                           'Monthly_dt': tmp_aim_time_start[0:8] + '01'}
        
    print('正样本数目:', tmp_aim_entity_pd[tmp_aim_entity_pd['Label'] == 1].shape)
    print('负样本数目:', tmp_aim_entity_pd[tmp_aim_entity_pd['Label'] == 0].shape)
    
    return tmp_aim_entity_info_dict

"""
作用：
    为目标点添加上所需要的特征时间列

输入：
    Spark_Session：pyspark接口


返回值：
    包含全量的目标点及相关信息的字典，键值'Node_Type'对应目标点的节点类型，键值'Data'对应目标点的全量数据，键值'Monthly_dt'对应目标点对应的月份的1号，从而去取对应的关系表和特征表
"""
def Extend_feature_time_for_aim_UID(Spark_Session, tmp_aim_UID_info_dict, Feature_Month_Range):
    Aim_Node_UID_Name = tmp_aim_UID_info_dict['Node_Type'] + '_UID'
    
    # 先取出对应pandas文件
    tmp_aim_UID_pd = tmp_aim_UID_info_dict['Data_pd'].copy()

    # 统计全部需要的月份时间
    tmp_Feature_Time_list = list(set(tmp_aim_UID_pd['Feature_Time']))

    # 为每个月份时间添加上所需的特征时间
    tmp_Feature_Time_to_Extend_Feature_Time_list = [[], []]
    for tmp_Feature_Time in tmp_Feature_Time_list:
        tmp_Feature_Time_to_Extend_Feature_Time_list[0].append(tmp_Feature_Time)
        tmp_Feature_Time_to_Extend_Feature_Time_list[1].append(tmp_Feature_Time)
            
        for tmp_month_delta in range(1, Feature_Month_Range):
            tmp_Feature_Time_dt = datetime.strptime(tmp_Feature_Time, "%Y-%m-%d")

            tmp_Feature_Time_dt = tmp_Feature_Time_dt - relativedelta(months = tmp_month_delta)

            tmp_Feature_Time_to_Extend_Feature_Time_list[0].append(tmp_Feature_Time)
            tmp_Feature_Time_to_Extend_Feature_Time_list[1].append(tmp_Feature_Time_dt.strftime("%Y-%m-%d"))

    # 将list转为pandas
    tmp_Feature_Time_to_Extend_Feature_Time_pd = pd.DataFrame(np.array(tmp_Feature_Time_to_Extend_Feature_Time_list).T, 
                                           columns = ['Feature_Time', 'Extend_Feature_Time'])

    # 对原始信息添加Extend_Feature_Time列
    tmp_aim_UID_pd = tmp_aim_UID_pd.merge(tmp_Feature_Time_to_Extend_Feature_Time_pd, on = 'Feature_Time')
    
    # 删去Feature_Time列
    tmp_aim_UID_pd = tmp_aim_UID_pd.drop(columns=['Feature_Time'])
    
    # 将Extend_Feature_Time改名为Feature_Time
    tmp_aim_UID_pd = tmp_aim_UID_pd.rename(columns={"Extend_Feature_Time": "Feature_Time"}).reset_index(drop=True)
    
    # 获取各列的类型
    StructType_list = [StructField(Aim_Node_UID_Name, StringType(), True)]
    if 'Label' in tmp_aim_UID_pd.columns:
        StructType_list.append(StructField("Label", IntegerType(), True))
    if 'Source_Time' in tmp_aim_UID_pd.columns:
        StructType_list.append(StructField("Source_Time", StringType(), True))
    StructType_list.append(StructField("Feature_Time", StringType(), True))
    
    tmp_target_node_table_schema = StructType(StructType_list)
    
    # 将pandas转为rdd数据(未必有Label)
    tmp_aim_UID_rdd = Spark_Session.createDataFrame(tmp_aim_UID_pd, tmp_target_node_table_schema)

    tmp_aim_UID_info_dict['Data'] = tmp_aim_UID_rdd
    tmp_aim_UID_info_dict['Data_pd'] = tmp_aim_UID_pd
    
    return tmp_aim_UID_info_dict