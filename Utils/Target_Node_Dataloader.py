import pandas as pd
import numpy as np
import os
import json
from pyspark.sql.types import *
from pyspark.sql.functions import col, explode, concat, lit, substring, expr
from Utils.Pyspark_utils import DEFAULT_STORAGE_LEVEL, estimate_partitions
from Utils.Pyspark_utils import hdfs_read_marker_file, hdfs_create_marker_file, hdfs_file_exists
from Utils.Decorator import Time_Costing

from functools import reduce
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

"""
作用：
    为目标点添加上所需要的特征时间列

输入：
    Spark_Session：pyspark接口


返回值：
    包含全量的目标点及相关信息的字典，键值'Node_Type'对应目标点的节点类型，键值'Data'对应目标点的全量数据，键值'Monthly_dt'对应目标点对应的月份的1号，从而去取对应的关系表和特征表
"""
def Extend_feature_time_for_aim_UID(Spark_Session, tmp_aim_UID_info_dict, Feature_Month_Range = 1):
    # 获取目标点数据
    tmp_aim_UID_df = tmp_aim_UID_info_dict['Data']
    
    # 提取唯一的 Feature_Time 值
    distinct_feature_time_df = tmp_aim_UID_df.select("Feature_Time").distinct()
    
    # 获取全部涉及到的Feature_Time
    feature_times = [row['Feature_Time'] for row in distinct_feature_time_df.collect()]
    feature_times.sort()
    
    # 查看是否要扩展各点要取的特征时间
    if Feature_Month_Range > 1:
        # 为每个 Feature_Time 生成时间范围
        def generate_date_range(start_date, month_range):
            start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            
            extended_date_range_list = [(start_datetime - relativedelta(months=delta)).strftime("%Y-%m-%d") for delta in range(month_range)]
            
            return extended_date_range_list

        # 假设 Feature_Month_Range 是你提供的月份范围
        extended_rows = [(ft, generate_date_range(ft, Feature_Month_Range)) for ft in feature_times]
        extended_feature_time_df = spark.createDataFrame(extended_rows, ["Feature_Time", "Extend_Feature_Time"])

        # 展开日期范围
        extended_feature_time_df = extended_feature_time_df.withColumn("Extend_Feature_Time", explode(col("Extend_Feature_Time")))

        # 合并 DataFrame
        merged_df = tmp_aim_UID_df.join(extended_feature_time_df, on="Feature_Time")

        # 删去 Feature_Time 列并重命名 Extend_Feature_Time 为 Feature_Time
        final_df = merged_df.drop("Feature_Time").withColumnRenamed("Extend_Feature_Time", "Feature_Time")
        
        # 去重
        final_df = final_df.dropDuplicates(subset = [Aim_Node_UID_Name, "Feature_Time"])
        
        final_df.persist(DEFAULT_STORAGE_LEVEL)
        
        tmp_aim_UID_info_dict['Data'] = final_df
    
    # 记录涉及到的全部特征时间
    tmp_aim_UID_info_dict['Feature_Times'] = feature_times
        
    return tmp_aim_UID_info_dict


"""
作用：
    根据目标点的配置文件和全部限制时间区间给出全量的目标点及相关信息

输入：
    Spark_Session：pyspark接口
    Label_Data_Config_dict：目标点配置文件
    tmp_aim_time_range_list：全部目标样本的时间区间
    tmp_store_dir: 标签表存储文件夹

返回值：
    包含全量的目标点及相关信息的字典，键值'Node_Type'对应目标点的节点类型，键值'Data'对应目标点的全量数据，键值'Monthly_dt'对应目标点对应的月份的1号，从而去取对应的关系表和特征表
"""
@Time_Costing
def Read_Target_Node_with_Label(Spark_Session, Label_Data_Config_dict, Aim_Time_Range_Dict, HDFS_Store_Dir, Regernerate = False):
    
    # 获取目标信息
    Aim_Node_Columns = Label_Data_Config_dict['Node_Columns']
    Aim_Node_Types = Label_Data_Config_dict['Node_Types']
    Aim_Node_UID_Names = Label_Data_Config_dict['Node_UID_Names']
        
    Aim_Label_Column = Label_Data_Config_dict['Label_Column']
    Aim_Time_Column = Label_Data_Config_dict['Time_Column']
    Aim_table_name = Label_Data_Config_dict['Table_Name']
    Aim_table_dt = Label_Data_Config_dict['dt']
    
    if 'Feature_Time_Range' in Label_Data_Config_dict:
        Feature_Time_Range = Label_Data_Config_dict['Feature_Time_Range']
    else:
        Feature_Time_Range = 1
    
    # 确定最终要输出的列及顺序
    Output_Columns = Aim_Node_UID_Names + ["Label", "Source_Time", "Feature_Time"]
    
    # 输出的字典
    Target_Node_Info_Dict = {}
    Target_Node_Info_Dict['Node_Types'] = Aim_Node_Types
    Target_Node_Info_Dict['Node_UID_Names'] = Aim_Node_UID_Names
    
    ###################################################################################################
    # 标签表存储位置
    Label_Data_Output_Dir = HDFS_Store_Dir + '/Target_Node'
    
    # 查询是否已存在
    if Regernerate or not hdfs_file_exists(Label_Data_Output_Dir + '/_SUCCESS'):
        # 读取数据
        Target_Node_df = Spark_Session.table(Aim_table_name).filter(col("dt") == Aim_table_dt)
        
        # 节点列不该有Null值或空值
        for tmp_i in range(len(Aim_Node_Columns)):
            Aim_Node_Column = Aim_Node_Columns[tmp_i]
            Target_Node_df = Target_Node_df.filter(col(Aim_Node_Column).isNotNull() & (col(Aim_Node_Column) != ""))
        
        # 标签列不该有null值
        Target_Node_df = Target_Node_df.filter(col(Aim_Label_Column).isNotNull())
        
        ###################################################################################################
        # 时间范围过滤
        time_conditions = []
        for tmp_time_key, tmp_aim_time_range in Aim_Time_Range_Dict.items():
            condition = (col(Aim_Time_Column) >= tmp_aim_time_range[0]) & (col(Aim_Time_Column) < tmp_aim_time_range[1])
            if tmp_time_key + '_Limits' in Label_Data_Config_dict:
                extra_condition = expr(Label_Data_Config_dict[tmp_time_key + '_Limits'])
                condition = condition & extra_condition
            time_conditions.append(condition)
        
        # 应用时间范围过滤
        Target_Node_df = Target_Node_df.filter(reduce(lambda x, y: x | y, time_conditions))

        # 应用统一的限制
        if 'Limits' in Label_Data_Config_dict and Label_Data_Config_dict['Limits'] != '':
            Target_Node_df = Target_Node_df.filter(Label_Data_Config_dict['Limits'])
        
        ###################################################################################################
        # 节点列转换
        for tmp_i in range(len(Aim_Node_Columns)):
            Aim_Node_Column = Aim_Node_Columns[tmp_i]
            Aim_Node_Type = Aim_Node_Types[tmp_i]
            Aim_Node_UID_Name = Aim_Node_Type + '_UID'
            Target_Node_df = Target_Node_df.withColumnRenamed(Aim_Node_Column, Aim_Node_UID_Name)
        
        # 标签列转换
        Target_Node_df = Target_Node_df.withColumnRenamed(Aim_Label_Column, "Label")
        
        # 时间列转换
        Target_Node_df = Target_Node_df.withColumnRenamed(Aim_Time_Column, "Source_Time")
        
        # 添加Feature_Time列
        Target_Node_df = Target_Node_df.withColumn("Feature_Time", concat(substring(col("Source_Time"), 0, 7), lit('-01')))
        
        # 仅保留所需的列
        Target_Node_df = Target_Node_df.select(Output_Columns)
        
        ###################################################################################################
        # 如果要对标签进行映射，则执行映射
        if 'Label_Map_Keys' in Label_Data_Config_dict:
            # 先只保留在Map_Keys范围内的值
            Target_Node_df = Target_Node_df.filter(Target_Node_df['Label'].isin(Label_Data_Config_dict['Label_Map_Keys']))
            
            # 再对数据进行映射
            Target_Node_df = Target_Node_df.replace(Label_Data_Config_dict['Label_Map_Keys'], 
                                       Label_Data_Config_dict['Label_Map_Values'], 'Label')

        # 如果有范围限制，则只保留指定范围的点
        if 'Label_Range_Limit' in Label_Data_Config_dict: 
            Target_Node_df = Target_Node_df.filter(Label_Data_Config_dict["Label_Range_Limit"])

        # 去除重复，每月只保留最大的标签对应的行
        Target_Node_df = Target_Node_df.sort(["Label", "Feature_Time"], ascending = False)
        Target_Node_df = Target_Node_df.dropDuplicates(subset = [Aim_Node_UID_Name, "Feature_Time"])

        ###################################################################################################
        # 获取最优分区数
        best_partitions_count = estimate_partitions(Target_Node_df, 200)
        print("估算的最优分区数：", best_partitions_count)
        
        # 减小分区数，加快之后的运算
        Target_Node_df = Target_Node_df.coalesce(best_partitions_count)
        
        ###################################################################################################
        # 限制训练数据的最大范围
        Target_Node_df.persist(DEFAULT_STORAGE_LEVEL)
        tmp_sample_count = Target_Node_df.count()
        print("原始样本总数：", tmp_sample_count)
        
        # 针对各类型样本限制总量
        tmp_raw_df_list = []
        tmp_filtered_df_list = []
        for tmp_time_key in Aim_Time_Range_Dict.keys():
            # 获取对应时间区间的数据
            tmp_aim_time_range = Aim_Time_Range_Dict[tmp_time_key]
            tmp_time_range_start = tmp_aim_time_range[0]
            tmp_time_range_end = tmp_aim_time_range[1]
            tmp_time_range_limit = f"Source_Time >= '{tmp_time_range_start}' AND Source_Time < '{tmp_time_range_end}'"

            tmp_sub_Target_Node_df = Target_Node_df.filter(tmp_time_range_limit)
            
            # 默认无需过滤
            tmp_time_Max_Rows = 0
            
            # 如果有对应时间的限制，或有总体的限制，则更新限制范围
            if tmp_time_key + '_Max_Rows' in Label_Data_Config_dict:
                tmp_time_Max_Rows = Label_Data_Config_dict[tmp_time_key + 'Max_Rows']
            elif 'Max_Rows' in Label_Data_Config_dict:
                tmp_time_Max_Rows = Label_Data_Config_dict['Max_Rows']
                
            # 如果有限制范围，则启动过滤
            if tmp_time_Max_Rows > 0:
                tmp_sub_Target_Node_df_count = tmp_sub_Target_Node_df.count()
                print(f"{tmp_time_key}类型样本原始总数：{tmp_sub_Target_Node_df_count}")
                
                # 过滤函数
                def Drop_Negative_Samples_Based_on_Max_Rows(tmp_target_df, tmp_Max_Rows, tmp_all_sample_count):
                    if tmp_Max_Rows < tmp_all_sample_count:
                        # 过滤出负样本
                        tmp_df_neg = tmp_target_df.filter(tmp_target_df["Label"] == 0)

                        # 计算列值为0的行的总数
                        neg_sample_count = tmp_df_neg.count()
                        pos_sample_count = tmp_all_sample_count - neg_sample_count
                        desired_neg_sample_size = Label_Data_Config_dict['Max_Rows'] - pos_sample_count

                        print("原始负样本总数：", neg_sample_count)
                        print("原始正样本总数：", pos_sample_count)

                        # 计算抽样比例
                        sample_fraction = desired_neg_sample_size / float(neg_sample_count)

                        # 随机抽样
                        tmp_sampled_df_neg = tmp_df_neg.sample(False, sample_fraction)

                        # 过滤出列值不为0的行
                        tmp_df_pos = Target_Node_df.filter(Target_Node_df["Label"] != 0)

                        # 合并两个DataFrame
                        tmp_filtered_df = tmp_df_pos.union(tmp_sampled_df_neg)

                        return tmp_filtered_df
                    else:
                        print("无需过滤")
                        return tmp_target_df
                
                # 进行过滤
                tmp_sub_Target_Node_df = Drop_Negative_Samples_Based_on_Max_Rows(tmp_sub_Target_Node_df, tmp_time_Max_Rows,
                                                            tmp_sub_Target_Node_df_count)
                
                tmp_filtered_df_list.append(tmp_sub_Target_Node_df)
            else:
                print(f"{tmp_time_key}类型样本总数无最大范围限制")
                
                tmp_raw_df_list.append(tmp_sub_Target_Node_df)
            
        # 如果有过滤后的df，则重新生成Target_Node_df
        if len(tmp_filtered_df_list) > 0:
            # 合并全部样本
            Filtered_Target_Node_df = tmp_filtered_df_list[0]
            for tmp_filtered_df in tmp_filtered_df_list[1:]:
                Filtered_Target_Node_df = Filtered_Target_Node_df.union(tmp_filtered_df)

            for tmp_raw_df in tmp_raw_df_list:
                Filtered_Target_Node_df = Filtered_Target_Node_df.union(tmp_raw_df)

            # 释放之前的df，更新目标节点信息
            Target_Node_df.unpersist()
            Target_Node_df = Filtered_Target_Node_df

            # 一定会复用，故persist
            Target_Node_df.persist(DEFAULT_STORAGE_LEVEL)
                
        Target_Node_Info_Dict['Data'] = Target_Node_df
        
        # 获得全部的特征时间，如果需要，扩展各样本对应的特征时间
        Target_Node_Info_Dict = Extend_feature_time_for_aim_UID(Spark_Session, Target_Node_Info_Dict, Feature_Time_Range)
        ###################################################################################################
        # 存储运算结果
        Target_Node_Info_Dict['Data'].write.mode("overwrite").parquet(Label_Data_Output_Dir)
        
        # 保存涉及到的全部时间
        Feature_Times_str = json.dumps(Target_Node_Info_Dict['Feature_Times'])
        hdfs_create_marker_file(Label_Data_Output_Dir, '_Feature_Times', Feature_Times_str)
        
        # 如果更新过Target_Node_df，则unpersist
        if Feature_Time_Range > 1:
            Target_Node_df.unpersist()
        
        print('已完成标签表的读取和保存')
    else:
        print('标签表已存在，直接读取')
        
        Target_Node_df = Spark_Session.read.parquet(Label_Data_Output_Dir)
        Target_Node_df.persist(DEFAULT_STORAGE_LEVEL)
        Target_Node_Info_Dict['Data'] = Target_Node_df
        
        Feature_Times_str = hdfs_read_marker_file(Label_Data_Output_Dir, '_Feature_Times')
        Target_Node_Info_Dict['Feature_Times'] = json.loads(Feature_Times_str)
       
    print('样本总数目:', Target_Node_Info_Dict['Data'].count())
    print('涉及的全部特征时间:', Target_Node_Info_Dict['Feature_Times'])
    
    return Target_Node_Info_Dict