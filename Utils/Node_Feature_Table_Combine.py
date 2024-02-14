import Utils.Pyspark_utils
from Utils.Pyspark_utils import Data_In_HDFS_Path
from Utils.Pyspark_utils import sample_random_n_samples_for_samll_rdd
from Utils.Pyspark_utils import sample_top_n_groupby_samples_for_samll_rdd
from Utils.Pyspark_utils import sample_random_n_groupby_samples_for_samll_rdd
from Utils.Pyspark_utils import sample_rdd_from_aim_row, sample_rdd_from_aim_range
from Utils.Pyspark_utils import Groupby_Feature_Table, hdfs_create_marker_file, hdfs_file_exists
from Utils.Pyspark_utils import Pyspark_Create_Table, Upload_RDD_Data_to_Database, Preprocess_Numerical_Data

from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import Row
from pyspark.sql.functions import udf, lit
from pyspark.sql.types import IntegerType, FloatType, DoubleType, DecimalType, LongType, ShortType, ByteType
from pyspark.sql.types import *
from pyspark.sql.functions import broadcast, col
from pyspark.storagelevel import StorageLevel

import pandas as pd
import numpy as np
import math
import os
import json
from tqdm import tqdm
from datetime import datetime


# """
# 作用：
#     合并同一节点的各个特征表
#
# 返回值：

# """
def Node_Feature_Table_Combine(Spark_Session, Aim_Node_Type, Aim_Node_Feature_Table_Info, Aim_Feature_Table_dt, Table_Name_Comment,
                     Output_Table_Type = 'HDFS'):
    # Set Upload Table Name
    Upload_Table_Name = ('CompP_N_Feat___Combined_Table___' + Aim_Node_Type + '_UID' + '_' + Table_Name_Comment)
    if len(Upload_Table_Name) > 120:
        Upload_Table_Name = Upload_Table_Name[:120]
        print('Error: Cut the table name to 120 characters')
    print('Output Table Name:', Upload_Table_Name)
    
    if Output_Table_Type == "HDFS":
        tmp_output_pyspark_df_store_dir = Data_In_HDFS_Path + '/ComplexPath_Aggregation/' + Upload_Table_Name + f"/dt={Aim_Feature_Table_dt}"
        
        if hdfs_file_exists(tmp_output_pyspark_df_store_dir + "/_SUCCESS"):
            print("File existed:", tmp_output_pyspark_df_store_dir)
            return
    #####################################################################################################################
    # 记录该节点涉及的全部特征
    tmp_all_useful_feature_cols_list = []

    # 记录该节点涉及的全部特征的注释
    tmp_all_useful_feature_cols_comments_list = []
    
    tmp_combine_start_time = datetime.now()
    
    # 按顺序读取各个表
    for tmp_feature_table_i in range(0, len(Aim_Node_Feature_Table_Info["Feature_Data_List"])):
        Feature_Table_Info = Aim_Node_Feature_Table_Info["Feature_Data_List"][tmp_feature_table_i]
        
        tmp_feature_table_name = Feature_Table_Info['Table_Name']
        tmp_aim_column_name = Feature_Table_Info['UID']
        
        # Check Data Source
        tmp_data_source = "Online"
        if "Source" in Feature_Table_Info:
            tmp_data_source = Feature_Table_Info["Source"]
        
        print('Process Node Table:', tmp_feature_table_name)
        
        if tmp_data_source == "Online":
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
        else:
            tmp_feature_data_src = Data_In_HDFS_Path + '/ComplexPath_Aggregation/' + tmp_feature_table_name + f"/dt={Aim_Feature_Table_dt}"
            tmp_feature_table_rdd = Spark_Session.read.parquet(tmp_feature_data_src)
        
        # Filter Null Value
        tmp_feature_table_rdd = tmp_feature_table_rdd.filter(col(tmp_aim_column_name).isNotNull() & (col(tmp_aim_column_name) != ""))
        
        # 确保添加特征目标列的列名为tmp_node_class + '_UID'
        tmp_feature_table_rdd = tmp_feature_table_rdd.withColumnRenamed(tmp_aim_column_name, Aim_Node_Type + '_UID')
        
        # 通过persist保留计算结果
        tmp_feature_table_rdd = tmp_feature_table_rdd.persist()

        tmp_feature_table_rdd_raw_count = tmp_feature_table_rdd.count()

        if tmp_feature_table_rdd_raw_count == 0:
            print('Error: Node Table', tmp_feature_table_name, 'is empty')
        else:
            # Drop duplicates
            tmp_feature_table_rdd = tmp_feature_table_rdd.dropDuplicates([Aim_Node_Type + '_UID'])

            tmp_feature_table_rdd = tmp_feature_table_rdd.persist()

            tmp_feature_table_rdd_count = tmp_feature_table_rdd.count()

            if tmp_feature_table_rdd_raw_count != tmp_feature_table_rdd_count:
                print('Error: Node Table:', tmp_feature_table_name, 'has duplicate nodes at time:', Aim_Feature_Table_dt, 
                    'Keep first row at present. The original row number:', tmp_feature_table_rdd_raw_count,
                     'After drop duplicates:', tmp_feature_table_rdd_count)
        
        ##############################################################################################
        # Keep the numerical columns and the corresponding comment
        def check_numeric_columns(df_schema):
            numeric_types = (IntegerType, FloatType, DoubleType, DecimalType, LongType, ShortType, ByteType)

            numeric_columns = []
            numeric_columns_comment = []
            for field in df_schema.fields:
                if isinstance(field.dataType, numeric_types):
                    numeric_columns.append(field.name)
                    numeric_columns_comment.append(field.metadata.get('comment', field.name) if 'comment' in field.metadata else field.name)
            return numeric_columns, numeric_columns_comment

        tmp_useful_feature_cols_list, tmp_useful_feature_cols_comments_list = check_numeric_columns(tmp_feature_table_rdd.schema)

        print('Numerical columns number in table', tmp_feature_table_name, 'is:', len(tmp_useful_feature_cols_list))

        # Only keep the UID column and the numerical columns
        tmp_feature_table_rdd = tmp_feature_table_rdd.select([Aim_Node_Type + '_UID'] + tmp_useful_feature_cols_list)
        ##############################################################################################
        # Check whether the feature data isValid
        if tmp_feature_table_rdd_raw_count != 0:
            # Get max and min for each numerical column
            tmp_summary_df = tmp_feature_table_rdd.drop(Aim_Node_Type + '_UID').summary("min", "max")

            # Transfer the result to dict
            min_values = tmp_summary_df.filter(col("summary") == "min").drop("summary").collect()[0].asDict()
            max_values = tmp_summary_df.filter(col("summary") == "max").drop("summary").collect()[0].asDict()

            # check whether they are equal and keep the result
            invalid_cols_values = {col_name: min_values[col_name] for col_name in min_values.keys() if min_values[col_name] == max_values[col_name]}

            if len(invalid_cols_values.keys()) > 0:
                print('ERROR: Node Table ', tmp_feature_table_name, 'has invalid columns at time', Aim_Feature_Table_dt, 
                'The numbers in the column are all the same, and the column name and value are:')
                print(invalid_cols_values)
        ##############################################################################################
        # Change Column Name
        tmp_new_useful_feature_cols_list = []
        for tmp_col_index in range(len(tmp_useful_feature_cols_list)):
            tmp_old_name = tmp_useful_feature_cols_list[tmp_col_index]
            tmp_new_name = 'Feature_' + str(len(tmp_all_useful_feature_cols_list) + tmp_col_index)

            tmp_new_useful_feature_cols_list.append(tmp_new_name)
            tmp_feature_table_rdd = tmp_feature_table_rdd.withColumnRenamed(tmp_old_name, tmp_new_name)

        # Record the column name and comment
        tmp_all_useful_feature_cols_list.extend(tmp_new_useful_feature_cols_list)
        tmp_all_useful_feature_cols_comments_list.extend(tmp_useful_feature_cols_comments_list)

        # If just start, then use tmp_feature_table_rdd or join with the past result
        if tmp_feature_table_i == 0:
            tmp_combined_feature_table_rdd = tmp_feature_table_rdd

        else:
            tmp_combined_feature_table_rdd = tmp_combined_feature_table_rdd.join(tmp_feature_table_rdd, Aim_Node_Type + '_UID', 'outer')
            
            if tmp_feature_table_i % 5 == 0:
                # Repartition to reduce Data Skew
                tmp_combined_feature_table_rdd = tmp_combined_feature_table_rdd.repartition(Aim_Node_Type + '_UID')
            
    # Repartition to reduce Data Skew
    tmp_combined_feature_table_rdd = tmp_combined_feature_table_rdd.repartition(10000, Aim_Node_Type + '_UID')
    tmp_combined_feature_table_rdd.persist()

    ###################################################################################################################################
    # Preprocess and Upload the result
    if Output_Table_Type == 'HDFS':
        # Preprocessing
        tmp_agg_feature_for_upload = Preprocess_Numerical_Data(tmp_combined_feature_table_rdd, ['Raw', 'Normalized', 'Standard'], 
                                             [Aim_Node_Type + '_UID'], tmp_all_useful_feature_cols_list)
        
        # Path to store hdfs
        tmp_output_pyspark_df_store_dir = Data_In_HDFS_Path + '/ComplexPath_Aggregation/' + Upload_Table_Name
        
        print('Start writing')
        
        # Store to target dir
        tmp_agg_feature_for_upload.write.mode("overwrite").parquet(tmp_output_pyspark_df_store_dir + f"/dt={Aim_Feature_Table_dt}")
        
        # Store the feature column name
        tmp_feature_name_str = '###'.join(tmp_all_useful_feature_cols_list)
        hdfs_create_marker_file(tmp_output_pyspark_df_store_dir + f"/dt={Aim_Feature_Table_dt}", '_Feature_Name', tmp_feature_name_str)
        
        # Store the feature column comment
        tmp_feature_comment_str = '###'.join(tmp_all_useful_feature_cols_comments_list)
        hdfs_create_marker_file(tmp_output_pyspark_df_store_dir + f"/dt={Aim_Feature_Table_dt}", '_Feature_Comment', tmp_feature_comment_str)
        
    elif Output_Table_Type == 'Online':
        # Upload batch
        Upload_batch_count = math.ceil((len(tmp_all_useful_feature_cols_list) * tmp_combined_feature_table_rdd.count()) / 10000000000)

        # Upload data
        Upload_RDD_Data_to_Database(Spark_Session, 'tmp.tmp_' + Upload_Table_Name, tmp_combined_feature_table_rdd, Aim_Feature_Table_dt, 
                           Set_Table_Columns_Comment_List = tmp_upload_cols_comment_list, batch_count = Upload_batch_count)

    tmp_combined_feature_table_rdd = tmp_combined_feature_table_rdd.unpersist()
    
    return