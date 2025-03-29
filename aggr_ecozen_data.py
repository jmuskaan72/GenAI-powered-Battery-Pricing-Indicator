import pandas as pd 
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 20000)
import numpy as np 
import datetime as dt
from datetime import timedelta
from datetime import date
from datetime import datetime
import os 
import re
pd.options.mode.chained_assignment = None  # default='warn'

def get_day_hour(value):
    day = value.date()
    hour = value.hour
    minute = value.minute
    return day,hour,minute

def get_ecozen_file(file_path):
    df = pd.read_csv(f'/Users/Muskaan_Jain/Dev/data_engineering/blusmart-battery-cell-level-data/{file_path}')
    print("Data file length:",df.shape)
    df.rename(columns={'Topic':'vehicle_number'}, inplace=True) 

    # Sort columns based on the desired order
    l1 = list(df.columns)
    l1.remove('vehicle_number')
    l1.remove('createdAt')
    desired_order = ['vehicle_number','createdAt'] + l1
    df = df[desired_order]
    df.insert(3,'trip_day','')
    df.insert(4,'trip_hour','')
    df.insert(5,'trip_min','')

    # sorting the data 
    df = df.sort_values(['vehicle_number','createdAt'])
    df.reset_index(inplace=True, drop=True)

    #extract hour, minute, day from devicetime
    df['updatedAt'] = pd.to_datetime(df['updatedAt'])
    df['createdAt'] = pd.to_datetime(df['createdAt'])
    df['deviceTime'] = pd.to_datetime(df['deviceTime'])

    df['trip_day'] = df.apply(lambda x: get_day_hour(x['createdAt'])[0], axis=1)
    df['trip_hour'] = df.apply(lambda x: get_day_hour(x['createdAt'])[1], axis=1)
    df['trip_min'] = df.apply(lambda x: get_day_hour(x['createdAt'])[2], axis=1)

    return df 

#get the vehicle model variants present  
def get_model_variants(file_path):
    vehicle_df = pd.read_csv('/Users/Muskaan_Jain/Dev/data_uploads/collector_db_vehicles.csv')
    df2 = vehicle_df[['id','vehicle_number','model','battery_capacity','km_range','manufacturer','charger_type','fast_charging_time_range','slow_charging_time_range','hubName']]
    model_types = df2['model'].unique()
    print("Model Variants in Vehicles Data are:",model_types) 

    df = get_ecozen_file(file_path)
    vehicle_models = pd.DataFrame(df['vehicle_number'].unique())
    vehicle_models = pd.merge(vehicle_models, df2, left_on=[0], right_on=['vehicle_number'])
    print("Data Shape after Merging:",vehicle_models.shape)
    print("\nBattery model variants present are:",vehicle_models['model'].unique()) 

def params_division(df):
    col_list = list(df.columns)
    print("\nFinal Columns List:",col_list)
    base_params = ['vehicle_number','deviceTime','trip_day','trip_hour','trip_min']

    #feature engineering - paramas division
    ocv_params = base_params + sorted(list(filter(re.compile(".*OCV").match, col_list)))
    voltage_params = base_params + sorted(list(filter(re.compile(".*(_V|DCV)").match, col_list)))
    temp_params = base_params + sorted(list(filter(re.compile(".*(CELL_T)").match, col_list)))
    ir_params = base_params + sorted(list(filter(re.compile(".*RI").match, col_list)))
    battery_health_params = base_params + ['RSOC','CYCLE','SOH','PDOD','DCA','DCV','DCL','CCL','BAL_AL','ODO','ADP_AMPHR']

    return base_params,ocv_params, voltage_params, temp_params, ir_params, battery_health_params

def calculate_aggs(df, grouped_data, input_params):
    #get agg values for every vehicle 
    base_params,ocv_params, voltage_params, temp_params, ir_params, battery_health_params = params_division(df)
    params_values = [x for x in input_params if x not in base_params]
    agg_df = grouped_data[params_values].agg(['sum', 'mean', 'std','min','max']).reset_index()

    #renaming the column - Handling multiindex levels 
    agg_df.columns = ['_'.join(col).strip() if type(col) is tuple else col for col in agg_df.columns.values]
    agg_df.rename(columns={'vehicle_number_':'vehicle_number', 'trip_day_':'trip_day'}, inplace=True)
    return agg_df

def get_agg_data(df, grouped_data):
    base_params,ocv_params, voltage_params, temp_params, ir_params, battery_health_params = params_division(df)

    #OCV agg data
    ocv_agg_df =  calculate_aggs(df, grouped_data, ocv_params)

    #voltage agg data 
    voltage_params2 =  [x for x in voltage_params if x not in ['MAX_V_CELL','MIN_V_CELL']]
    votlage_agg_df =  calculate_aggs(df, grouped_data, voltage_params2)
    votlage_agg_df.drop(['MIN_CELL_V_sum','MIN_CELL_V_mean','MAX_CELL_V_sum','MAX_CELL_V_mean'], axis=1, inplace=True)

    #temp agg data 
    temp_agg_df =  calculate_aggs(df, grouped_data, temp_params)
    temp_agg_df.drop(['MAX_CELL_T_sum','MIN_CELL_T_sum','MAX_CELL_T_mean','MIN_CELL_T_mean'], axis=1, inplace=True)

    #Internal resistance agg data
    ir_agg_df =  calculate_aggs(df, grouped_data, ir_params)

    #battery params agg data 
    battery_health_agg_df=  calculate_aggs(df, grouped_data, battery_health_params)
    battery_health_agg_df.drop(['BAL_AL_mean','BAL_AL_std','BAL_AL_min','BAL_AL_sum','ADP_AMPHR_sum','CCL_sum','DCL_sum','DCV_sum','DCA_sum'], axis=1, inplace=True)

    #merge all aggs dfs 
    agg_df_final = pd.concat([ocv_agg_df,votlage_agg_df, ir_agg_df, temp_agg_df,battery_health_agg_df], axis=1, join='inner')
    print("Final merged data shape:",agg_df_final.shape)
    return agg_df_final

#final call 0 parsing through all the datafiles 
folder_path = '/Users/Muskaan_Jain/Dev/data_engineering/blusmart-battery-cell-level-data'
datafiles = os.listdir(folder_path)

for file_path in datafiles:
    print(file_path)
    df = get_ecozen_file(file_path)
    get_model_variants(file_path)

    #get daily basis 
    grouped_data_daily = df.groupby(['vehicle_number','trip_day'])
    agg_df_final_day = get_agg_data(df, grouped_data_daily)
    agg_df_final_day.to_csv(f'/Users/Muskaan_Jain/Dev/data_engineering/aggr_data_blusmart/daily_aggr_{file_path}', index=False)
    # agg_df_final_day.tail()

    #get hourly basis 
    grouped_hourly_data = df.groupby(['vehicle_number','trip_day','trip_hour'])
    agg_df_final_hour = get_agg_data(df, grouped_hourly_data)
    agg_df_final_hour.to_csv(f'/Users/Muskaan_Jain/Dev/data_engineering/aggr_data_blusmart/hourly_aggr_{file_path}', index=False)
    # agg_df_final_hour.head()
