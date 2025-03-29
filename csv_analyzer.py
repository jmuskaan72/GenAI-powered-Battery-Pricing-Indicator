import pandas as pd 
import numpy as np 
import ollama 
import re 
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from electra_battery_usage_market_prompt import *
from concurrent.futures import ThreadPoolExecutor
from IPython.display import display
import concurrent.futures

generate_agg_fields_prompt = """
Role:
You are an expert in analyzing CSV DataFrames containing electric vehicle telemetry data. Your task is to generate a Python function that performs aggregate analysis on a given dataset, summarizing key battery and vehicle performance metrics per vehicle.

Input:
A CSV DataFrame containing the following columns:
- vehicle_number or Topic (identifies each vehicle)
- SOH (State of Health)
- MAX_CELL_T (maximum battery cell temperature)
- ADP_AMPHR (final battery capacity measurement)
- ODO (odometer reading in kilometers)
- CYCLE (total cycle count)
- MAX_CELL_V (maximum cell voltage)
- MIN_CELL_V (minimum cell voltage)

Task:
Generate a Python function named get_vehicle_usage_summary that:
- Computes mean SOH, rounded to 2 decimal places.
- Counts the number of temperature excursions (where MAX_CELL_T > 40°C).
- Determines the final capacity from ADP_AMPHR as its mean value.
- Finds the age of the vehicle based on the highest ODO value (in km).
- Retrieves the total cycle count from the maximum CYCLE value.
- Extracts the maximum and minimum cell voltage from MAX_CELL_V and MIN_CELL_V.
- The python function should handle name error as well: NameError: name 'vehicles_column' is not defined
- Outputs results in a structured DataFrame only and no other output should be printed instead of vehicle_usage_df. 
- The output should contain python code only and nothing else or any other strings and code doesn't faces any indendation issues and attribution errors as well.

Few-shot examples to learn how to generate code from:

Example 1: Basic Aggregation Function
'''
import pandas as pd

def get_vehicle_usage_summary(df):
    vehicles_list = df['Topic' if 'Topic' in df.columns else 'vehicle_number'].unique()
    
    vehicle_usage_data = []
    for vehicle in vehicles_list:
        df_filter = df[df['Topic' if 'Topic' in df.columns else 'vehicle_number'] == vehicle]
        df_filter.reset_index(inplace=True, drop=True)

        usage_metrics = {
            "vehicle_number": vehicle,
            "mean_soh": float(df_filter['SOH'].mean().round(2)),
            "temperature_excursions": int(df_filter[df_filter['MAX_CELL_T'] > 40.0]['MAX_CELL_T'].count()),
            "final_capacity": float(df_filter['ADP_AMPHR'].mean().round(2)),
            "age_of_vehicle": float(df_filter['ODO'].max().round(2)),
            "num_cycles": int(df_filter['CYCLE'].max()),
            "max_voltage": float(df_filter['MAX_CELL_V'].max()),
            "min_voltage": float(df_filter['MIN_CELL_V'].min())
        }

        usage_metrics["vehicle_summary"] = usage_metrics
        vehicle_usage_data.append(usage_metrics)
    
    vehicle_usage_df = pd.DataFrame(vehicle_usage_data)
    vehicle_usage_df['vehicle_summary'] = vehicle_usage_df['vehicle_summary'].apply(lambda x: (x.pop('vehicle_summary', None), x)[1] if isinstance(x, dict) else x)
    return vehicle_usage_df
'''

Example 2: Handling Missing Data and Edge Cases
'''
import numpy as np
import pandas as pd

def get_vehicle_usage_summary(df):
    vehicles_list = df['Topic' if 'Topic' in df.columns else 'vehicle_number'].unique()
    
    vehicle_usage_data = []
    for vehicle in vehicles_list:
        df_filter = df[df['Topic' if 'Topic' in df.columns else 'vehicle_number'] == vehicle]
        df_filter.reset_index(inplace=True, drop=True)

        usage_metrics = {
            "vehicle_number": vehicle,
            "mean_soh": float(df_filter['SOH'].mean().round(2)) if not df_filter['SOH'].isnull().all() else np.nan,
            "temperature_excursions": int(df_filter[df_filter['MAX_CELL_T'] > 40.0]['MAX_CELL_T'].count()) if 'MAX_CELL_T' in df_filter.columns else 0,
            "final_capacity": float(df_filter['ADP_AMPHR'].mean().round(2)) if 'ADP_AMPHR' in df_filter.columns else np.nan,
            "age_of_vehicle": float(df_filter['ODO'].max().round(2)) if 'ODO' in df_filter.columns else np.nan,
            "num_cycles": int(df_filter['CYCLE'].max()) if 'CYCLE' in df_filter.columns else 0,
            "max_voltage": float(df_filter['MAX_CELL_V'].max()) if 'MAX_CELL_V' in df_filter.columns else np.nan,
            "min_voltage": float(df_filter['MIN_CELL_V'].min()) if 'MIN_CELL_V' in df_filter.columns else np.nan
        }

        usage_metrics["vehicle_summary"] = usage_metrics
        vehicle_usage_data.append(usage_metrics)
    
    vehicle_usage_df = pd.DataFrame(vehicle_usage_data)
    vehicle_usage_df['vehicle_summary'] = vehicle_usage_df['vehicle_summary'].apply(lambda x: (x.pop('vehicle_summary', None), x)[1] if isinstance(x, dict) else x)
    return vehicle_usage_df
'''

Example 3: Basic Aggregation Function Handling Attribute Errors
'''
import pandas as pd
import numpy as np

def get_vehicle_usage_summary(df):
    vehicles_list = df['Topic' if 'Topic' in df.columns else 'vehicle_number'].unique()
    
    vehicle_usage_data = []
    for vehicle in vehicles_list:
        df_filter = df[df['Topic' if 'Topic' in df.columns else 'vehicle_number'] == vehicle]
        df_filter.reset_index(inplace=True, drop=True)

        def safe_round(value, decimals=2):
            return round(value, decimals) if isinstance(value, (int, float)) and not pd.isnull(value) else np.nan

        usage_metrics = {
            "vehicle_number": vehicle,
            "mean_soh": safe_round(df_filter['SOH'].mean()) if 'SOH' in df_filter.columns else np.nan,
            "temperature_excursions": int(df_filter[df_filter['MAX_CELL_T'] > 40.0]['MAX_CELL_T'].count()) if 'MAX_CELL_T' in df_filter.columns else 0,
            "final_capacity": safe_round(df_filter['ADP_AMPHR'].mean()) if 'ADP_AMPHR' in df_filter.columns else np.nan,
            "age_of_vehicle": safe_round(df_filter['ODO'].max()) if 'ODO' in df_filter.columns else np.nan,
            "num_cycles": int(df_filter['CYCLE'].max()) if 'CYCLE' in df_filter.columns else 0,
            "max_voltage": safe_round(df_filter['MAX_CELL_V'].max()) if 'MAX_CELL_V' in df_filter.columns else np.nan,
            "min_voltage": safe_round(df_filter['MIN_CELL_V'].min()) if 'MIN_CELL_V' in df_filter.columns else np.nan
        }

        usage_metrics["vehicle_summary"] = usage_metrics
        vehicle_usage_data.append(usage_metrics)
    
    vehicle_usage_df = pd.DataFrame(vehicle_usage_data)
    vehicle_usage_df['vehicle_summary'] = vehicle_usage_df['vehicle_summary'].apply(lambda x: (x.pop('vehicle_summary', None), x)[1] if isinstance(x, dict) else x)
    return vehicle_usage_df
'''

# Example usage:
vehicle_usage_df = get_vehicle_usage_summary(df)
"""

@st.cache_data  # Cache the prices estimated for each vehicle combined and visualized in line format
def get_cached_pricing_all_vehicles(vehicle_usage_df):
    return get_pricing_all_vehicles(vehicle_usage_df)

def generate_py_code_agg_fields(generate_agg_fields_prompt):
    try:
        # st.markdown("*GenAI is running..*")
        start_time = time.time()
        agg_func_response = ollama.generate(model='mistral', prompt=generate_agg_fields_prompt)
        py_func_value = agg_func_response['response']
        
        # st.write("Python function formulated!")
        # st.write("Output displayed in run %.2f seconds" % (time.time() - start_time))
        # st.write(py_func_value)
        return py_func_value

    except Exception as e:
        print(f"Error: {e}")   

def extract_python_function(text):
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    return match.group(1) if match else None

def get_vehicle_usage_df(df, generate_agg_fields_prompt):
    #extract the aggregated fields df for vehicle usage 
    py_func_value = generate_py_code_agg_fields(generate_agg_fields_prompt)
    extracted_code = extract_python_function(py_func_value) 

    #execute the extracted py code which returns the get_vehicle_usage_summary func
    if extracted_code: 
        exec(extracted_code, globals())

    #get the dataframe 
    vehicle_usage_df = get_vehicle_usage_summary(df)
    return vehicle_usage_df 

def plot_battery_health_across_vehicles(vehicle_usage_df):
    vehicle_usage_df = vehicle_usage_df.sort_values(by='vehicle_number')
    fig = go.Figure()
    
    # Add SOH as bar chart
    fig.add_trace(go.Bar(
        x=vehicle_usage_df['vehicle_number'], 
        y=vehicle_usage_df['mean_soh'], 
        name='Mean SOH',
        marker_color='yellow',
        opacity=0.8
    ))
    
    # Add Capacity as bar chart
    fig.add_trace(go.Bar(
        x=vehicle_usage_df['vehicle_number'], 
        y=vehicle_usage_df['final_capacity'], 
        name='Battery Capacity (Ah)',
        marker_color='blue',
        opacity=0.8
    ))
    
    # Add Number of Cycles as line chart
    fig.add_trace(go.Scatter(
        x=vehicle_usage_df['vehicle_number'], 
        y=vehicle_usage_df['num_cycles'], 
        name='Number of Cycles',
        mode='lines+markers',
        line=dict(color='cyan', width=2),
        marker=dict(size=6, symbol='circle')
    ))
    
    # Update layout
    fig.update_layout(
        title='Battery Health Metrics Across Vehicles',
        xaxis_title='Vehicle Number',
        yaxis_title='Value',
        barmode='group',
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        legend=dict(font=dict(color='white'))
    )
    return fig

def process_vehicle(usage_data):
    """Process a single vehicle's price analysis af1nd return the result."""
    price_analysis_report = get_price_analysis_report(usage_data)
    price_values = get_price_values(price_analysis_report)
    
    return {
        'vehicle_number': usage_data['vehicle_number'], 
        'mean_soh': usage_data['mean_soh'], 
        'temperature_excursions': usage_data['temperature_excursions'], 
        'final_capacity': usage_data['final_capacity'], 
        'age_of_vehicle': usage_data['age_of_vehicle'], 
        'num_cycles': usage_data['num_cycles'], 
        'max_voltage': usage_data['max_voltage'], 
        'min_voltage': usage_data['min_voltage'], 
        'current_price': price_values['current_value']
    }

def get_pricing_all_vehicles(vehicle_usage_df):
    # Run in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers as needed
        all_vehicles_prices_data = list(executor.map(process_vehicle, vehicle_usage_df['vehicle_summary']))
    
    # Convert results to DataFrame
    all_vehicles_prices_df = pd.DataFrame(all_vehicles_prices_data)
    return all_vehicles_prices_df

def plot_prices_all_vehicles(vehicle_usage_df):
    # vehicle_usage_df = vehicle_usage_df.sort_values(by='vehicle_number')
    all_vehicles_prices_df = get_cached_pricing_all_vehicles(vehicle_usage_df)
    
    # Get the highest price and set Y-axis limit
    max_price = all_vehicles_prices_df['current_price'].max()
    y_axis_limit = max_price + 30000  # Adding 30k buffer
    
    # Create a bar chart
    fig = px.bar(all_vehicles_prices_df, 
                 x='vehicle_number', 
                 y='current_price', 
                 text='current_price',
                 title='Current Battery Prices of Vehicles',
                 labels={'vehicle_number': 'Vehicle Number', 'current_price': 'Current Price (INR)'},
                 color='current_price', 
                 color_continuous_scale='Viridis')
    
    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis={'categoryorder': 'total descending'},  # Sort by price
        yaxis=dict(range=[0, y_axis_limit], showgrid=False, zeroline=False),   # Set Y-axis limit
        paper_bgcolor='black', plot_bgcolor='white', font=dict(color='black')  # white theme
    )
    return fig, all_vehicles_prices_df


    