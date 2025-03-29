import plotly.graph_objs as go
from typing import Dict, List, Optional
import ollama
import streamlit as st
import pandas as pd 
import json
import time

def colored_metric(label, value, color):
    # Custom function to display colored metric cards
    st.markdown(
        f"""
        <div style="
            background-color: {color};
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            color: white;
            font-weight: bold;">
            <p style="margin:0; font-size:16px;"><strong>{label}</strong></p>
            <p style="margin:0; font-size:24px;">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def display_vehicle_battery_stats(vehicle_agg_filter_df):
    # Creating 2 rows of columns
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    #get values from filtered df 
    state_of_health = vehicle_agg_filter_df['mean_soh'][0]
    capacity = vehicle_agg_filter_df['final_capacity'][0]
    cycle_count = vehicle_agg_filter_df['num_cycles'][0]
    current_value = vehicle_agg_filter_df['current_price'][0]

    # Display colored metric cards
    with col1:
        colored_metric("State of Health", f"{state_of_health}%", "#4A90E2")  # Blue
    with col2:
        colored_metric("Capacity", f"{capacity} Ah", "#f39c12")  # orange
    with col3:
        colored_metric("Cycle Count", f"{cycle_count}", "#BD10E0")  # Purple
    with col4:
        colored_metric("Current Value", f"₹{current_value:,}", "#239b56")  # green

    # return vehicle_agg_filter_df.to_dict(orient="records")[0]

def generate_battery_reutil_prods_prompt(usage_data: Dict) -> str:
    
    battery_reutilisation_prompt = """
    Create a comprehensive list of battery repurposing options based on the following EV battery assessment parameters:
    - State of Health (SoH): {mean_soh} %
    - Temperature Excursions: {temperature_excursions}
    - Capacity: {final_capacity} Ah
    - Vehicle Age: {age_of_vehicle} km
    - Cycle Count: {num_cycles}
    - Voltage Range: {min_voltage} V to {max_voltage} V
    - Current Market Value: {current_price} INR
    
    For each repurposing option, provide:
    1. Product Name: Repurposed product Name
    2. Description: Brief explanation of the use case in max 10 words
    3. Capacity Specification: Expected capacity in the new application in kWh
    4. Recovery Value: Estimated monetary value in INR
    5. Recovery Percentage (%): Value compared to current battery price, multiplied by a factor of 100
    6. Implementation Complexity: Easy/Medium/Complex rating
    7. Market Demand: Assessment of current 2024-2025 Indian market interest
    8. Technical Viability Score: 1-10 rating based on the battery parameters

    Prioritize options that maximize value recovery while considering the battery's current condition.

    Output Format:
    - Format the response as a structured JSON array that can be directly integrated into a pricing application. 
    - No other texts should be printed except the json array under quotes ```json closing with ```
    - Resolve Getting errors such as json.decoder.JSONDecodeError: 
        Example 1: json.decoder.JSONDecodeError: Expecting ',' delimiter: line 5 column 41 (char 175) in the json string array.
        Example 2: json.decoder.JSONDecodeError: Expecting ',' delimiter: line 5 column 43 (char 169)

    - Also limit the products for upto top 5 products use cases only .
    - Store the data in the json format so that it can be later use in a pandas dataframe with column fields as: 

    {{
        "productName": <string>,
        "description": <string>,
        "capacitySpecification": <float>,
        "recoveryValue": <float>,
        "recoveryPercentage": <float>,
        "implementationComplexity": <string>,
        "marketDemand": <string>,
        "technicalViabilityScore": <float>
    }}

    """

    return battery_reutilisation_prompt.format(**usage_data) 

def get_battery_reutil_prods_report(usage_data):
    if usage_data:
        # Generate prompt
        battery_reutil_prods_prompt = generate_battery_reutil_prods_prompt(usage_data)
        # st.write(battery_reutil_prods_prompt)
        
        try:
            start_time = time.time()
            #run the prompt
            prod_response = ollama.generate(model='mistral', prompt=battery_reutil_prods_prompt)
            prod_response_report = prod_response['response']
            return prod_response_report
            
        except Exception as e:
            print(f"Error: {e}")  

def get_reutil_prod_df(usage_data):
    prod_response_report = get_battery_reutil_prods_report(usage_data)
    # st.write(prod_response_report)
    
    if prod_response_report:
        prod_res_json = json.loads(prod_response_report)
        prod_df = pd.DataFrame(prod_res_json)
        st.write("Reutilisation Product report generated!")
        # st.write(prod_df)
        return prod_df 
    else:
        st.write("products reutil report not found!")
        
def display_all_reutil_prods(usage_data):
    #get the prod_df
    prod_df = get_reutil_prod_df(usage_data)

    #display in the streamlit UI
    for index, row in prod_df.iterrows():
        with st.container():
            st.markdown(
                f"""
                <div style="border-radius: 10px; padding: 15px; margin-bottom: 10px; background-color: white; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <h4 style="color: #333; margin: 0;">{row['productName']}</h4>
                        <p style="color: #666; margin: 5px 0;">{row['description']}</p>
                        <p style="color: #555; margin: 5px 0;"><b>Capacity:</b> {row['capacitySpecification']} kWh</p>
                        <p style="color: {'orange' if row['implementationComplexity'] == 'Medium' else 'green' if row['implementationComplexity'] == 'Easy' else 'red'}; margin: 5px 0;">
                            <b>Implementation:</b> {row['implementationComplexity']}
                        </p>
                        <p style="color: #555; margin: 5px 0;"><b>Market Demand:</b> {row['marketDemand']}</p>
                    </div>
                    <div style="text-align: right;">
                        <h3 style="color: green; margin: 0;">₹{row['recoveryValue']:,.0f}</h3>
                        <p style="color: {'red' if row['recoveryPercentage'] < 50 else 'green'}; margin: 5px 0;">
                            {row['recoveryPercentage']}% recovery
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
        

# #test sample
# usage_data = {
#  'mean_soh': 72.71,
#  'temperature_excursions': 0,
#  'final_capacity': 99.24,
#  'age_of_vehicle': 77987.0,
#  'num_cycles': 1174,
#  'max_voltage': 3.36,
#  'min_voltage': 3.19,
#  'current_price':145000
# }
