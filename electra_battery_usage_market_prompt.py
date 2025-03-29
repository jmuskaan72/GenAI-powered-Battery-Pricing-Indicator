import plotly.graph_objs as go
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import ollama
import streamlit as st
import re 
import json
import time 

@dataclass
class BatterySpecs:
    capacity_kwh: float = 16.5
    nominal_capacity_ah: int = 326
    nominal_voltage: float = 51.2
    pack_config: str = "2P 16S"
    ip_rating: str = "IP67"
    depth_of_discharge: int = 90
    cooling_type: str = "Passive"

@dataclass
class OperatingParams:
    charge_current: float = 0.5  # C-rate
    discharge_current: float = 0.3  # C-rate
    max_discharge_current: float = 1.5  # C-rate
    voltage_range: tuple = (46.4, 58.4)
    temp_range_charge: tuple = (0, 45)
    temp_range_discharge: tuple = (0, 50)
    optimal_temp_range: tuple = (25, 35)

@dataclass
class SafetyStatus:
    thermal_stability: int = 2  # Level 1-4
    thermal_runway: int = 4
    nail_penetration: int = 4
    vibration_test: int = 2
    crush_test: int = 2
    overcharge: int = 4
    over_discharge: int = 2
    short_circuit: int = 2

def generate_enhanced_pricing_prompt(
        battery_specs: BatterySpecs,
        operating_params: OperatingParams,
        safety_status: SafetyStatus,
        usage_data: Dict
    ) -> str:
    
    base_prompt = """
    Role: You are an expert Electric Vehicle battery pricing analyst specializing in Electra battery systems with deep knowledge of both Indian and global lithium-ion battery markets as of 2024-2025.

    Technical Specifications:
    1. Battery Pack Details:
        - Capacity: {capacity_kwh} kWh
        - Nominal Capacity: {nominal_capacity_ah} Ah
        - Nominal Voltage: {nominal_voltage} V
        - Pack Configuration: {pack_config}
        - IP Rating: {ip_rating}
        - Depth of Discharge: {depth_of_discharge}%
        - Cooling System: {cooling_type}

    2. Operating Parameters:
        - Charging Current (Continuous): {charge_current}C
        - Discharge Current (Continuous): {discharge_current}C
        - Maximum Discharge Current: {max_discharge_current}C (30 sec)
        - Voltage Range: {voltage_min}V - {voltage_max}V
        - Temperature Ranges:
            * Charging: {temp_charge_min}°C to {temp_charge_max}°C
            * Discharging: {temp_discharge_min}°C to {temp_discharge_max}°C
            * Optimal: {temp_optimal_min}°C to {temp_optimal_max}°C

    3. Safety Status Assessment (Level 1-4, where 4 is critical):
        Thermal Stability:
        - Basic Stability: Level {thermal_stability}
        - Thermal Runway: Level {thermal_runway}
        
        Mechanical Stability:
        - Nail Penetration: Level {nail_penetration}
        - Vibration Resistance: Level {vibration_test}
        - Crush Resistance: Level {crush_test}
        
        Protection Systems:
        - Overcharge Protection: Level {overcharge}
        - Over-discharge Protection: Level {over_discharge}
        - Short Circuit Protection: Level {short_circuit}

    4. Usage History:
        - State of Health : {mean_soh} 
        - Temperature Excursions: {temperature_excursions}
        - Final Capacity (in Ah units): {final_capacity}
        - Age of battery operating (in kms): {age_of_vehicle}
        - Cycle count : {num_cycles}
        - Max Cell Voltage: {max_voltage}
        - Min Cell Voltage: {min_voltage}
    
    Premium features adding to the cost:
    1. IP67 rating (+5-8%)
    2. Advanced protection systems:
        - Overcharge Protection (Level 4)
        - Short Circuit Protection
    3. 2P 16S configuration with 326 Ah nominal capacity
    4. 90% Depth of Discharge capability

    Market Context (2024-2025):
    - Consider the cost of lithium-ion battery pack costs in India for per kWh for automotive grade batteries
    - Premium features adding to the cost:
        1. IP67 rating (+5-8%)
        2. Advanced protection systems:
            - Overcharge Protection (Level 4)
            - Short Circuit Protection
        3. 2P 16S configuration with 326 Ah nominal capacity
        4. 90% Depth of Discharge capability
    - Lithium-ion Material Costs - Current lithium carbonate spot price trends:
        Lithium carbonate prices are stabilizing globally, but local manufacturing costs in India may fluctuate due to currency volatility and geopolitical factors.
    - Improving Local Maintenance Capabilities - Local cell manufacturing capacity utilization:
        The growth of EV maintenance networks could help maintain residual value, mitigating sharper depreciation. 
    - Grid Stability impact on Charging and Energy Costs:
        Rising electricity rates and grid stability issues could impact the long-term desirability of EV batteries, reducing demand and thus residual valu
    - Regional temperature patterns affecting battery life
    - Regulatory compliance requirements
    - Insurance risk assessments based on safety ratings

    Task:
    Generate a comprehensive price assessment in INR considering:
    1. Technical health factors:
        - Impact of safety test results on value
        - Operating parameter deviations
        - Thermal management effectiveness
        - Protection system reliability

    2. Usage pattern impact:
        - Temperature exposure history
        - State of Health
        - Age of battery
        - Final capacity attained 
        - Maintenance record correlation
        - Battery residual value based on usage patterns

    3. Market adjustments:
        - Safety rating impact on insurance
        - Regional climate considerations
        - Grid infrastructure quality
        - Local maintenance capability

    4. Value Forecast - Factors Driving Over the Next 12 Months given current year 2025 market considerations for battery pricing: 
        - State of Health (SOH) Decline:
         Battery SOH is already at {mean_soh}. As the vehicle continues operating, a typical 1-2% decline in SOH over six months can be expected. This directly reduces the battery pack's capacity and resale value.
        
        - Thermal Stress:
         Given a passive cooling system and high temperature exposure history ({temperature_excursions} excursions), performance degradation due to overheating is likely to persist, especially in regions with warm climates.
        
        Capacity Loss:
        - Final capacity of {final_capacity} Ah (compared to a nominal 326 Ah) suggests further capacity degradation is imminent, contributing to reduced value.

        - Give me battery price INR value only in the value_forecast field in the output format not the impact factor (+ or -).
        - Make sure the Confidence_level for value forecast in the output format is in percentage value ranging between 0 to 100 only. 
        - Resolve key error of 1 month or 1 months in the value_forecast output and treat it same as 1_months only 

    Output Format:
    {{
        "current_value": <float>,        
        "technical_health_impact": {{
            "safety_rating_adjustment": <float>,
            "thermal_management": <float>,
            "protection_systems": <float>
        }},
        "usage_impact": {{
            "battery_residual_value":<float>,
            "temperature_exposure": <float>,
            "state_of_health": <float>,
            "age_of_battery":<float>,
            "final_capacity":<float>,
            "maintenance_quality": <float>
        }},
        "market_factors": {{
            "insurance_risk": <float>,
            "regional_climate": <float>,
            "support_infrastructure": <float>
        }},
        "overall_health_score": <float>,
        "safety_risk_score": <float>,
        "value_forecast": {{
            "1_months": <float>,
            "3_months": <float>,
            "6_months": <float>,
            "12_months": <float>,
            "confidence_level": <float>
        }}
    }}
    """
    
    return base_prompt.format(
        capacity_kwh=battery_specs.capacity_kwh,
        nominal_capacity_ah=battery_specs.nominal_capacity_ah,
        nominal_voltage=battery_specs.nominal_voltage,
        pack_config=battery_specs.pack_config,
        ip_rating=battery_specs.ip_rating,
        depth_of_discharge=battery_specs.depth_of_discharge,
        cooling_type=battery_specs.cooling_type,
        charge_current=operating_params.charge_current,
        discharge_current=operating_params.discharge_current,
        max_discharge_current=operating_params.max_discharge_current,
        voltage_min=operating_params.voltage_range[0],
        voltage_max=operating_params.voltage_range[1],
        temp_charge_min=operating_params.temp_range_charge[0],
        temp_charge_max=operating_params.temp_range_charge[1],
        temp_discharge_min=operating_params.temp_range_discharge[0],
        temp_discharge_max=operating_params.temp_range_discharge[1],
        temp_optimal_min=operating_params.optimal_temp_range[0],
        temp_optimal_max=operating_params.optimal_temp_range[1],
        thermal_stability=safety_status.thermal_stability,
        thermal_runway=safety_status.thermal_runway,
        nail_penetration=safety_status.nail_penetration,
        vibration_test=safety_status.vibration_test,
        crush_test=safety_status.crush_test,
        overcharge=safety_status.overcharge,
        over_discharge=safety_status.over_discharge,
        short_circuit=safety_status.short_circuit,
        **usage_data
    )

def get_price_analysis_prompt(usage_data):
    # Create instances
    specs = BatterySpecs()
    params = OperatingParams()
    safety = SafetyStatus()
    
    # Generate prompt
    battery_stats_usage_price_prompt = generate_enhanced_pricing_prompt(specs, params, safety, usage_data)
    return battery_stats_usage_price_prompt

def get_price_analysis_report(usage_data):
    if usage_data:
        # Combine the usage stats with the battery static data properties prompt  
        battery_stats_usage_price_prompt = get_price_analysis_prompt(usage_data)
        
        try:
            start_time = time.time()
            price_response = ollama.generate(model='mistral', prompt=battery_stats_usage_price_prompt)
            # st.write("Success!")
            
            price_analysis_report = price_response['response']
            # print("Output displayed in %.2f seconds" % (time.time() - start_time))
            
            return price_analysis_report
        except Exception as e:
            print(f"Error: {e}")   
            
def get_price_values(price_analysis_report):
    # Regular expressions to extract the required values
    pattern = r'"current_value": (\d+)|"1_month[s]?": (\d+)|"3_month[s]?": (\d+)|"6_month[s]?": (\d+)|"12_month[s]?": (\d+)|"confidence_level": ([\d.]+)'

    # get the matches 
    matches = re.findall(pattern, price_analysis_report)
    keys = ["current_value", "1_months", "3_months", "6_months", "12_months", "confidence_level"]
    
    # Build dictionary with corresponding keys
    price_final_dict = {keys[i]: float(value) if '.' in value else int(value) 
                    for match in matches 
                    for i, value in enumerate(match) if value
                    }
    return price_final_dict

import plotly.graph_objects as go

def plot_price_forecasting_values(price_final_dict, vehicle_id):
    # Extract x and y values for the plot
    time_periods = ["Current Value", "1 Months", "3 Months", "6 Months", "12 Months"]
    values = [
        price_final_dict.get('current_value', None),
        price_final_dict.get('1_months', None),
        price_final_dict.get('3_months', None),
        price_final_dict.get('6_months', None),
        price_final_dict.get('12_months', None),
    ]
    
    # Handle missing values by forward-filling or replacing with zero
    cleaned_values = []
    last_valid_value = None
    for v in values:
        if v is not None:
            last_valid_value = v  # Update last valid value
        cleaned_values.append(last_valid_value)

    # Calculate percentage decline for hover
    percent_decline = [None] + [
        f"{((cleaned_values[i] - cleaned_values[i - 1]) / cleaned_values[i - 1]) * 100:.2f}%" 
        if cleaned_values[i - 1] is not None else "N/A"
        for i in range(1, len(cleaned_values))
    ]

    # Extract confidence score
    confidence_score = price_final_dict.get('confidence_level', 'N/A')  # Fallback if missing
    
    # Create the plot
    fig = go.Figure()
    fig.update_layout(width=800, height=400)
    
    # Add line plot for value changes
    fig.add_trace(go.Scatter(
        x=time_periods,
        y=cleaned_values,
        mode='lines+markers+text',
        marker=dict(size=10),
        textposition="top center",
        hovertext=[
            f"{value:,.0f} INR (Current Price)" if decline is None 
            else f"{value:,.0f} INR ({decline} change)"
            for value, decline in zip(cleaned_values, percent_decline)
        ],
        hoverlabel=dict(
            bgcolor='darkblue',
            font=dict(
                color='white',
                size=16
            )
        ),
        hoverinfo="text"
    ))

    # Add confidence score at the top right
    fig.add_annotation(
        x=1, y=1.15, 
        xref="paper", yref="paper",
        text=f"Confidence Score: {confidence_score}%",
        showarrow=False,
        font=dict(size=14, color="yellow"),
        align="right"
    )

    # Layout customization
    fig.update_layout(
        title=f"Market Value Forecast for Vehicle {vehicle_id}",
        xaxis_title="Time Period",
        yaxis_title="Battery Price (INR)",
        showlegend=False,
        margin=dict(l=50, r=50, t=70, b=50)
    )
    
    return fig

def latest_market_news_headlines():
    battery_pricing_market_news_prompt = """
    ### EV Fleet Battery Price Intelligence Prompt

    OUTPUT FORMAT REQUIREMENTS:

    * Generate 4 price-focused headlines started with * bullet mark specifically for EV fleet operations analysis: 
    * Also note, show the bullet points only and nothing else. Do not display the cost component in the output.
    * The ouput format should not contain any detailed explanation of the headers, only the 4 bullet pointers should be displayed.

    Each headline must include:
    * Specific ₹/kWh value
    * Vehicle category impact
    * Fleet size considerations
    * Operational cost impact
    
    Example headlines:
    * "20kWh LFP Packs Hit ₹6,500/kWh for 1000+ Fleet Orders"
    * "New BMS Cuts Light EV Battery Costs by ₹800/kWh"
    * "5-Year Battery Warranty Now at ₹7,200/kWh All-Inclusive"
    * "Fleet Battery Replacement Costs Drop to ₹5,900/kWh"

    
    BATTERY COST METRICS (FLEET FOCUS)
    Track & report:
    * Cost per vehicle category:
      - Light Commercial EVs: ₹/kWh for 20-40kWh packs
      - Medium Commercial EVs: ₹/kWh for 40-80kWh packs
      - Heavy Commercial EVs: ₹/kWh for 80-150kWh packs
    * Price alerts when crossing:
      - Base threshold: ₹8,000/kWh
      - Fleet bulk purchase: ₹7,000/kWh
      - Large fleet deal: ₹6,000/kWh
    
    TOTAL COST INDICATORS
    Monitor & report:
    * Battery replacement costs
    * Warranty terms changes
    * Cycle life improvements
    * Temperature degradation factors
    * Charging cycle costs
    * End-of-life value
    * Recycling credit value
    
    FLEET-SPECIFIC PARAMETERS
    Track changes in:
    * Price/km for battery operation
    * Cost per charging cycle
    * Battery life in delivery cycles
    * Fast charging impact on costs
    * Temperature impact on range
    * Urban vs highway efficiency
    * Load impact on degradation
    
    AI MODEL INPUTS
    Report values affecting:
    * Base battery pack price
    * Chemistry-specific costs
    * Thermal management costs
    * BMS system costs
    * Installation/integration costs
    * Maintenance interval costs
    * Replacement planning costs
    
    BOT INTEGRATION PARAMETERS
    Each headline should include:
    * Base price component
    * Volume discount factor
    * Warranty cost component
    * Operating cost impact
    * Regional variance factor
    * Chemistry type factor
    * Life cycle cost indicator
    
    Keep headlines focused on:
    * Price changes > 5%
    * Warranty term changes
    * Bulk purchase offers
    * Total cost of ownership updates
    * Regional price variations
    * Chemistry-specific pricing
    * Fleet-scale opportunities
    """
    
    try:
        market_news_response = ollama.generate(model='mistral', prompt=battery_pricing_market_news_prompt)        
        latest_market_news_report = market_news_response['response']
        return latest_market_news_report
    except Exception as e:
        print(f"Error: {e}")   

    


