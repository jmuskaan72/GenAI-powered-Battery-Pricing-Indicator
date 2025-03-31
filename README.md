# GenAI powered Battery Pricing Indicator Bot

## Overview
Engineered a GenAI-powered Streamlit-based EV Battery Dynamic Pricing Indicator with 12-month dynamic pricing forecasting capabilities, as per the vehicle on-road usage and reutilisation with other devices. 
Battery Pricing LLM Bot integrates battery descriptions with pricing-affecting factors and training feature parameters using [Ollama Mistral-7B](https://ollama.com/library/mistral) model. This system enables:
- Battery pricing determination based on health state and condition.
- Battery reutilization with other devices as per battery state.

## Benefits of GenAI Over Traditional ML Models in Battery Pricing

### **Real-time Dynamic Pricing Analysis**
Generative AI helps develop dynamic pricing models that adjust based on:
- Market conditions
- On-road vehicle usage
- Competitor pricing
- Customer demand

### **Adaptive Learning & Market Intelligence**
- Traditional ML requires regular retraining; GenAI continuously learns from new data.
- Uses few-shot learning to adapt to new scenarios.

### **Synthetic Data Generation**
- Factors in environmental conditions and usage patterns.
- Considers geographic and route-specific degradation factors.

### **Complex Pattern Recognition**
GenAI surpasses traditional ML in identifying subtle correlations like:
- Driving behavior and battery degradation.
- Environmental conditions and their impact on performance.
- Charging patterns and cell deterioration.

### **Scenario Generation**
- Creates diverse scenarios based on historical data.
- Enables proactive pricing strategy adjustments.

### **Multi-Dimensional Valuation Parameters**
- **Battery Health Prediction:** SoH estimation, temperature excursions, max/min voltage, cycle count, degradation, and vehicle age.
- **CSV Analyzer:** Extracts on-road vehicle usage stats dynamically, analyzes telemetry data, and derives pricing insights.

## Technical Flow

1. **Process vehicle telemetry data** (voltage, current, temperature patterns).
2. **Analyze charging-discharging cycles and patterns.**
3. **Generate dynamic pricing recommendations.**

## Battery LLM Pricing Indicator - Streamlit App Details

- **Price Forecasting:** Predicts battery pricing behavior up to 12 months based on on-road vehicle usage.
- **Battery Reutilization Products (Top 5):**
  - Product name & description
  - Capacity required
  - Implementation level
  - Market demand
  - Recovery value in INR
  - Recovery percentage

## In-Depth Technical Features

- **Detailed Price Info** for selected vehicles.
- **Dynamic multi-parameter simulation** for pricing evaluation.

## Next Steps

- **LLM Evaluation & Accuracy:** Confidence metrics.
- **Industry Analysis in Real-Time.**
- **Optimization of Running Time.**

## References
- [Complete Guide to EV Batteries in India: Pricing, Top Manufacturers & Future Trends (2024)](https://lohum.com/scrap-battery-price-calculator/)
- [Lohum Battery Price Calculator](https://lohum.com/scrap-battery-price-calculator/)
- Full details for the architecture and Streamlit UI explained can be found in the pdf attached. 
