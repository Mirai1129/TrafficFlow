# TrafficFlow

This study aims to predict traffic flow and explain the trained model.
Use a multi-module hybrid model to make predictions on traffic datasets. 
SHAP was used to explain the model.    

<div align="center">
  <img src="https://github.com/yuntech-bdrc/TrafficFlow/blob/main/image/%E6%B5%81%E7%A8%8B%E5%9C%96.jpg" alt="flowchart" width="600"/>
</div>

## Traffic Features

### 🚗 Traffic Data with Temporal Context
- **Total Volume**: Traffic volume within one hour  
- **Time**: Hour of the day (0 to 23)  
- **Day**: Indicator for day of the week (Monday to Sunday)  
- **Holiday**: Indicator of whether the day is a holiday or not  

### 🔁 Historical Cycle Features
- Incorporates traffic flow from the **same time** over the **past 7 days**


## Requirement  
<!--  
``` shell
pip install -r requirement.txt
``` -->

## Performance
### Traffic data with temporal context
| Dataset      | Feature                        | MAE                       | MAPE (%)                   | RMSE                       |
|:-----------:|:--------------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|
| Taichung  | TotalVolume                    | 73.98 ± 1.57               | 17.23 ± 0.005              | 104.66 ± 2.52              |
|           | TotalVolume + Time             | 69.79 ± 2.98 (-5.56%)      | 16.11 ± 0.006 (-6.51%)     | 98.68 ± 3.45 (-5.73%)      |
|           | TotalVolume + Day              | 72.06 ± 1.62 (-2.59%)      | 16.95 ± 0.006 (-1.63%)     | 101.30 ± 2.58 (-3.21%)     |
|           | TotalVolume + Holiday          | 72.58 ± 0.67 (-1.90%)      | 17.55 ± 0.004 (+1.86%)     | 101.50 ± 1.58 (-3.02%)     |
|           | TotalVolume + Time + Day       | 67.97 ± 2.82 (-8.11%)      | 15.71 ± 0.009 (-8.83%)     | 96.91 ± 4.33 (-7.40%)      |
|           | TotalVolume + Time + Holiday   | 66.42 ± 1.66 (-10.22%)     | 15.36 ± 0.006 (-10.84%)    | 95.29 ± 2.76 (-8.97%)      |
|           | TotalVolume + Day + Holiday    | 71.59 ± 0.82 (-3.22%)      | 17.22 ± 0.004 (-0.06%)     | 99.79 ± 1.09 (-4.66%)      |
|           | **TotalVolume + Time + Day + Holiday** | **65.97 ± 1.61 (-10.82%)** | **15.33 ± 0.006 (-11.03%)** | **94.45 ± 2.67 (-9.78%)**  |
| Minnesota | TotalVolume                    | 221.54 ± 11.26             | 11.23 ± 0.007              | 328.91 ± 14.69             |
|           | TotalVolume + Time             | 200.99 ± 7.86 (-9.28%)     | 9.89 ± 0.005 (-11.93%)     | **300.23 ± 9.80 (-8.72%)** |
|           | TotalVolume + Day              | 199.58 ± 7.90 (-9.91%)     | 9.49 ± 0.005 (-15.49%)     | 320.54 ± 10.44 (-2.54%)    |
|           | TotalVolume + Holiday          | 209.65 ± 6.28 (-5.37%)     | 9.59 ± 0.003 (-14.60%)     | 334.70 ± 8.60 (+1.76%)     |
|           | TotalVolume + Time + Day       | 194.31 ± 6.51 (-12.29%)    | 8.95 ± 0.004 (-20.30%)     | 318.40 ± 13.87 (-3.20%)    |
|           | TotalVolume + Time + Holiday   | 196.51 ± 5.13 (-11.29%)    | 9.11 ± 0.006 (-18.89%)     | 349.79 ± 6.22 (+6.35%)     |
|           | TotalVolume + Day + Holiday    | 212.20 ± 9.18 (-4.42%)     | 9.98 ± 0.005 (-11.14%)     | 334.37 ± 11.81 (+1.66%)    |
|           | **TotalVolume + Time + Day + Holiday** | **190.38 ± 7.43 (-14.07%)** | **8.32 ± 0.003 (-25.90%)** | 318.26 ± 11.59 (-3.24%)    |


### Feature Module Comparison in Traffic Forecasting
| Dataset       | Model     | MAE                         | MAPE (%)                     | RMSE                        |
|:--------------:|:----------:|:---------------------------:|:----------------------------:|:---------------------------:|
| Taichung         | M1       | 65.97 ± 1.608               | 15.33 ± 0.006                | 94.454 ± 2.668              |
|              | M2       | 74.41 ± 1.10 (+12.79%)      | 18.25 ± 0.003 (+19.05%)      | 106.62 ± 1.12 (+12.88%)     |
|              | M1+M2    | **64.03 ± 0.79 (-2.95%)**   | **14.64 ± 0.004 (-4.50%)**   | **91.42 ± 1.03 (-3.22%)**   |
| Minnesota   | M1       | 190.38 ± 7.43               | 8.32 ± 0.003                 | 318.26 ± 11.594             |
|              | M2       | 311.88 ± 3.99 (+63.85%)     | 14.23 ± 0.003 (+70.95%)      | 601.33 ± 8.64 (+88.94%)     |
|              | M1+M2    | **182.47 ± 9.01 (-4.15%)**      | **7.49 ± 0.004 (-9.99%)**        | **303.53 ± 16.48 (-4.63%)**     |


## Datasets

### 📍 Taichung (Taiwan) Traffic Dataset
- **Source**: [Taichung Real-Time Traffic Information Platform](https://e-traffic.taichung.gov.tw/RoadGrid/Pages/VD/History2.html)
- **Location**: Intersection of Taiwan Boulevard and Henan Road (northbound)
- **Period**: Full year of 2024
- **Records**: 8,784 hourly entries

---

### 📍 Minnesota (USA) Traffic Dataset
- **Source**: [Minnesota Department of Transportation](https://www.dot.state.mn.us/)
- **Location**: I-94 Highway, ATR Station 301
- **Period**: Full year of 2017
- **Records**: 8,760 hourly entries


## References
[SHAP](https://arxiv.org/abs/1705.07874 "A Unified Approach to Interpreting Model Predictions")  
