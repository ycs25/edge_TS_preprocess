# edge_TS_preprocess
A simple demo of edge-cloud collaborative time-series forecasting.

## Description
Each edge device uses predetermined parameters to normalize reading data collected and upload the results to central cloud. Then cloud use pretrained model to make predictions or detect anomalies. This method not only saves computation time and make on-edge preprocessing possible, but also reduces the size of data packets sent from edge devices. Those processed data can be easily restored at central cloud if necessary. 

## Challenges
1. ~~Find suitable time-series data to simulate real streaming IoT data~~
2. Record calculation time and/or package size to show efficiency
3. Find the preprocessing method that yields best model accuracy
4. Compare multiple "seasonal" model vs one model performance comparison
5. Compare centered normalization cost with on-edge normalization cost
6. Try simple anomaly detection algorithm at edge device
7. Construct a (pseudo) distributed structure
