# edge_TS_preprocess
A simple demo of edge-cloud collaborative time-series forecasting.

## Description
Each edge device uses predetermined parameters to normalize reading data collected and upload the results to central cloud. Then cloud use pretrained model to make predictions or detect anomalies. This method not only saves computation time and make on-edge preprocessing possible, but also reduces the size of data packets sent from edge devices. Those processed data can be easily restored at central cloud if necessary. 

## How to Run the demo app
1. Install dependencies using `pip install -r requirements.txt` in Codespace
2. Run in terminal: `streamlit run streamlit_dashboard.py`

## Attension with the Data Loading Codes
The raw data file path are pointing to data outside of the GitHub repo. You need to download data files to `../data/raw` for the `preprocessing/loading_XXX_data.py` codes to run successfully.
