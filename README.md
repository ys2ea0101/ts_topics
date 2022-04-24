# README #

This repo includes code for running multivariate TS and exploring different features

### Tasks ###

For running just multivariate TS, run the code run_mv_iter.py or run_mv_direct.py
The experimental setup is the same as in this paper: https://arxiv.org/pdf/1905.03806.pdf

For multivariate TS with features, run s2s_retail.py


### How do I get set up? ###

One needs the dataset to be able to run this. 

To run the experiment on retail dataset, first get the 'CB_MULTI_FORECASTING_K01_retaildataset.csv' dataset, this is available on S3
Then run 'retail_dataset_analyze.py', make sure to uncomment the lines (71-74) that reads the original data and save the time series and exogenous data as numpy arrays
Then the file 's2s_retail.py' can read the 2 npy files from the last step

To get the datasets for multivariate experiments, run the 'download-data.sh' file to get the npy files


### Code structure ####

the "multivariate_xy" class processes data to a format the model can directly input
it has 2 options, iter or direct. It can also handle exogenous data if present
However it is very inefficient and can be improved using better numpy manipulations


### models included ####

the code includes most popular DL models for TSF, 1D CNN, TCN, LSTM, Transformer, 
as long as they have the same input and output format, the model can be switched relatively easy.  
