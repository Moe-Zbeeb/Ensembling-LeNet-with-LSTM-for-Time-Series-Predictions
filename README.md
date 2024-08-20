# Ensembling-LeNet-with-LSTM-for-Time-Series-Predictions
### Introduction
In the realm of stock price prediction, accurately forecasting future values is a complex challenge due to the inherently volatile and dynamic nature of financial markets. Traditional approaches often rely heavily on manual feature engineering, where domain experts painstakingly craft features based on historical data. While this method has its merits, it can be limited by the subjective nature of feature selection and the inability to capture intricate patterns that may be hidden within the data.

This is where Convolutional Neural Networks (CNNs) come into play. Originally designed for image processing tasks, CNNs have proven to be highly effective in automatically extracting meaningful features from raw data. When applied to time series data, such as stock prices, CNNs can identify patterns and trends that might not be evident through traditional methods. 
**By allowing the time series to define the features themselves better (by keeping the output dimension the same),or enable a more flexible and powerful approach to feature engineering, by expanding the output space.**

In this project, we explore the integration of CNNs with LSTM networks to leverage the strengths of both models. The CNN's ability to automatically extract features from time series data is combined with the LSTM's strength in capturing temporal dependencies, providing a robust solution for stock price prediction.
