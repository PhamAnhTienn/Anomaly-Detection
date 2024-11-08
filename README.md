# Anomaly-Detection Project

## Overview

The **Anomaly-Detection** project is designed to identify anomalies in energy consumption data, utilizing a combination of **Variational Autoencoder (VAE)** and **Long Short-Term Memory (LSTM)** models. Additionally, the project leverages a **Large Language Model (LLM)** to generate explanations for detected anomalies, offering insights into potential causes. This approach aims to improve energy management efficiency by providing detailed interpretations of factors influencing energy usage.

## Key Features

1. **Anomaly Detection**: Implements VAE and LSTM models to detect irregularities in energy consumption patterns.
2. **Explanation Generation**: Utilizes an LLM to explain possible causes for detected anomalies, enhancing interpretability.
3. **Configurable Parameters**: Allows for easy customization of model parameters and settings through the `params.yaml` file.

## Methodology

The workflow of the anomaly detection system is as follows:

![Anomaly Detection Workflow](https://github.com/user-attachments/assets/8280d48c-4b95-484c-8be4-187c80f57145)

### LLM Explanation Workflow

After detecting anomalies, the LLM is used to generate contextual explanations:

![LLM Workflow](https://github.com/user-attachments/assets/cf655289-01dd-4d56-8a31-52266e89ef08)

## Installation

To set up the project, first install the required dependencies:

```bash
pip install -r requirements.txt
```

Once the dependencies are installed, you can execute the main script as follows:

```bash
python main.py
```

## Configuration

Parameters such as model settings, training details, and data paths are configured in the params.yaml file, making it easy to adapt the project to different datasets or use cases.

## References

[1] S. Lin, R. Clark, R. Birke, S. Schönborn, N. Trigoni and S. Roberts, "Anomaly Detection for Time Series Using VAE-LSTM Hybrid Model," ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 4322-4326, doi: 10.1109/ICASSP40776.2020.9053558. keywords: {Correlation;Time series analysis;Signal processing;Detection algorithms;Speech processing;Anomaly detection;Unsupervised learning;Anomaly Detection;Time Series;Deep Learning;Unsupervised Learning},
