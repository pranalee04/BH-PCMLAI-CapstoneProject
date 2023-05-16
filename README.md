# Capstone Project
This is the capstone project for my ML/AI course. In this project, I ML/AI model to detect malicious URL
## Project Title: Malacious URL detection using Machine Learning and Artificial Intelligence

**Author:** Pranalee Peshne
## Executive Summary
The Web has long become a major platform for online criminal activities. URLs are used as the main vehicle in this domain. To counter this issues security community focused its efforts on developing techniques for mostly blacklisting of malicious URLs. While successful in protecting users from known malicious domains, this approach only solves part of the problem. The new malicious URLs that sprang up all over the web in masses commonly get a head start in this race. . In this project I explored a lightweight approach to detection and categorization of the malicious URLs according to their attack type and show that lexical analysis is effective and efficient for proactive detection of these URLs.
## Dataset
The Malicious URL dataset was taken from [Kaggle](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)
The dataset  include a large number of examples of Malicious URLs so that a machine learning-based model can be developed to identify malicious urls and stop them in advance before infecting computer system or spreading through inteinternet. The dataset consist of 651,191 URLs, out of which 428103 benign or safe URLs, 96457 defacement URLs, 94111 phishing URLs, and 32520 malware URLs. The Kaggel dataset is preprocessed databset, the original source of data is from [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/url-2016.html). For increasing phishing and malware URLs, a Malware domain black list dataset was used. To increased benign URLs  faizan git repo was used. Phishing URLs were increased using Phishtank dataset and PhishStorm dataset.  In nutshell the dataset used in this project is is collected from different sources. The URLs were collected from different sources into a separate data frame and finally merge them to retain only URLs and their class type.
## Methodology
The methodology of this project includes data collection, data cleaning, exploratory data analysis, feature engineering, model selection and training, model evaluation, and model interpretation.
## Project Structure
1. notebook
2. notebook/images
3. data and 
4. README.md
## Programming language
The programming language used is Python, and the libraries used were: Pandas, Plotly, Seaborn, Matplotlib, Numpy , Sklearn,Statsmodels,plotly.io,colorama, urllib.parse,tld,re
## Code and Analysis
The complete analysis, including observations,commentss so all the work done is contained in the following Jupiter Notebook:
[capstoneproject.ipynb](https://github.com/pranalee04/BH-PCMLAI-CapstoneProject/blob/main/notebook/capstoneproject.ipynb)
## Result
Model comparison can be see in the [bar-plot](https://github.com/pranalee04/BH-PCMLAI-CapstoneProject/tree/main/notebook/images/ModelAccuracyComparison)
## Conclusion
Random Forest Classifier followed by Extra Tree Classifier were the best model for detecting malacious URL in the `malicious_phish.csv` dataset
## Next Steps
1. Improve above explored models, play with various hyper parameters.
2. Put the model to practical use
3. I wanted to  explore LSTM and Deep learning for the malacious URL problem
