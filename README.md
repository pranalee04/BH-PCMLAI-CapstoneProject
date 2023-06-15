# Capstone Project
This is the capstone project for my ML/AI course. In this project, I ML/AI model to detect malicious URL
## Project Title:Capstone Project: Malicious URL detection using Machine Learning and Artificial Intelligence

**Author:** Pranalee Peshne

## Executive Summary
The Web has long become a major platform for online criminal activities. URLs are used as the main vehicle in this domain. To counter this issues security community focused its efforts on developing techniques for mostly blacklisting of malicious URLs. While successful in protecting users from known malicious domains, this approach only solves part of the problem. The new malicious URLs that sprang up all over the web in masses commonly get a head start in this race. . In this project I explored a lightweight approach to detection and categorization of the malicious URLs according to their attack type and show that lexical analysis is effective and efficient for proactive detection of these URLs.
## Dataset
The Malicious URL dataset was taken from [Kaggle](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)
The dataset  include a large number of examples of Malicious URLs so that a machine learning-based model can be developed to identify malicious urls and stop them in advance before infecting computer system or spreading through inteinternet. The dataset consist of 651,191 URLs, out of which 428103 benign or safe URLs, 96457 defacement URLs, 94111 phishing URLs, and 32520 malware URLs. The Kaggel dataset is preprocessed databset, the original source of data is from [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/url-2016.html). For increasing phishing and malware URLs, a Malware domain black list dataset was used. To increased benign URLs  faizan git repo was used. Phishing URLs were increased using Phishtank dataset and PhishStorm dataset.  In nutshell the dataset used in this project is is collected from different sources. The URLs were collected from different sources into a separate data frame and finally merge them to retain only URLs and their class type.

## Methodology
The methodology of this project includes data collection, data cleaning, exploratory data analysis, feature engineering, model selection and training, model evaluation, and model interpretation. The execution and presentation of this project is done using the CRISP DM methodology

## Project Structure
1. notebook
2. notebook/images
3. data 
4. data/finaldataset
5. Report.pdf 
6. README.md
## Programming language
The programming language used is Python, and the libraries used were: Pandas, Plotly, Seaborn, Matplotlib, Numpy , Sklearn,NLTK,Statsmodels,colorama, urllib.parse,tld,re,Word2Vec, whois, wordcloud, gensim  tldextract hmmlearn Pillow WordCloud word_tokenize,Keras, Tensorflow
## Code,data processing, modelling and analysis
The complete analysis, including observations,commentss so all the work done is contained in the following Jupiter Notebook and Report.pdf:
1. Data processing, plots and data analysis is is contained in the following Jupiter Notebook [FeaturePreprocessingAndAnalysis-1.ipynb](https://github.com/pranalee04/BH-PCMLAI-CapstoneProject/blob/main/notebook/FeaturePreprocessingAndAnalysis-1.ipynb)

2. Baseline model exploration is contained in the following Jupiter Notebook [ModelExperimentationAndAnalysis-2.ipynb](https://github.com/pranalee04/BH-PCMLAI-CapstoneProject/blob/main/notebook/ModelExperimentationAndAnalysis-2.ipynb)
3. Advance model exploration is contained in the following Jupiter Notebook [AdvanceModelingExperimentation-3.ipynb](https://github.com/pranalee04/BH-PCMLAI-CapstoneProject/blob/main/notebook/AdvanceModelingExperimentation-3.ipynb)
4. Exploation of Nerual Networks Keras Tensorflow models contained in the following Jupiter Notebook [KerasTensorflowExperimentation-4.ipynb](https://github.com/pranalee04/BH-PCMLAI-CapstoneProject/blob/main/notebook/KerasTensorflowExperimentation-4.ipynb)
5. Selcted modle LGBMClassifier and its hyperparameter tuning and analysis is contained in the following Jupiter Notebook  [SelectedModel-LGBMClassifier-5.ipynb](https://github.com/pranalee04/BH-PCMLAI-CapstoneProject/blob/main/notebook/SelectedModel-LGBMClassifier-5.ipynb)
6. Selcted modle RandomForestClassifie and its hyperparameter tuning and analysis is contained in the following Jupiter Notebook  [SelectedModel-RandomForestClassifier-6.ipynb](https://github.com/pranalee04/BH-PCMLAI-CapstoneProject/blob/main/notebook/SelectedModel-RandomForestClassifier-6.ipynb)
7. Selcted modle RandomForestClassifie and its hyperparameter tuning and analysis is contained in the following Jupiter Notebook  [FinalSelectedModel-XGBClassifier-7.ipynb](https://github.com/pranalee04/BH-PCMLAI-CapstoneProject/blob/main/notebook/FinalSelectedModel-XGBClassifier-7.ipynb)

## Result
1. Processed dataset saved in CSV file:[maliciousurl_processed.csv](https://github.com/pranalee04/BH-PCMLAI-CapstoneProject/blob/main/data/finaldataset/maliciousurl_processed.csv)
2. Model comparison can be see in the [bar-plot](https://github.com/pranalee04/BH-PCMLAI-CapstoneProject/tree/main/notebook/images/model-accuracy-comparision.png)
3. ExtraTreeClassifier model gave the best accuracy in initial investigation. However, XGBClassifier, RandomForestClassifier and LGBMClassifier were selected for futher evaluation baseed on accuracy and feature impact.
4. The finding were summerized in the report [Report.pdf](https://github.com/pranalee04/BH-PCMLAI-CapstoneProject/tree/main/Report.pdf)

## Next Steps
1. Experiment further with hyperparameter to identity the best fit for the selected model
2. Put the model to practical use 
2. Explore deploying the model to MLOps Platform such as AWS SageMaker, Azure ML and Google Cloud ML

## References
 Course: UC BERKELEY Engineering and Haas Professional Certificate in Machine Learning & Artificial Intelligence course content, tutorial, videos, etc.
 Home - Keras Documentation - https://keras.io/
 TensorFlow | TensorFlow - https://www.tensorflow.org/
 SKlearn|https://scikit-learn.org/
 Kaggle|https://www.kaggle.com/code/thisishusseinali/malicious-url-detection
 Canadian Institute for Cybersecurity|https://www.unb.ca/cic/datasets/url-2016.htm
 Online Examples|https://github.com/Colorado-Mesa-University-Cybersecurity
 People: Jessica Cervi, Savio Saldanha, Holly Bees, and Leanna Biddle,

