# Predictive_Maintenance_UCI_Modes
AI4I 2020 Predictive Maintenance Dataset Data Set -- Predicting Failure Modes


# Random Forrest --- predict Failure Mode #
# UCI dataset "AI4I 2020 Predictive Maintenance Dataset Data Set" #

Online-Source [here](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset#)

 AI4I 2020 Predictive Maintenance Dataset Data Set
Download: Data Folder, Data Set Description

Abstract: The AI4I 2020 Predictive Maintenance Dataset is a synthetic dataset that reflects real predictive maintenance data encountered in industry.

| | | | | | |	
|:---|:---:|:---|:---:|:---|:---:|
|Data Set Characteristics: |Multivariate, Time-Series |Number of Instances: |10.000 |Area: |Computer |
|Attribute Characteristics: |Real |Number of Attributes: |14 | Date Donated: | 2020-08-30 |
|Associated Tasks: | Classification, Regression, Causal-Discovery | Missing Values? |N\A|Number of Web Hits: | 16614 |


Source:

Stephan Matzka, School of Engineering - Technology and Life, Hochschule für Technik und Wirtschaft Berlin, 12459 Berlin, Germany, stephan.matzka '@' htw-berlin.de

Data Set Information:

Since real predictive maintenance datasets are generally difficult to obtain and in particular difficult to publish, we present and provide a synthetic dataset that reflects real predictive maintenance encountered in industry to the best of our knowledge.




Attribute Information:

The dataset consists of 10 000 data points stored as rows with 14 features in columns <br>
<b> UID : </b> unique identifier ranging from 1 to 10000<br>
<b> product ID: </b>consisting of a letter L, M, or H for low (50% of all products), medium (30%) and high (20%) as product quality variants and a variant-specific serial number <br>
<b> air temperature [K]: </b> generated using a random walk process later normalized to a standard deviation of 2 K around 300 K <br>
<b> process temperature [K]: </b> generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K. <br>
<b> rotational speed [rpm]: </b> calculated from a power of 2860 W, overlaid with a normally distributed noise <br>
<b> torque [Nm]: </b> torque values are normally distributed around 40 Nm with a Ïƒ = 10 Nm and no negative values. <br>
<b> tool wear [min]: </b> The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process. and a
'machine failure' label that indicates, whether the machine has failed in this particular datapoint for any of the following failure modes are true. <br>

The machine failure consists of five independent failure modes <br>
<b> tool wear failure (TWF): </b> the tool will be replaced or fail at a randomly selected tool wear time between 200 â€“ 240 mins (120 times in our dataset). At this point in time, the tool is replaced 69 times, and fails 51 times (randomly assigned). <br>
<b> heat dissipation failure (HDF): </b> heat dissipation causes a process failure, if the difference between air- and process temperature is below 8.6 K and the toolâ€™s rotational speed is below 1380 rpm. This is the case for 115 data points. <br>
<b> power failure (PWF): </b> the product of torque and rotational speed (in rad/s) equals the power required for the process. If this power is below 3500 W or above 9000 W, the process fails, which is the case 95 times in our dataset. <br>
<b> overstrain failure (OSF): </b> if the product of tool wear and torque exceeds 11,000 minNm for the L product variant (12,000 M, 13,000 H), the process fails due to overstrain. This is true for 98 datapoints. <br>
<b> random failures (RNF): </b> each process has a chance of 0,1 % to fail regardless of its process parameters. This is the case for only 5 datapoints, less than could be expected for 10,000 datapoints in our dataset.<br>
<br>
If at least one of the above failure modes is true, the process fails and the 'machine failure' label is set to 1. It is therefore not transparent to the machine learning method, which of the failure modes has caused the process to fail

Relevant Papers:

Stephan Matzka, 'Explainable Artificial Intelligence for Predictive Maintenance Applications', Third International Conference on Artificial Intelligence for Industries (AI4I 2020), 2020 (in press)


Citation Request:

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science. 


# <b> What is done in this notebook? </b> #
Every data record contains information on the failure and the failure mode.
In this notebook we try to predict, which failure mode applies to a given record.

To this end we chose a subset of features of the dataset and use:
1. a decisiontree 
2. a random forrest

## Plan: ##
1. enhance the failue modes by a new one: NOF - no failure. Thus a predictor has the possibility to move probability mass to this mode in case of no failure.
2. construct a random forrest 
3. determine the best size of the forrest measured by AUC


# Accuracy is not the correct metric #
Machine failure is a rare event, i.e. accuracy is no meaningful performace-measure since a classifier allways predicting "No Machine Failure" will have a high accurcy without being usefull at all.<br>
We use:<br>
+ AUC area under (ROC) curve ---
