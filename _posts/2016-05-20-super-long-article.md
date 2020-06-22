---
layout: post
title: "Feature Engineering Guide"
categories: misc
---

{% include lib/mathjax.html %}
## Brief Overview
The aim of this notebook is to walk users through the commonly used features in building regression and binary classification models.

For different types of problem statements and modelling techniques, the choice of variables and feature selection varies.

For Regression based models, we might be interested in building aggregated views of certain continuous independent variables using history data.

For classification based models, we might be interested in introducing event based features to help the model focus on variables which might be helpful in explaining maximum variance and patterns in the data

<img src="https://www.vidora.com/wp-content/uploads/2017/10/Screen-Shot-2017-10-16-at-6.33.28-PM.png" width="700" height="300" title="Feature Engineering" align="center"/>




<p align="center">
  <img width="600" height="200" src="https://www.vidora.com/wp-content/uploads/2017/10/Screen-Shot-2017-10-16-at-6.33.28-PM.png">
</p>






## Types of Features

### Risk
Risk in general terms means the probablity that an event will happen.An Event could be a claim denial, Getting or not getting a call in call center analytics, An overpayment on a claim and so on.In all our binary classification problems, we tend to build risk features one way or the other. Generally Data Scientists  build risk feature as simple as ratio of all 1s to sum of 1s and 0s. 

#### But isn't it wrong???
Lets say we have 2 doctors
* Doctor A fills 100 claims out of which 10 were having overpayment. Simple Event rate risk would be 0.1.
* Doctor B Fills 2 claims out of which 1 had overpayment. Risk for Doctor B would be 0.5.

#### Don't you think, we should be focusing more on Doctor A than Doctor B?????

There should be some smoothening factor associated while calculating risk which should consider entire population while computing risk

Risk Formula : ((x.Errors_for_category+0.5)/(x.Claims_for_category+(0.5/(Error_population/Frequency_population))))

$$ {DocA}_{risk}  :(10+0.5)/(100+(0.5/(11/102) $$ = 0.1003

$$ {DocB}_{risk}  :(1+0.5)/(2+(0.5/(11/102) $$ = 0.226

If you noticed, risk value for A was not affected much but risk for B was reduced significantly and at same time was still higher than A. This way, we not only smoothen the risk value but also carry the essence of the actual risk

Further, think about seasonality and how it could generate outliers in the risk calculation. To mitigate impact of seasonality on our risk calcualtion, its always recommended to calculate these matrices using a history of data. The amount of history could vary from project to project depending upon patterns in the data. Like in 1 project, operational changes occur quarterly, then a 3 months lookback would be able to pick all sorts of seasonal patterns.


#### When to Choose Risk over the value itself?
Well, generating such risk features could increase the model lifecycle and save you from regular model maintenance or refresh activity
Lets say you have categorical variable "Member State". Considering low cadinality of this field say 50, one could pass this variable as such to the model. But 1 year from model deployment, new states could begin to appear in the data and the model would start imputing them. What if the new states had an higher risk associated with them but model would treat all new fields as same i.e. Unknown value. If one uses Risk of Member State as a feature, then no matter how many new values appear in the data, we would be exposing a numeric value to the model which would be risk associated with a new state

Periodic refresh of these Risk features helps the "Deployed" production model to learn new variations in the input dataset and give more accurate and precise results

#### Missing Value Imputation
It is generally recommended to impute missing values with the overall risk value for the month 

$$ {Risk}_{gen} : {Error}_{count}/{Observations} $$




### Entropy
Entropy is a measure of uncertainty or randomness associated with a feature. A more stable system would have Zero Entropy whereas a highy volatile system would have a very high entropy value.

$$ Entropy = \biggl(\sum_{i=1}^{n}x_{i}/w*log(w/x_{i})\biggr) $$

where n is maximum number of child column values across which occured for a parent column.

w defines frequency at which parent column occured.

$${x}_{i}$$ defines the frequency at which parent column and child column interaction occured

In machine learning, the rationale behind using entropy feature is that it helps to identify abnormalities or surprise element associated with certain obervations or records in the data under observation.

Lets take a small example to build intuition for when to use Entropy Features.

Lets say, we have a data for all the general practitioners operating in United States. In an ideal case scenario, 1 doctor would be operating in only 1 state or max 2. This would be general trend for most of the providers.

A higher Entropy of Doctor Name over State means that a certain doctor is operating in a large # of states of US which is not really possible for a general practitioner right!!!. 

#### Missing Value Imputation
It is generally recommended to impute missing values with the overall median value for the month. Generally, it is 0 value or a value very close to 0. So, its also safe to impute it with 0 value




### Lookback
In most of our projects, we tend to compute certain aggregations like mean, max,min,count of certain continuous features/columns. Lookback here means that we would be using a certain history data in a dynamic fashion to calculate these aggregations



<img src="https://raw.githubusercontent.com/Satwant201/Feature_Engineering_kit/master/Capture.PNG" width="250" height="250" alt="# Months lookback Timeline" title="Lookback Intuition" align="center"/>



In the above trend chart, for calculating count of claims for december month, I would use Sept - Nov Data and so on.
Here, Jan- Mar data($ { Red Shaded } $) is used to calcuate lookback features for april month. Since, there are not enough observations for calculating lookback for these 3 months, we would remove this data while model development




### Principal component analysis (PCA) 

It is a technique used to emphasize variation and bring out strong patterns in a dataset. It's often used to make data easy to explore and visualize


<img src="https://raw.githubusercontent.com/Satwant201/Feature_Engineering_kit/master/Capture2.PNG" width="500" height="350" title="Dimension reduction using PCA" align="center"/>


In the above image, we tranform a 3D data into 2D data using PCA

Advantages of using PCA in Model Development:

1. Number of features via which model needs to be trainied can be drastically reduced
2. Correlation between the raw features can be eliminiated
3. Faster model development and fine tuning

Disadvantages of using PCA:

Since PCA generates the output features in a new N Dimensional space, the values cannot be interpreted.
So, if the intension of your project it to not only provide analytical results but at the same time provide reason codes and other business justifications, PCA might not be really helplful

#### Data Requirements for computing PCA :
You don’t always need to remove outliers and skewness from your data. but algorithms like PCA expect your data to have somewhat normal distributions. Scaling also has a big effect on any mode which calculates distances between observations. I have noticed pretty distinct jumps in model performance before and after removing skewness from my data in some projects.

For the experimentation purposes, i prepared an Analytical datatset with 35 conventional continuous features and build a baseline GBM model for binary classification.

Then, i normalized my dataset and built the same model with same hyper-parameters using only first 5 principle components generated using PCA.

Below, you can find the comparison of the AUC plot for both the models. Even after getting rid of 30 Features, we suffered a performance loss of only 1.5%



<img src="https://github.optum.com/raw/ssing339/PCR_ENI_1_codes/master/Capture3.PNG" width="550" height="350" alt="# Months lookback Timeline" title="Lookback Intuition" align="centre"/>





##### Still not convinced with what wonders PCA can do!!!!

Please visit https://projector.tensorflow.org/ and see how PCA helps in visualizing a 50+ dimension data(Word Embeddings) in 3d









#### Future Scope

There are alot of additional features which Data Scientists build specific to their project requirement and business requirement. We look forward to feedback and information about additional features which might be helpful to the larger team.

In the next tutorial, I  plan to cover the implementation of different types of features in PySPark and discuss about the different normalization techniques and their inplementation in PySpark

#### End of the Document



We would appreciate your feedback, suggestion on how we can improve this.

#### Email id -satwantsingh201@gmail.com

