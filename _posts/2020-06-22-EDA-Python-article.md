---
layout: post
title: "Python Utilities"
categories: misc
---

## Table of contents

1. Exploratory Data Analysis in Python

2. Building Binary classification model using GBM and H2o Library

3. Grid Search in H2o

4. Data connections & Pipelines in Python - Hive2Python and Serial connection

5. Basic Webscapping using BeautifulSoup library

6. Topic modelling using LDA

7. N gram analysis using NLTK





### Exploratory Data Analysis in Python

For this section, we will use the the famous Titanic dataset for performing univariate and bivariate analysis


##### Thanks to Kaggle and encyclopedia-titanica for the dataset


```python
import pandas as pd

df_input = pd.read_csv("titanic.csv",sep=",")
```


```python
df_input.head(4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Siblings/Spouses Aboard</th>
      <th>Parents/Children Aboard</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>Mr. Owen Harris Braund</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>Mrs. John Bradley (Florence Briggs Thayer) Cum...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>Miss. Laina Heikkinen</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>Mrs. Jacques Heath (Lily May Peel) Futrelle</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
    </tr>
  </tbody>
</table>
</div>



In this dataset, the Column "Survived" is the target binary variable which tells if the particular id survived or not


```python
##One can get alot of insights for numeric fields from the describe function of pandas

df_input.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>Siblings/Spouses Aboard</th>
      <th>Parents/Children Aboard</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>887.000000</td>
      <td>887.000000</td>
      <td>887.000000</td>
      <td>887.000000</td>
      <td>887.000000</td>
      <td>887.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.385569</td>
      <td>2.305524</td>
      <td>29.471443</td>
      <td>0.525366</td>
      <td>0.383315</td>
      <td>32.30542</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.487004</td>
      <td>0.836662</td>
      <td>14.121908</td>
      <td>1.104669</td>
      <td>0.807466</td>
      <td>49.78204</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.92500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.45420</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.13750</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.32920</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Lets check for unique values for each column

print(df_input.nunique())
```

    Survived                     2
    Pclass                       3
    Name                       887
    Sex                          2
    Age                         89
    Siblings/Spouses Aboard      7
    Parents/Children Aboard      7
    Fare                       248
    dtype: int64



```python
print(df_input.dtypes)
```

    Survived                     int64
    Pclass                       int64
    Name                        object
    Sex                         object
    Age                        float64
    Siblings/Spouses Aboard      int64
    Parents/Children Aboard      int64
    Fare                       float64
    dtype: object



```python
### You must have noticed that the unique value count for column Pclass is only 3 but its a numeric field.
## Lets convert it into an object dtype so as to treat it as categorical

df_input["Pclass"] = df_input["Pclass"].astype("O") 
```


```python
print(df_input.dtypes)
```

    Survived                     int64
    Pclass                      object
    Name                        object
    Sex                         object
    Age                        float64
    Siblings/Spouses Aboard      int64
    Parents/Children Aboard      int64
    Fare                       float64
    dtype: object



```python
def univariate(data):
    #Creating seperate dataframes for categorical and continuous variables
    data_cat = data.select_dtypes(include=[object])    
    data_cont = data.select_dtypes(exclude=[object])


    data_cont_univ = data_cont.describe(percentiles=[.001,.005,.01,.25,.5,.75,.95,.99,.995,.999]).transpose()

    data_cont_univ["blanks"] = data_cont.isna().sum(axis=0) + data_cont.isnull().sum(axis=0)

    data_cont_univ["zeros"] = (data_cont == 0).sum(axis=0)

    data_cat_univ = data_cat.describe().transpose()

    data_cat_univ["blanks"] = data_cat.isna().sum(axis=0) + data_cat.isnull().sum(axis=0)

    data_cat_univ["zeros"] = data_cat.isin(["0","00","000","0000","00000","000000"]).sum(axis=0)

    return data_cat_univ,data_cont_univ
```


```python
data_cat_univ,data_cont_univ = univariate(df_input)
```


```python
data_cat_univ.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
      <th>blanks</th>
      <th>zeros</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Pclass</th>
      <td>887</td>
      <td>3</td>
      <td>3</td>
      <td>487</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Name</th>
      <td>887</td>
      <td>887</td>
      <td>Mr. Sarkis Lahoud</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>887</td>
      <td>2</td>
      <td>male</td>
      <td>573</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_cont_univ.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>0.1%</th>
      <th>0.5%</th>
      <th>1%</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>95%</th>
      <th>99%</th>
      <th>99.5%</th>
      <th>99.9%</th>
      <th>max</th>
      <th>blanks</th>
      <th>zeros</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Survived</th>
      <td>887.0</td>
      <td>0.385569</td>
      <td>0.487004</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0</td>
      <td>545</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>887.0</td>
      <td>29.471443</td>
      <td>14.121908</td>
      <td>0.42</td>
      <td>0.6415</td>
      <td>0.83</td>
      <td>1.0</td>
      <td>20.250</td>
      <td>28.0000</td>
      <td>38.0000</td>
      <td>55.85000</td>
      <td>66.000000</td>
      <td>70.285</td>
      <td>74.6840</td>
      <td>80.0000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Siblings/Spouses Aboard</th>
      <td>887.0</td>
      <td>0.525366</td>
      <td>1.104669</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>3.00000</td>
      <td>5.000000</td>
      <td>8.000</td>
      <td>8.0000</td>
      <td>8.0000</td>
      <td>0</td>
      <td>604</td>
    </tr>
    <tr>
      <th>Parents/Children Aboard</th>
      <td>887.0</td>
      <td>0.383315</td>
      <td>0.807466</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>2.00000</td>
      <td>4.000000</td>
      <td>5.000</td>
      <td>5.1140</td>
      <td>6.0000</td>
      <td>0</td>
      <td>674</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>887.0</td>
      <td>32.305420</td>
      <td>49.782040</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>7.925</td>
      <td>14.4542</td>
      <td>31.1375</td>
      <td>112.55749</td>
      <td>249.600388</td>
      <td>263.000</td>
      <td>512.3292</td>
      <td>512.3292</td>
      <td>0</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Lets do the bivariate now 

```


```python
def bivariate(data,ignore_col_list,event):
    
    #Subsetting categorical variables for bivariate analysis
    col_list_tmp = list(df_input.select_dtypes(include=[object]).columns)+[event]
    col_list = [w for w in col_list_tmp if w not in ignore_col_list]
    data_cat = data[col_list]
    
    cols = list(data_cat.columns.values)
    
    #Looping bivariate analysis for all variables and appaending the results in a single dataframe
    #The len()-2 is to exclude the variables for which bivariate is not needed (as per the column sequence)
    appended_data = pd.DataFrame()
    for x in range(0, len(data_cat.columns)-1):
    
        data2 = pd.DataFrame({'1.Variable':data_cat.columns[x],
                              '2.Level':data_cat.groupby(data_cat.columns[x])[event].sum().index,
                              '3.Event':data_cat.groupby(data_cat.columns[x])[event].sum(),
                              '4.Volume':data_cat.groupby(data_cat.columns[x])[event].count(),
                              '5.Rate':((data_cat.groupby(data_cat.columns[x])[event].sum()/data_cat.groupby(data_cat.columns[x])[event].count())*100).round(2)})
    

        appended_data = appended_data.append(data2)
    return appended_data
```


```python
df_bivariate = bivariate(df_input,['Name'],"Survived")
```


```python
df_bivariate.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1.Variable</th>
      <th>2.Level</th>
      <th>3.Event</th>
      <th>4.Volume</th>
      <th>5.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Pclass</td>
      <td>1</td>
      <td>136</td>
      <td>216</td>
      <td>62.96</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pclass</td>
      <td>2</td>
      <td>87</td>
      <td>184</td>
      <td>47.28</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pclass</td>
      <td>3</td>
      <td>119</td>
      <td>487</td>
      <td>24.44</td>
    </tr>
    <tr>
      <th>female</th>
      <td>Sex</td>
      <td>female</td>
      <td>233</td>
      <td>314</td>
      <td>74.20</td>
    </tr>
    <tr>
      <th>male</th>
      <td>Sex</td>
      <td>male</td>
      <td>109</td>
      <td>573</td>
      <td>19.02</td>
    </tr>
  </tbody>
</table>
</div>



#### We could now build beautiful plots for these results, infact users can go ahead and change the body of our bivariate function to create python plots


```python
# import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# if using a Jupyter notebook, includue:
%matplotlib inline
```


```python
### Plotting a Grid of plots for bivariate between different columns in the data.

#Caution : Might be very slow depending upon datasize
```
sns.pairplot(df_input, hue="Survived")

<p align="center">
  <img src="https://github.optum.com/ssing339/PCR_ENI_1_codes/blob/master/output_20_2.png" />
</p>

<p align="center">
  <img src="https://github.optum.com/raw/ssing339/PCR_ENI_1_codes/master/output_20_2.png" />
</p>


[Seaborn documentation][https://seaborn.pydata.org/tutorial/distributions.html]

#### Using the above plots, we can see how closely two columns are interating with one another

At the same time, the hue based on our dependent variable helps us idenitfy potential features in predicting dependent variable

### Building a Binary classification model using GBM in H2o Library


```python
import h2o
import os
import pandas as pd
import numpy as np
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch
```


```python
try:
    h2o.shutdown()
except:
    try:
        h2o.init()
    except:
        h2o.connect()

```


```python
train_df = h2o.import_file('titanic.csv')
```

    Parse progress: |█████████████████████████████████████████████████████████| 100%



```python
train_df.head()
```


<table>
<thead>
<tr><th style="text-align: right;">  Survived</th><th style="text-align: right;">  Pclass</th><th>Name                                              </th><th>Sex   </th><th style="text-align: right;">  Age</th><th style="text-align: right;">  Siblings/Spouses Aboard</th><th style="text-align: right;">  Parents/Children Aboard</th><th style="text-align: right;">   Fare</th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">         0</td><td style="text-align: right;">       3</td><td>Mr. Owen Harris Braund                            </td><td>male  </td><td style="text-align: right;">   22</td><td style="text-align: right;">                        1</td><td style="text-align: right;">                        0</td><td style="text-align: right;"> 7.25  </td></tr>
<tr><td style="text-align: right;">         1</td><td style="text-align: right;">       1</td><td>Mrs. John Bradley (Florence Briggs Thayer) Cumings</td><td>female</td><td style="text-align: right;">   38</td><td style="text-align: right;">                        1</td><td style="text-align: right;">                        0</td><td style="text-align: right;">71.2833</td></tr>
<tr><td style="text-align: right;">         1</td><td style="text-align: right;">       3</td><td>Miss. Laina Heikkinen                             </td><td>female</td><td style="text-align: right;">   26</td><td style="text-align: right;">                        0</td><td style="text-align: right;">                        0</td><td style="text-align: right;"> 7.925 </td></tr>
<tr><td style="text-align: right;">         1</td><td style="text-align: right;">       1</td><td>Mrs. Jacques Heath (Lily May Peel) Futrelle       </td><td>female</td></tbody>
</table>


```python

```


```python
## Pass the target variable to be predicted
y = 'Survived'

### Pass the list of predictors 
x = ['Pclass','Age','Siblings/Spouses Aboard',	'Parents/Children Aboard',	'Fare']

# If target variable is binary, convert it to factor
train_df[y]= train_df[y].asfactor()
```


```python
## Split the dataset into train and valid frames
train, valid = train_df.split_frame(ratios=[.8], seed=1234)
```


```python
### Lets define our grid parameters. It is same as running a for loop to generate models from the same ML algorithm but with different hyper parameters

gbm_params2 = {'learn_rate': [i * 0.01 for i in range(4, 5)],
                'max_depth': list(range(11, 12)),
                'sample_rate': [i * 0.1 for i in range(7, 9)],
                'col_sample_rate': [i * 0.1 for i in range(8, 9)],
               'col_sample_rate_per_tree': [i * 0.1 for i in range(7, 8)],
              'learn_rate_annealing':[0.99]
              }

# Search criteria
search_criteria = {'strategy': 'RandomDiscrete', 'max_models': 20, 'seed': 1}

# Train and validate a random grid of GBMs
gbm_grid2 = H2OGridSearch(model=H2OGradientBoostingEstimator,
                          grid_id='gbm_grid',
                          hyper_params=gbm_params2,
                          search_criteria=search_criteria)
    gbm_grid2.train(x=x, y=y,
                training_frame=train,
                validation_frame=valid,
                ntrees=120,
                seed=1,ignore_const_cols = 1)

```

#### Exporting the results to a csv file


```python
ModelSummary = pd.DataFrame(columns=['Model_ID','learn_rate', 'Trees', 'Depth', 'Row_Sampling', 'Col_Sampling', 'MTries','NBins', 
                                    'Categorical_Encoding', 'Hist_Type', 'Training_Log_Loss', 'Validation_Log_Loss', 
                                    'Training_AUC', 'Validation_AUC'])

ModelSummary2 = pd.DataFrame(columns=['Model_ID', 'learn_rate','Trees', 'Depth', 'Row_Sampling', 'Col_Sampling', 'MTries','NBins', 
                                    'Categorical_Encoding', 'Hist_Type', 'Training_Log_Loss', 'Validation_Log_Loss', 
                                    'Training_AUC', 'Validation_AUC'])
for Model in gbm_grid2:
    ModelSummary['Model_ID'] = [Model.model_id]
    ModelSummary['learn_rate'] = [Model.actual_params.get('learn_rate')]
    ModelSummary['Trees'] = [Model.actual_params.get('ntrees')]
    ModelSummary['Depth'] = [Model.actual_params.get('max_depth')]
    ModelSummary['Row_Sampling'] = [Model.actual_params.get('sample_rate')]
    ModelSummary['Col_Sampling'] = [Model.actual_params.get('col_sample_rate_per_tree')]
    ModelSummary['MTries'] = [Model.actual_params.get('mtries')]
    ModelSummary['NBins'] = [Model.actual_params.get('nbins')]
    ModelSummary['Categorical_Encoding'] = [Model.actual_params.get('categorical_encoding')]
    ModelSummary['Hist_Type'] = [Model.actual_params.get('histogram_type')]
    ModelSummary['Training_Log_Loss'] = [Model.logloss()]
    ModelSummary['Validation_Log_Loss'] = [Model.logloss(valid=True)]
    ModelSummary['Training_AUC'] = Model.auc()
    ModelSummary['Validation_AUC'] = Model.auc(valid=True)
    ModelSummary2 = ModelSummary2.append(ModelSummary, sort = False)

ModelSummary2.to_csv("gbm_grid_results.csv")
```


```python

```

### Building data connections between different servers

In building machine learning applications, sometimes the data resides in more than 1 server or disk. Let's say we have a client server from where we indend to fetch data into our server for data processing and push back results to the client server.

In the above scenario, following steps could be helpful
1. Sqoop Import Data from server and store its metadata in your server.
2. Perform data processing and analysis
3. Sqoop Export output data back to the client server

Let's how we can do this in Python 





### Hive and Python


```python
################# Hive connection in Python using pyhive package 

## In Python, we have this library called pyhive, this can be used to access a Hive table residing in a HDFS location
import pandas as pd
from pyhive import hive
import time
```


```python
statement = '''SELECT * from dbo.tablename limit 10'''


print("Setting up connection")

### In the connect function, we will establish the connection between the hive database and our Python Kernel
conn = hive.connect(host='<server name>',\
                    port=<port number>,\
                    auth='LDAP',\
                    username='<id>',\
                    password="<password?>")

print("Connection successful")
```


```python
start1=time.time()
start=time.ctime()
print( "started at:"),start

cur = conn.cursor()
#### The below command will fetch the results for the query named "statement"
cur.execute(statement)
df = cur.fetchall()
### The output of the fetchall returns a List and doesnt contain the column names. So we will have to pick up the schema 

### The below command will pull the column names as a list
cur.execute("show columns in dbo.tablename")
df1 = cur.fetchall()


col_names=[w[0] for w in df1]
print( "Data fetch completed at:"),time.ctime()
print(  "Time taken: "),time.time()-start1


print("output saved as a pandas df")

df_final=pd.DataFrame(df,columns=col_names)

```

### SSH Connection in Python

Using Paramiko to establish SSH connection in Python


```python
################ Serial connection in Python 

Sometimes ,we intend to establish a serial data connection capable of transmitting datastream like this 100000000000100000000010000

### Loading the package paramiko for establishing serial connection SSH

import paramiko

print("Initiate a serial SSH client handle")

ssh_client =paramiko.SSHClient()

### Below command is an optional command required
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

### Below command actually creates the connection with the UNIX

ssh_client.connect(hostname="<server location>",username="<id>",password="<passwordhere>")

handle = ssh_client.open_sftp()

######### Reading file from HDFS location into python session

file_path =handle.file("<remote server location to be read>")

data_read= pd.read_csv(file_path)

### Write the file to local location

data_read.to_csv("testinfile.csv")

#### Writing back file from domino location to HDFS location 

handle.put("testinfile.csv","<remote server location where we want to write back")

```

Note that SSH connection is very very slow since its a serial connection and transmits data bit by bit. 

So, please use it for data transmission for only small datatsets.

For big datasets, use jdbc or odbc connection in Python or R

### Web-Scrapping in Python

Here we will cover different libraries in python which can be used in scrappinng data from public websites

Put on your hacking cap and lets break the myths and jazziness around web scrapping

Important notes before scrapping data from any website

1. Please check in advance if its legal to scrap data from that website by inspecting a few compliance related aspects using robot.txt
2. Inspect the structure of the website and analyse how the flow and schema of that website looks like
3. always improvise while fetching desired data.


```python
### requests library is used for fetching the dta from a html page
import requests
import urllib.request
import time
## BeautifulSoup is a very cool library, you can also check out scrapy library ( highly autmoated and parameterized )
## to study more about this lib, refer https://www.crummy.com/software/BeautifulSoup/bs4/doc/
from bs4 import BeautifulSoup

## Target URL
url = 'http://web.mta.info/developers/turnstile.html'

response = requests.get(url)

## if the previuos command was successful, then 

soup = BeautifulSoup(response.text, “html.parser”)

## in scrapping, the whole intention is to pull labels satisfying either a string or a regex
## here i am planning to pull all labels containing "coders"

all_tags = soup.findAll('coders')

##This code gives us every line of code that has an <coders> tag in form of a "list"

# now lets try to actually reach out to some underlying url with tag "a"

single_tag = all_tags[12]
link = single_tag["href"]

download_url = 'http://web.mta.info/developers/'+ link
urllib.request.urlretrieve(download_url,'./’'link[link.find('/turnstile_')+1:]) 


```

note that a request command could take from few ms to to a few seconds, so when you try to run a loop, do keep sleep commands in the code

##  Topic modelling & N-Gram in Python 


```python
import pandas as pd
import re
import numpy as np
import matplotlib
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from gensim import corpora, models
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem import PorterStemmer
from nltk import word_tokenize

```

### LDA For Topic Modelling

LDA is an iterative algorithm. Here are the two main steps:
    
1. In the Initialization stage, each word is assigned to a random topic.


2. Iteratively, the algorithm goes through each word and reassigns the word to a topic taking into consideration

    a. What's the probability of the word belonging to a topic

    b. What's the probaility of a document to be generated by a topic


```python
########## Data cleaning for Topic modelling 
def data_cleaning(tweet,custom_list):
    tweet = re.sub(r'\$\w*','',tweet) # Remove tickers
    tweet = re.sub(r'https?:\/\/.*\/\w*','',tweet) # Remove hyperlinks
    tweet = re.sub(r'['+string.punctuation+']+', ' ',tweet) # Remove puncutations like 's
    #stop = set(stopwords.words('english'))
    tweet = re.sub(r'[^a-zA-Z0-9]'," ",tweet)
    stop_words=set(['a',    'about', 'above', 'after',   'again',  'against',              'ain',      'all',        'am',               'an',       'and',     'any',     'are',      'as',        'at',        'be',       'because',            'been',   'before',               'being',  'below', 'between',           'both',   'but',      'by',        'can',     'couldn',               'd',               'did',      'didn',    'do',       'does',   'doesn', 'doing',  'don',     'down',  'during',               'each',               'few',     'for',      'from',   'further',              'had',     'hadn',   'has',      'hasn',   'have',   'haven',               'having',               'he',       'her',      'here',    'hers',    'herself',              'him',     'himself',               'his',       'how',    'i',           'if',         'in',         'into',     'is',         'isn',       'it',         'its',        'itself',               'just',     'll',          'm',        'ma',      'me',      'mightn',              'more',  'most',   'mustn', 'my',               'myself',               'needn', 'now',    'o',         'of',        'off',      'on',       'once',   'only',    'or',               'other',  'our',      'ours',    'ourselves',          'out',      'over',    'own',    're',        's',          'same',               'shan',   'she',      'should',               'so',        'some',  'such',    't',          'than',    'that',    'the',               'their',   'theirs',  'them',  'themselves',      'then',    'there',  'these',  'they',    'this',     'those',               'through',            'to',        'too',     'under', 'until',    'up',       've',        'very',    'was',     'we',               'were',   'weren',               'what',   'when',  'where',               'which', 'while',  'who',    'whom',               'why',    'will',      'with',    'won',    'y',          'you',     'your',    'yours',  'yourself',               'yourselves'])
    exclude = set(string.punctuation)
    exclude1= set(custom_list)
    stop_words.update(exclude1)
    lemma = WordNetLemmatizer()
    stop_free = " ".join([i for i in tweet.lower().split() if i not in stop_words])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
```


```python
########## Pass custom list of stop words in the following list "custom_list"
	
custom_list=["fall"]
```


```python
####### Load input data for text analysis 
text1 = pd.read_csv(r'path.csv', sep=',', encoding='iso-8859-1')

text1.dropna(axis=0,how='any',inplace=True)

#### Here "Text" column is our target column 
text=text1["Text"].values.tolist()

```


```python
####### mention number of topics 
NUM_TOPICS = 10
STOPWORDS = stopwords.words('english')

### text column was converted into a list for easy iterations 
tweet_text=text  # text is a list of column to be modelled
doc_complete=tweet_text
#doc_complete_2=[]
doc_complete_2=tweet_text


a = 0 
 
### For gensim we need to tokenize the data and filter out stopwords

tokenized_data = []
for w in doc_complete_2:
    a=a+1
    tokenized_data.append(data_cleaning(w,custom_list))

# Build a Dictionary - association word to numeric id
dictionary = corpora.Dictionary(tokenized_data)

# Transform the collection of texts to a numerical form

corpus = [dictionary.doc2bow(text) for text in tokenized_data] 
# Have a look at how the Nth document looks like: [(word_id, count), ...]
# [(12, 3), (14, 1), (21, 1), (25, 5), (30, 2), (31, 5), (33, 1), (42, 1), (43, 2),  ...
 # Build the LDA model

### Lets create a LDA model object specifiying corpus , number of topics 5,6 ... and the words dictionary 

lda_model = models.LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)
final=[]
for topic in lda_model.show_topics(num_topics=NUM_TOPICS, formatted=False, num_words=6):
    topicwords = [w for (w, val) in topic[1]]
    topicwords_val = [val for (w, val) in topic[1]]
    final.append([topicwords,topicwords_val])
final1=pd.DataFrame(final,columns=["topic","prob"])



final1.to_csv(path+"/topics.csv")
```

* The number of topics (n_topics) as a parameter. None of the algorithms can infer the number of topics in the document collection


* All of the algorithms have as input the Document-Word Matrix (or Document-Term Matrix). DWM[i][j] = The number of occurrences of word_j in document_i


* All of them output 2 matrices: WTM (Word Topic Matrix) and TDM (Topic Document Matrix). The matrices are significantly smaller and the result of their multiplication should be as close as possible to the original DWM matrix.


### N Gram Analysis 


```python
########################################N-gram analysis

import re
import string
from nltk.corpus import stopwords
import string
from collections import Counter
from nltk import ngrams

############## Creating UDF for ngram analysis , n means number of grams 1,2,3 ....


############ Data cleaning for Ngram analysis 
def clean_text(text,custom_list):
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    cleaned_text1 = [t for t in cleaned_text if t not in custom_list]
    return cleaned_text

def cal_ngram(text,n):
    token = nltk.word_tokenize(text)
    n_grams = ngrams(token,n)
    return n_grams

############# 
n = 3 #### Here n means number of grams

n_gram = cal_ngram(text[0],n)   # generating N grams for input data

### Counter library 
n_gram_common = Counter(n_gram).most_common()  # Listing top 10 trending N grams

n_gram_df=pd.DataFrame(n_gram_common,columns=["N Gram","Frquency"])

```


```python
#### Now this n gram results can be used to publish a word cloud
```


```python

```


```python

```

