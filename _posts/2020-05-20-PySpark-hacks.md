---
layout: post
title: "PySpark Basics"
categories: misc
---

## Flow of the blog
1. Basic terminology associated with PySpark
2. Basic commands of PySpark
3. Using SQL in PySpark
4. Using Pandas and Spark operations on same dataframe
5. Data Compression techniques
6. Code optimization techniques - Repartitioning & Caching
7. Complex programs such as PCA and predictive modeling using Spark
8. Picking best spark cluster configuration

## Apache Spark
Apache Spark™ is a unified analytics engine for large-scale data processing. Apache Spark is an open-source distributed general-purpose cluster-computing framework. Spark provides an interface for programming entire clusters with implicit data parallelism and fault tolerance. It provides high-level APIs in Java, Scala, Python and R

As compared to the disk-based, two-stage MapReduce of Hadoop, Spark provides up to 100 times faster performance for a few applications with in-memory primitives. Fast performance makes it suitable for machine learning algorithms as it allows programs to load data into the memory of a cluster and query the data constantly

## PySpark
PySpark is the Python API written in python to support Apache Spark. In a PySPark code, developers have the liberty to use SPark , Python , SQL commands hence makes it a perfect programming lanaguage for machine learning development

If you’re already familiar with Python and libraries such as Pandas, then PySpark is a great language to learn in order to create more scalable analyses and pipelines

#### Before, we get started with the basics of PySPark programming, lets discuss some key terms associated with spark framework.

### Resilient Distributed Datasets (RDD)
RDD’s are collection of data items that are split into partitions and can be stored in-memory on workers nodes of the spark cluster. In terms of datasets, apache spark supports two types of RDD’s – Hadoop Datasets which are created from the files stored on HDFS and parallelized collections which are based on existing Scala collections. Spark RDD’s support two different types of operations – Transformations and Actions.

### Lazy Evaluation
As the name itself indicates its definition, lazy evaluation in Spark means that the execution will not start until an action is triggered. For example, we filter a datatset. The filtering will not actually take place until the user expects the output such as Count after filtering or exporting the filtered dataset to disk.

### Action
Spark Action is the operation where spark will actually compute the underlying flow of sequence.

To better understand the concept of Lazy Evaluation and Action, lets take an example of following code



```python
import time

```


```python
### Load a csv file as spark dataframe
df = spark.read.format("csv").option("header", "true").load("09_inventory.csv") 

```

#### Lazy Evaluation


```python
t1 = time.time()
###### Lazy Evaluation ###################

df_filtered = df.filter(df("state") === "OH")

t2 = time.time()
print(t2-t1)

```

    4.673004150390625e-05


#### Action 


```python
t3 = time.time()

df_filtered.count()

t4 = time.time()
print(t4-t3)

```

    6.0064146518707275





