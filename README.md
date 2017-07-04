# titanic-spark
Kaggle titanic example for Apache Spark

Train and test data are from Kaggle site: https://www.kaggle.com/c/titanic

Need Apache Spark setup, won't run as python code. Update Spark master address in SparkSession builder before running

```
SparkSession.builder.appName("Titanic").master("spark://localhost:7077").getOrCreate()
```
