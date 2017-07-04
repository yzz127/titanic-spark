from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, ShortType, StringType
from pyspark.sql.functions import UserDefinedFunction, col, when
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, GBTClassifier, RandomForestClassifier, NaiveBayes
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from random import randint

if __name__ == "__main__":
    
    # initialize SparkSession
    spark = SparkSession.builder.appName("Titanic").master("spark://localhost:7077").getOrCreate()

    # load training data and test data
    titanic_df = spark.read.csv("train.csv", header = True, mode = "DROPMALFORMED")
    test_df = spark.read.csv("test.csv", header = True, mode = "DROPMALFORMED", )

    # check training and test data schema
    titanic_df.show()
    titanic_df.count()
    titanic_df.printSchema()

    test_df.show()
    test_df.count()
    test_df.printSchema()

    # fill null of "Embarked" with "S" for both training and test data
    titanic_df.fillna({"Embarked": "S"})
    test_df.fillna({"Embarked": "S"})

    # create new feature of "C" based on "Embarked" data
    def dummy_C(x):
        return when(col(x) == "C", 1).otherwise(0)
    titanic_df = titanic_df.withColumn("C", dummy_C("Embarked"))
    test_df = test_df.withColumn("C", dummy_C("Embarked"))

    # create new feature of "Q" based on "Embarked" data
    def dummy_Q(x):
        return when(col(x) == "Q", 1).otherwise(0)
    titanic_df = titanic_df.withColumn("Q", dummy_Q("Embarked"))
    test_df = test_df.withColumn("Q", dummy_Q("Embarked"))

    # drop out features based on domain knowledges
    titanic_df = titanic_df.select([c for c in titanic_df.columns if c not in {'PassengerId', 'Name', 'Ticket','Embarked', 'Cabin'}])
    test_df = test_df.select([c for c in test_df.columns if c not in {'PassengerId', 'Name', 'Ticket', 'Embarked', 'Cabin'}])

    # check drop out results for training and test data
    titanic_df.show()
    test_df.show()

    # update data type for "Fare" from string to double
    titanic_df = titanic_df.withColumn("Fare", titanic_df["Fare"].cast("double"))
    test_df = test_df.withColumn("Fare", test_df["Fare"].cast("double"))

    # get median value of "Fare"
    titanic_df_fare_median = titanic_df.approxQuantile("Fare", [0.5], 0.25)
    #test_df_fare_median = test_df.approxQuantile("Fare", [0.5], 0.25)

    # fill null of "Fare" with median value
    titanic_df.select("Fare").na.fill(titanic_df_fare_median[0]).show()
    #test_df.select("Fare").na.fill(test_df_fare_median[0]).show()

    # update data type for "Fare" and "Survived" to int
    titanic_df = titanic_df.withColumn("Fare", titanic_df["Fare"].cast("int"))
    titanic_df = titanic_df.withColumn("Survived", titanic_df["Survived"].cast("int"))

    fare_not_survived = titanic_df.select("Fare", "Survived").filter(titanic_df["Survived"] == 0)
    fare_not_survived.show()

    fare_survived = titanic_df.select("Fare", "Survived").filter(titanic_df["Survived"] == 1)
    fare_survived.show()

    # get mean and standard deviation of "Fare" to check its correlation with "Survived" 
    fare_not_survived.select([mean("Fare"), stddev("Fare")]).show()
    fare_survived.select([mean("Fare"), stddev("Fare")]).show()

    # update data type for "Age" to int
    titanic_df = titanic_df.withColumn("Age", titanic_df["Age"].cast("int"))
    test_df = test_df.withColumn("Age", test_df["Age"].cast("int"))

    # get the age range and generate random number to fill null in "Age"
    titanic_df.select([mean("Age"), stddev("Age")]).show()

    titanic_df = titanic_df.fillna({"Age" : randint(15, 33)})
    test_df = test_df.fillna({"Age" : randint(15, 33)})
    titanic_df.show()

    # generate new feature "Family" based on "Parch" and "SibSp"
    titanic_df = titanic_df.withColumn("Family", titanic_df["Parch"] + titanic_df["SibSp"])
    test_df = test_df.withColumn("Family", test_df["Parch"] + test_df["SibSp"])

    # update data type for "Family" to int
    titanic_df = titanic_df.withColumn("Family", titanic_df["Family"].cast("int"))
    test_df = test_df.withColumn("Family", test_df["Family"].cast("int"))

    # drop "Parch" and "SibSp"
    titanic_df = titanic_df.drop("Parch").drop("SibSp")
    test_df = test_df.drop("Parch").drop("SibSp")

    # define udf to update values for "Family"
    name = 'Family'
    udf = UserDefinedFunction(lambda x: 1 if x > 0 else 0, ShortType())

    titanic_new_df = titanic_df.select(*[udf("Family").alias(name) if column == name else column for column in titanic_df.columns])
    test_df = test_df.select(*[udf("Family").alias(name) if column == name else column for column in test_df.columns])
    
    # define get_person function to update "Sex"
    def get_person(Age, Sex):
        return when(col(Age) < 16, 'child').otherwise(col(Sex))

    titanic_age_sex_df = titanic_new_df.withColumn("Person", get_person("Age", "Sex"))
    test_df = test_df.withColumn("Person", get_person("Age", "Sex"))
    titanic_age_sex_df.show()

    # create two new features "female" and "child", drop "Sex"
    def dummy_female(x):
        return when(col(x) == "female", 1).otherwise(0)

    def dummy_child(x):
        return when(col(x) == "child", 1).otherwise(0)

    titanic_female_dummy_df = titanic_age_sex_df.withColumn("Female", dummy_female("Person"))
    test_df = test_df.withColumn("Female", dummy_female("Person"))
    titanic_female_dummy_df.show()

    titanic_child_dummy_df = titanic_female_dummy_df.withColumn("Child", dummy_child("Person"))
    test_df = test_df.withColumn("Child", dummy_child("Person"))
    titanic_child_dummy_df.show()

    titanic_train_df = titanic_child_dummy_df.drop("Sex").drop("Person")
    test_df = test_df.drop("Sex").drop("Person"))

    # create two new features "Class1" and "Class2", drop "Pclass"
    def dummy_class1(x):
        return when(col(x) == "1", 1).otherwise(0)

    titanic_train_df = titanic_train_df.withColumn("Class1", dummy_class1("Pclass"))
    test_df = test_df.withColumn("Class1", dummy_class1("Pclass"))

    def dummy_class2(x):
        return when(col(x) == "2", 1).otherwise(0)

    titanic_train_df = titanic_train_df.withColumn("Class2", dummy_class2("Pclass"))
    test_df = test_df.withColumn("Class2", dummy_class2("Pclass"))

    titanic_train_df = titanic_train_df.drop("Pclass")
    test_df = test_df.drop("Pclass")

    # finalize training data and test data
    label_train_df = titanic_train_df.select("Survived")
    titanic_train_df = titanic_train_df.withColumnRenamed("Survived", "label")
    
    titanic_train_df.show()
    test_df.show()

    # create "features" with vector assembler
    assembler = VectorAssembler(
        inputCols = ["Age", "Fare", "C", "Q", "Family", "Female", "Child", "Class1", "Class2"],
        outputCol = "features"
    )
    output_train = assembler.transform(titanic_train_df)
    output_test = assembler.transform(test_df)
    
    # check features for both training and test data
    print(output_train.select("features").head(1))
    print(output_test.select("features").head(1))

    # prediction with logistic regression model
    lr = LogisticRegression(maxIter = 100, regParam = 0.01, elasticNetParam = 0.8)

    lrModel = lr.fit(output_train)

    #print "Coefficients: " + str(lrModel.coefficients)

    result = lrModel.transform(output_test)
    
    print "Logistic Regression Predictions: "

    result.select("prediction", "features").show(10)
    
    # create indexed features used for multi-class classification
    labelIndexer = StringIndexer(inputCol = "label", outputCol = "indexedLabel").fit(output_train)

    featureIndexer = VectorIndexer(inputCol = "features", outputCol = "indexedFeatures", maxCategories = 9).fit(output_train)
    
    # prediction with decision tree model
    dt = DecisionTreeClassifier(labelCol = "indexedLabel", featuresCol = "indexedFeatures")
    
    dt_pipeline = Pipeline(stages = [labelIndexer, featureIndexer, dt])
    
    dt_model = dt_pipeline.fit(output_train)
    
    dt_predictions = dt_model.transform(output_test)
    
    print "Decision Tree Predictions: "
    
    dt_predictions.select("prediction", "indexedFeatures").show(10)

    # prediction with gradient-boosted tree model
    gbt = GBTClassifier(labelCol = "indexedLabel", featuresCol = "indexedFeatures", maxIter = 100)
    
    gbt_pipeline = Pipeline(stages = [labelIndexer, featureIndexer, gbt])
    
    gbt_model = gbt_pipeline.fit(output_train)
    
    gbt_predictions = gbt_model.transform(output_test)
    
    print "Gradient-Boosted Tree Predictions: "
    
    gbt_predictions.select("prediction", "indexedFeatures").show(10)
    
    # prediction with random forest model
    rf = RandomForestClassifier(labelCol = "indexedLabel", featuresCol = "indexedFeatures", numTrees = 10)
    
    labelConverter = IndexToString(inputCol = "prediction", outputCol = "predictedLabel", labels = labelIndexer.labels)
    
    rf_pipeline = Pipeline(stages = [labelIndexer, featureIndexer, rf, labelConverter])
    
    rf_model = rf_pipeline.fit(output_train)
    
    rf_predictions = rf_model.transform(output_test)
    
    print "Random Forest Predictions: "
    
    rf_predictions.select("predictedLabel", "features").show(10)
    
    # prediction with naive bayes model
    nb = NaiveBayes(smoothing = 1.0, modelType = "multinomial")
    
    nb_model = nb.fit(output_train)
    
    nb_predictions = nb_model.transform(output_test)
    
    print "Naive Bayes Predictions: "
    nb_predictions.select("prediction", "features").show(10)
    
    spark.stop()

