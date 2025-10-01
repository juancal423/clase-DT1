# Databricks notebook source
# MAGIC %md
# MAGIC # Predicción en Streaming con Spark ML y Spark Streaming

# COMMAND ----------

# MAGIC %md
# MAGIC En este notebook vamos a entrenar un modelo de clasificación para predecir la probabilidad de un paciente de sufrir un ataque al corazón

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.sql.types import StructType,StructField,LongType, StringType,DoubleType,TimestampType

from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.classification import LogisticRegression

# COMMAND ----------

schema = StructType( \
                     [StructField("age", LongType(),True), \
                      StructField("sex", LongType(), True), \
                      StructField("cp", LongType(), True), \
                      StructField('trtbps', LongType(), True), \
                      StructField("chol", LongType(), True), \
                      StructField("fbs", LongType(), True), \
                      StructField("restecg", LongType(), True), \
                      StructField("thalachh", LongType(), True),\
                      StructField("exng", LongType(), True), \
                      StructField("oldpeak", DoubleType(), True), \
                      StructField("slp", LongType(),True), \
                      StructField("caa", LongType(), True), \
                      StructField("thall", LongType(), True), \
                      StructField("output", LongType(), True), \
                        ])

# COMMAND ----------

import pandas as pd

# Leer el archivo CSV con Pandas
df_pd = pd.read_csv("/Workspace/Users/pilimanga592@gmail.com/clase/heart.csv")

# Realizar procesamiento o análisis con Pandas
# ...

# Convertir el DataFrame de Pandas a un DataFrame de Spark
df = spark.createDataFrame(df_pd)

# Realizar operaciones con Spark
df = df.withColumnRenamed("output", "label")
df.show()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

testDF, trainDF = df.randomSplit([0.3, 0.7])

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10, regParam=0.01)
lr.setMaxIter(10)
lr.setRegParam(0.01)
lr.setFeaturesCol('features')
lr.setLabelCol('label')

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10, regParam=0.01)



# COMMAND ----------

# Create the logistic regression model
lr = LogisticRegression(maxIter=10, regParam= 0.01)

# COMMAND ----------

# We create a one hot encoder.
ohe = OneHotEncoder(inputCols = ['sex', 'cp', 'fbs', 'restecg', 'slp', 
                                 'exng', 'caa', 'thall'], 
                    outputCols=['sex_ohe', 'cp_ohe', 'fbs_ohe', 
                                'restecg_ohe', 'slp_ohe', 'exng_ohe', 
                                'caa_ohe', 'thall_ohe'])

# Input list for scaling
inputs = ['age','trtbps','chol','thalachh','oldpeak']

# We scale our inputs
assembler1 = VectorAssembler(inputCols=inputs, outputCol="features_scaled1")
scaler = MinMaxScaler(inputCol="features_scaled1", outputCol="features_scaled")

# We create a second assembler for the encoded columns.
assembler2 = VectorAssembler(inputCols=['sex_ohe', 'cp_ohe', 
                                        'fbs_ohe', 'restecg_ohe', 
                                        'slp_ohe', 'exng_ohe', 'caa_ohe', 
                                        'thall_ohe','features_scaled'], 
                             outputCol="features")


# COMMAND ----------

# Create stages list
myStages = [assembler1, scaler, ohe, assembler2,lr]

# Set up the pipeline
pipeline = Pipeline(stages= myStages)

# COMMAND ----------

# We fit the model using the training data.
pModel = pipeline.fit(trainDF)

# We transform the data.
trainingPred = pModel.transform(trainDF)

# # We select the actual label, probability and predictions
trainingPred.select('label','probability','prediction').show()

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(
    labelCol='label', 
    predictionCol='prediction', 
    metricName='accuracy'
)

# COMMAND ----------

# Accuracy
accuracy = evaluator.evaluate(trainingPred)
print('Train Accuracy = ', accuracy)
porcentaje = accuracy * 100
print('Train Accuracy = ', porcentaje,'%')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creando predicciones en Streaming

# COMMAND ----------

testData = testDF.repartition(10)
testData

#Remove directory in case we rerun it multiple times.
dbutils.fs.rm("FileStore/tables/HeartTest/",True)

#Create a directory
testData.write.format("CSV").option("header",True).save("FileStore/tables/HeartTest/")

# COMMAND ----------

# Source
sourceStream=spark.readStream.format("csv").option("header",True).schema(schema).option("ignoreLeadingWhiteSpace",True).option("mode","dropMalformed").option("maxFilesPerTrigger",1).load("dbfs:/FileStore/tables/HeartTest").withColumnRenamed("output","label")

# COMMAND ----------

prediction1 = pModel.transform(sourceStream).select('label',
                                                   'probability',
                                                   'prediction')

# COMMAND ----------

prediction1.writeStream.format("console").option("truncate", False).start()

# COMMAND ----------


display(prediction1.limit(5))
