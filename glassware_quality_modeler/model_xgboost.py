# Databricks notebook source
# MAGIC %sh 
# MAGIC sudo apt-get -y install wget
# MAGIC wget -P /tmp https://github.com/dmlc/xgboost/files/2161553/sparkxgb.zip

# COMMAND ----------

dbutils.fs.cp("file:/tmp/sparkxgb.zip", "/FileStore/username/sparkxgb.zip")

# COMMAND ----------

sc.addPyFile("/dbfs/FileStore/username/sparkxgb.zip")

# COMMAND ----------

# DBTITLE 1,Required ML Libs
import os
import pyspark

from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler, IndexToString, StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sparkxgb import XGBoostEstimator

# COMMAND ----------

def xgboost_model(stages, params, train, test):
  pipeline = Pipeline(stages=stages)
  
  with mlflow.start_run(run_name=mlflow_exp_name) as ml_run:
    for k,v in params.items():
      mlflow.log_param(k, v)
      
    mlflow.set_tag("state", "dev")
      
    model = pipeline.fit(train)
    predictions = model.transform(test)

    evaluator = MulticlassClassificationEvaluator(
                labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    predictions.select("predicted_quality", "quality").groupBy("predicted_quality", "quality").count().toPandas().to_pickle("confusion_matrix.pkl")
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_artifact("confusion_matrix.pkl")
    mlflow.spark.log_model(model, "spark-model")
    
    print("Documented with MLflow Run id %s" % ml_run.info.run_uuid)
  
  return model, predictions, accuracy, ml_run.info

# COMMAND ----------

def run_xgboost(df):
  
  (train_data, test_data) = df.randomSplit([0.8, 0.2])
  
  labelIndexer = StringIndexer(inputCol="quality", outputCol="indexedLabel").fit(df)  # Identify and index labels that could be fit through classification pipeline
  assembler = VectorAssembler(inputCols=['temp', 'pressure', 'duration'], outputCol="features_assembler").setHandleInvalid("skip")  # Incorporate all input fields as vector for classificaion pipeline
  scaler = StandardScaler(inputCol="features_assembler", outputCol="features")  # Scale input fields using standard scale
  labelConverter = IndexToString(inputCol="prediction", outputCol="predicted_quality", labels=labelIndexer.labels)  # Convert/Lookup prediction label index to actual label
  
  numTreesList = [10, 25, 50]
  learningRateList = [.1, .2, .3]
  
  for numTrees, learningRate in [(numTrees,learningRate) for numTrees in numTreesList for learningRate in learningRateList]:
    params = {"numTrees":numTrees, "learningRate":learningRate, "model": "XGBoost"}
    params.update(model_data_date)
    if run_exists(mlflow_exp_id, params):
      print("Trees: %s, learning Rate: %s, Run already exists"% (numTrees, learningRate))
    else:
      xgboost = XGBoostEstimator(labelCol="indexedLabel", featuresCol="features", eta=learningRate, maxDepth=maxDepth)
      model, predictions, accuracy, ml_run_info = xgboost_model([labelIndexer, assembler, scaler, rf, labelConverter], params, train_data, test_data)
      print("Trees: %s, learning Rate: %s, Accuracy: %s\n" % (numTrees, learningRate, accuracy))

  mlflow_search_query = "params.model = 'XGBoost' and params.model_data_date = '"+ model_data_date['model_data_date'] +"'"
  
  return best_run(mlflow_exp_id, mlflow_search_query)

# COMMAND ----------


