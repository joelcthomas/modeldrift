# Databricks notebook source
# DBTITLE 1,Get Data and Config
# MAGIC %run ../utils/data_generator

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <h2> Data Modeling</h2>

# COMMAND ----------

# DBTITLE 1,Required ML Libs
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, IndexToString, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

# DBTITLE 1,Initialize MLflow Settings
import mlflow
import mlflow.mleap
import mlflow.spark
mlflow.set_experiment(mlflow_exp_loc)

# COMMAND ----------

# DBTITLE 1,Train & Test Data
(train_data, test_data) = df.randomSplit([0.8, 0.2])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <h2> Model Pipeline </h2>
# MAGIC 
# MAGIC ### Model -> Tune -> Evaluate -> MLflow

# COMMAND ----------

# Identify and index labels that could be fit through classification pipeline
labelIndexer = StringIndexer(inputCol="quality", outputCol="indexedLabel").fit(df)

# Incorporate all input fields as vector for classificaion pipeline
assembler = VectorAssembler(inputCols=['temp', 'pressure', 'duration'], outputCol="features_assembler").setHandleInvalid("skip")

# Scale input fields using standard scale
scaler = StandardScaler(inputCol="features_assembler", outputCol="features")

# Convert/Lookup prediction label index to actual label
labelConverter = IndexToString(inputCol="prediction", outputCol="predicted_quality", labels=labelIndexer.labels)

# COMMAND ----------

def classificationModel(stages, params, train, test):
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

numTreesList = [10, 25, 50]
maxDepthList = [3, 10, 5]
for numTrees, maxDepth in [(numTrees,maxDepth) for numTrees in numTreesList for maxDepth in maxDepthList]:
  params = {"numTrees":numTrees, "maxDepth":maxDepth, "model": "RandomForest"}
  params.update(dg_noise)
  params.update(model_data_date)
  rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=numTrees, maxDepth=maxDepth)
  model, predictions, accuracy, ml_run_info = classificationModel([labelIndexer, assembler, scaler, rf, labelConverter], params, train_data, test_data)
  print("Trees: %s, Depth: %s, Accuracy: %s\n" % (numTrees, maxDepth, accuracy))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get Best Run and Metric from MLflow

# COMMAND ----------

mlflow_experiment_id = ml_run_info.experiment_id
mlflowclient = mlflow.tracking.MlflowClient()
best_run = None
mlflow_search_query = "params.model = 'RandomForest' and params.model_data_date = '"+ model_data_date['model_data_date'] +"'"
runs = mlflowclient.search_runs([mlflow_experiment_id],"")
for run in runs:
  if best_run is None or run.data.metrics[model_compare_metric] > best_run[1]:
    best_run = (run.info.run_uuid,run.data.metrics[model_compare_metric])
best_runid = best_run[0]

# COMMAND ----------

mlflowclient.get_run(best_runid).to_dictionary()["data"]["params"]

# COMMAND ----------

mlflowclient.get_run(best_runid).to_dictionary()["data"]["metrics"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confusion Matrix for Best Run

# COMMAND ----------

artifact_uri = mlflowclient.get_run(best_runid).to_dictionary()["info"]["artifact_uri"]
confusion_matrix_uri = "/" + artifact_uri.replace(":","") + "/confusion_matrix.pkl"
confusion_matrix_uri

# COMMAND ----------

import pandas as pd
import numpy as np
confmat = pd.read_pickle(confusion_matrix_uri)
confmat = pd.pivot_table(confmat, values="count", index=["predicted_quality"], columns=["quality"], aggfunc=np.sum, fill_value=0)

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize=(4,4))

sns.heatmap(confmat, annot=True, fmt="d", square=True, cmap="OrRd")
plt.yticks(rotation=0)
plt.xticks(rotation=90)

display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Post Model Inference
# MAGIC Simulating based on data - Replace with getting model from MLflow

# COMMAND ----------

post_model_predictions = model.transform(df.union(post_df)) # User post_df eventually
# post_model_predictions = model.transform(post_df) # User post_df eventually

# COMMAND ----------

post_model_predictions = post_model_predictions.withColumn(
    'accurate_prediction',
    F.when((F.col('quality')==F.col('predicted_quality')), 1)\
    .otherwise(0)
)

# COMMAND ----------

from pyspark.sql import Window

# COMMAND ----------

prediction_summary = (post_model_predictions.groupBy(F.window(F.col('timestamp'), '1 day').alias('window'), F.col('predicted_quality'))
                      .count()
                      .withColumn('window_day', F.expr('to_date(window.start)'))
                      .withColumn('total',F.sum(F.col('count')).over(Window.partitionBy('window_day')))
                      .withColumn('ratio', F.col('count')*100/F.col('total'))
                      .select('window_day','predicted_quality', 'count', 'total', 'ratio')
                      .orderBy('window_day')
                     )
display(prediction_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Trend Showing Model Drift

# COMMAND ----------

accurate_prediction_summary = (post_model_predictions.groupBy(F.window(F.col('timestamp'), '1 day').alias('window'), F.col('accurate_prediction'))
                      .count()
                      .withColumn('window_day', F.expr('to_date(window.start)'))
                      .withColumn('total',F.sum(F.col('count')).over(Window.partitionBy('window_day')))
                      .withColumn('ratio', F.col('count')*100/F.col('total'))
                      .select('window_day','accurate_prediction', 'count', 'total', 'ratio')
                      .withColumn('accurate_prediction', F.when(F.col('accurate_prediction')==1, 'Accurate').otherwise('Inaccurate'))
                      .orderBy('window_day')
                     )

# COMMAND ----------

accurate_prediction_summary_2 = (post_model_predictions.groupBy(F.window(F.col('timestamp'), '1 day').alias('window'), F.col('accurate_prediction'))
                      .count()
                      .withColumn('window_day', F.expr('to_date(window.start)'))
                      .withColumn('total',F.sum(F.col('count')).over(Window.partitionBy('window_day')))
                      .withColumn('ratio', F.col('count')*100/F.col('total'))
                      .select('window_day','accurate_prediction', 'count', 'total', 'ratio')
                      .withColumn('accurate_prediction', F.when(F.col('accurate_prediction')==1, 'Accurate').otherwise('Inaccurate'))
                      .orderBy('window_day')
                     )

# COMMAND ----------

import matplotlib.patches as patches
sns.set(style='dark')
sns.set()
fig, ax = plt.subplots(figsize=(14,4))

sns.lineplot(x='window_day', y='ratio', hue='accurate_prediction', style='accurate_prediction', data = accurate_prediction_summary.filter(accurate_prediction_summary.window_day < '2019-07-21').toPandas())
sns.lineplot(x='window_day', y='ratio', hue='accurate_prediction', style='accurate_prediction', legend=False, alpha=0.2, data = accurate_prediction_summary.toPandas())
sns.lineplot(x='window_day', y='ratio', hue='accurate_prediction', style='accurate_prediction', legend=False, alpha=1, data = accurate_prediction_summary_2.toPandas())
plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.ylabel('% in population')
plt.xlabel('Date')

ax.axvline(x='2019-07-10', linewidth=1, linestyle='--', alpha=0.3)
ax.axvline(x='2019-07-19', linewidth=1, linestyle='--', alpha=0.3)
ax.axvline(x='2019-07-28', linewidth=1, linestyle='--', alpha=0.3)

ax.legend(bbox_to_anchor=(1.1, 1.05))

rect = patches.Rectangle(
    xy=(ax.get_xlim()[0], 80),  
    width=ax.get_xlim()[1]-ax.get_xlim()[0],  
    height=20,
    color='green', alpha=0.1, ec='red'
)
ax.add_patch(rect)

fig.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

display(accurate_prediction_summary.filter(accurate_prediction_summary.window_day > '2019-07-20'))

# COMMAND ----------


