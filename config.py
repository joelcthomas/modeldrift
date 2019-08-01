# Databricks notebook source
# Noise for data generator
dg_noise = {"temp_noise": 0.2, "pressure_noise": 0.2, "duration_noise": 0.2}

userid = 'add_user'

# Data paths (replace with actual locations. Could be directly to S3, Azure blob/ADLS, or these locations mounted locally)
sensor_reading_blob = "/mnt/tmp/sensor_reading"
product_quality_blob = "/mnt/tmp/product_quality"

predicted_quality_blob = "/mnt/tmp/predicted_quality"
predicted_quality_cp_blob = "/mnt/tmp/predicted_quality_checkpoint"

# Modeling & MLflow settings
mlflow_exp_name = "Glassware Quality Predictor"
mlflow_exp_id = "3650654" # Replace with id from your environment

model_compare_metric = 'accuracy'

# COMMAND ----------

# MAGIC %run ./utils/viz_utils

# COMMAND ----------

# MAGIC %run ./utils/mlflow_utils

# COMMAND ----------

from pyspark.sql import Window
import pyspark.sql.functions as F
