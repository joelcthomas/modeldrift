# Databricks notebook source
## Example to access Azure blob directly
# storage_account_name = "acctname"
# storage_account_access_key = "addkeyhere=="
# spark.conf.set(
#   "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
#   storage_account_access_key)

# COMMAND ----------

## Example mount
# dbutils.fs.mount(
#   source = "wasbs://acctname@blobstorename.blob.core.windows.net/",
#   mount_point = "/mnt/glassware",
#   extra_configs = {"fs.azure.account.key.joelsimpleblobstore.blob.core.windows.net": storage_account_access_key}
# )

# COMMAND ----------

# MAGIC %run ./sensor_reading

# COMMAND ----------

# MAGIC %run ./product_quality

# COMMAND ----------

# MAGIC %run ./predicted_quality
