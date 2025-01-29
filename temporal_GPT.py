import time

# Initialize SparkSession
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, unix_timestamp, hour, avg, count, min, max, row_number, lag, dayofweek, size, split, when

spark = SparkSession.builder \
    .appName("WildChat Temporal Analysis") \
    .getOrCreate()

# Load datasets
wildchat_folder = "hdfs:///user/s3485269/WildChat-1M-Full"
wildchat_df = spark.read.parquet(wildchat_folder)
country_state_path = "hdfs:///user/s3506932/country_state_timezone.parquet"
country_state_df = spark.read.parquet(country_state_path)

# Start timer
start_time = time.time()
conversation_window = Window.partitionBy("conversation_hash").orderBy("turn")
sentiment_trend = wildchat_df.withColumn(
    "turn_index", row_number().over(conversation_window)
).withColumn(
    "toxicity_score", col("detoxify_moderation")[0]["toxicity"]
).groupBy("turn_index").agg(
    avg("toxicity_score").alias("avg_sentiment_score")
).orderBy("turn_index")

sentiment_trend.write.csv("sentiment_trend.csv", header=True, mode="overwrite")

# End timer and calculate runtime
end_time = time.time()
total_runtime = end_time - start_time

# Write runtime to file locally
with open("runtime.txt", "w") as runtime_file:
    runtime_file.write(f"Total Runtime: {total_runtime:.2f} seconds\\n")
