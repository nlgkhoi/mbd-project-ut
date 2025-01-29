from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col

from pyspark.sql.functions import col, count, countDistinct, when, concat
from pyspark.sql import Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType
from pyspark.sql.functions import udf
from pyspark.sql import Row


spark = SparkSession.builder.appName("mbd-prj").getOrCreate()

lymsys_df = spark.read.parquet("/user/s3485269/lmsys-chat-1m")

harmful_df = lymsys_df.select("conversation_id", "conversation", "openai_moderation", "model")

# Explode the conversation array to get one row per message
exploded_df = harmful_df.select(
	"model",
	F.explode(F.arrays_zip("conversation.role", "openai_moderation.flagged")).alias("message")
)
exploded_df.show(5)

# Count total messages and harmful messages by model and role
message_stats = exploded_df.groupBy("model").agg(
	F.count(F.when(F.col("message.role") == "assistant", 1)).alias("total_model_messages"),
	F.count(
		F.when(
			(F.col("message.role") == "assistant") & (F.col("message.flagged") == True),
			1
		)
	).alias("harmful_model_messages")
)

# Calculate percentage
message_stats = message_stats.withColumn(
	"harmful_percentage", 
	(F.col("harmful_model_messages") / F.col("total_model_messages") * 100).cast("double")
)

# Show results
message_stats.orderBy(F.desc("harmful_percentage")).show(30)

# Calculate the global percentage of harmful messages generated by the assistant
global_harmful_percentage = message_stats.agg(
	F.sum("harmful_model_messages") / F.sum("total_model_messages") * 100
).collect()[0][0]

print(f"Global percentage of harmful messages generated by the assistant: {global_harmful_percentage:.2f}%")
