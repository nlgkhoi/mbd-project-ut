from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col

from pyspark.sql.functions import col, count, countDistinct, when, concat
from pyspark.sql import Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType
from pyspark.sql.functions import udf
from pyspark.sql import Row


spark = SparkSession.builder.appName("mbd-prj").getOrCreate()

wildchat_df = spark.read.parquet("/user/s3485269/WildChat-1M-Full")

# filter out rows where 'country' is null
wildchat_df = wildchat_df.filter(F.col("country").isNotNull())

print(wildchat_df.show())

# open the parquet file
country_state_df = spark.read.parquet("/user/s3485269/country_state_timezone.parquet")
print(country_state_df.count()) # 2433

# join condition to handle cases when 'state' is null
wildchat_df = wildchat_df.join(
    country_state_df,
    (
        (wildchat_df["country"] == country_state_df["ccountry"]) &
        (
            (wildchat_df["state"] == country_state_df["sstate"]) |
            (wildchat_df["state"].isNull() & country_state_df["sstate"].isNull())
        )
    ),
    "left"
)


# Select the required columns and filter to show rows where 'timezone' is null, supposed to be 1262 (only a small fraction)
wildchat_df.select("country", "state", "timezone").filter(F.col("timezone").isNull()).count() # 1262

# drop ccountry and sstate columns
wildchat_df = wildchat_df.drop("ccountry", "sstate")
wildchat_df.columns

# Shift the timestamp by adding the GMT offset hours
wildchat_df = wildchat_df.withColumn(
	"offseted_timestamp",
	(F.unix_timestamp("timestamp") + F.col("gmt_offset.hours") * 3600 + F.col("gmt_offset.minutes") * 60).cast("timestamp")
)

##############


wildchat_df.select("timestamp", "gmt_offset", "offseted_timestamp").show(5)

user_counts = wildchat_df.groupBy("hashed_ip").agg(F.count("*").alias("conv_count"))

# Find earliest date of usage for each user
# Group by hashed_ip and find the minimum timestamp
earliest_date = wildchat_df.groupBy("hashed_ip").agg(F.min("timestamp").alias("earliest_date"))

# Find latest date of usage for each user
# Group by hashed_ip and find the maximum timestamp
latest_date = wildchat_df.groupBy("hashed_ip").agg(F.max("timestamp").alias("latest_date"))

# Join with user_counts
user_counts = user_counts.join(earliest_date, "hashed_ip").join(latest_date, "hashed_ip")

# Find avg number of conversations per day for each user
# Calculate the number of days between earliest and latest date
user_counts = user_counts.withColumn("num_days", F.datediff("latest_date", "earliest_date"))
# Calculate avg number of conversations per day
user_counts = user_counts.withColumn("avg_conv_per_day", F.col("conv_count") / F.col("num_days"))

# # Calculate the 90th percentile of avg_conv_per_day
# percentile_90 = wildchat_df.approxQuantile("avg_conv_per_day", [0.90], 0.001)[0]
# print(f'90th percentile: {percentile_90}')

# Filter out users with avg_conv_per_day > 1
repeat_users = user_counts.filter(F.col("avg_conv_per_day") > 2)
repeat_users.count()

# Count the number of conversations after filtering
repeat_users.join(wildchat_df, "hashed_ip").count() # 406092

repeat_users.orderBy("avg_conv_per_day", ascending=False).show(5)

# Remove outliers, which are users that have significantly more/fewer conversations 
# Calculate the 90th percentile of avg_conv_per_day
percentile_90 = repeat_users.approxQuantile("avg_conv_per_day", [0.90], 0.001)[0]
print(f'90th percentile: {percentile_90}')
# Calculate the 10th percentile of avg_conv_per_day
percentile_10 = repeat_users.approxQuantile("avg_conv_per_day", [0.10], 0.001)[0]
print(f'10th percentile: {percentile_10}')

# # Filter out users with avg_conv_per_day outside the 10th and 90th percentile
# repeat_users_filtered = repeat_users.filter(F.col("avg_conv_per_day").between(percentile_10, percentile_90))

# Filter out users with avg_conv_per_day less than 90th percentile
repeat_users_filtered = repeat_users.filter(F.col("avg_conv_per_day") < percentile_90)

# Calculate the number of users after filtering
num_repeat_users_filtered = repeat_users_filtered.count() 
print(f"Number of repeat users after filtering: {num_repeat_users_filtered:,}")

repeat_users_filtered.orderBy("avg_conv_per_day", ascending=False).show(5)
repeat_users_filtered.orderBy("avg_conv_per_day", ascending=True).show(5)

# Extract the day of the week from the timestamp of those users
# Join with the original dataframe
repeat_users_df = wildchat_df.join(repeat_users_filtered, "hashed_ip")

# Extract day of the week
repeat_users_df = repeat_users_df.withColumn("day_of_week", F.dayofweek("timestamp"))

# Create a new column 'model_family' based on the model name
wildchat_df = wildchat_df.withColumn(
	"model_family",
	F.when(F.col("model").like("gpt-3.5%"), "GPT-3.5")
	 .when(F.col("model").like("gpt-4%"), "GPT-4")
	 .otherwise("Other")
)

# Group by model family and month and year and calculate avg turn
model_month_year = wildchat_df.groupBy("model_family", F.year("offseted_timestamp").alias("year"), F.month("offseted_timestamp").alias("month")).agg(F.avg("turn").alias("avg_turns")).orderBy("year", "month")

# Create a column of month-year but actually the timestamp of the first date of the month of that year
model_month_year = model_month_year.withColumn("month_year_timestamp", F.to_timestamp(F.concat_ws("-", F.col("year"), F.col("month"), F.lit("01"))))
model_month_year.show(12)

