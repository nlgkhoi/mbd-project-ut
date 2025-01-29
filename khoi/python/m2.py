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

# Divide 24 hours to 24 bins and assign each hour to a bin
# and assign each row to a bin based on the timestamp
wildchat_df = wildchat_df.withColumn("hour_bin_offseted", F.hour("offseted_timestamp"))

# Group by country and hour_bin and count the number of conversations, omit NULL country
timezone_hour_counts_offseted = wildchat_df.filter(F.col("country").isNotNull()).groupBy("country", "hour_bin_offseted").count().orderBy("country", "hour_bin_offseted")

# Aggregate groups by country and find the top 5 hours with the most conversations, get them into a list in a column called 'peak_hours'
window = Window.partitionBy("country").orderBy(F.col("count").desc())
timezone_hour_counts_offseted = timezone_hour_counts_offseted.withColumn("rank", F.rank().over(window))
timezone_hour_counts_offseted = timezone_hour_counts_offseted.filter(F.col("rank") <= 5)
timezone_hour_counts_offseted = timezone_hour_counts_offseted.groupBy("country").agg(F.collect_list("hour_bin_offseted").alias("offseted_peak_hours"))

# Show the results
timezone_hour_counts_offseted.show(20)

from pyspark.sql.functions import udf

import sys
sys.path.insert(0, './dependencies.zip')
import pycountry_convert as pc

def country_to_continent(country_name):
	try:
		# Handle special cases
		if country_name == 'United States':
			return 'North America'
		elif country_name == 'Russia':
			return 'Europe'
		elif country_name == 'The Netherlands':
			return 'Europe'
		elif country_name == 'DR Congo':
			return 'Africa'
		elif country_name == 'Kosovo':
			return 'Europe'
		elif country_name == 'Bonaire, Sint Eustatius, and Saba':
			return 'South America'
		elif country_name == 'Congo Republic':
			return 'Africa'
		elif country_name == 'St Vincent and Grenadines':
			return 'South America'
		elif country_name == 'Timor-Leste':
			return 'Asia'
		elif country_name == 'Sint Maarten':
			return 'South America'

		# Get country alpha-2 code
		country_alpha2 = pc.country_name_to_country_alpha2(country_name)
		# Get continent code
		country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
		# Get continent name
		country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
		return country_continent_name
	except:
		print(f"Continent not found for {country_name}")
		return None

# Test the function
print(country_to_continent("United States"))  # North America
print(country_to_continent("Russia"))  # Europe
print(country_to_continent("India"))  # Asia
print(country_to_continent("Narnia"))  # None
print(country_to_continent("Australia"))  # Oceania
print(country_to_continent("Japan"))  # Asia

# Create UDF
country_to_continent_udf = udf(country_to_continent, StringType())

# Add continent column
wildchat_df = wildchat_df.withColumn("continent", country_to_continent_udf(col("country")))

# Extract day of week and hour from offseted_timestamp
wildchat_df = wildchat_df.withColumn("day_of_week", F.dayofweek("offseted_timestamp"))

# Group by continent and hour, count conversations, exclude weekend (day 2 to 6)
continent_hour_counts = wildchat_df.filter(col("day_of_week").between(2, 6)).groupBy("continent", "hour_bin_offseted")\
	.count() \
	.orderBy("continent", "hour_bin_offseted")

# Calculate percentage by continent
total_conversations_per_continent = wildchat_df.groupBy("continent").agg(F.count("*").alias("total_conversations_per_continent"))
continent_hour_counts = continent_hour_counts.join(total_conversations_per_continent, "continent")
continent_hour_counts = continent_hour_counts.withColumn("percentage", (F.col("count") / F.col("total_conversations_per_continent")).cast("double") * 100)
continent_hour_counts = continent_hour_counts.orderBy("continent", "hour_bin_offseted")

continent_hour_counts.show(5)
