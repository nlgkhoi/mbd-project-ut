import time
import numpy as np
from pyspark.sql.functions import col,desc,split,regexp_extract, row_number, when,explode, lag, sum as spark_sum, avg,count, lit
from pyspark.sql import SparkSession
from pyspark.sql.functions import posexplode, element_at

spark = SparkSession.builder.appName("language_use").getOrCreate()

wildchat_path = "/user/s3485269/WildChat-1M-Full"
lmsys_path = "/user/s3485269/lmsys-chat-1m"

df_wildchat = spark.read.parquet(wildchat_path)
df_lmsys = spark.read.parquet(lmsys_path)

df_wildchat = df_wildchat.withColumn('model_tag', regexp_extract('model', r'(gpt-\d(?:\.\d)?)', 1))

df_wildchat = df_wildchat.filter(col("language") == "English")

df_sel = df_wildchat.select("model_tag", "country", "conversation", "openai_moderation")

df_exploded = df_sel.select(
    "model_tag",
    "country",
    posexplode("conversation").alias("pos", "utterance"),
    col("openai_moderation")
)

df_exploded = df_exploded.withColumn(
    "moderation",
    element_at(col("openai_moderation"), col("pos") + 1)
)

df_utterance = df_exploded.select(
    col("model_tag"),
    col("country"),
    col("utterance.turn_identifier").alias("turn_identifier"),
    col("utterance.content").alias("content"),
    col("utterance.role").alias("role"),
    col("utterance.toxic").alias("toxic"),
    col("utterance.redacted").alias("redacted"),
    col("moderation.categories.harassment").alias("harassment"),
    col("moderation.categories.hate").alias("hate"),
    col("moderation.categories.`self-harm`").alias("self_harm"),
    col("moderation.categories.sexual").alias("sexual"),
    col("moderation.categories.violence").alias("violence")
)

features = ["harassment", "hate", "self_harm", "sexual", "violence"]
for feature in features:
    df_utterance = df_utterance.withColumn(
        feature, when(col(feature) == True, True).otherwise(False)
    )

# df_utterance.show(truncate=False)

# toxic analysis group by country
total_samples = df_utterance.groupBy("country").count().alias("total_count").withColumnRenamed("count", "total_count").orderBy(desc("total_count"))
top_countries = total_samples.orderBy(desc("total_count")).limit(10)

toxic_samples = df_utterance.filter(col("toxic") == True).groupBy("country").count().alias("toxic_count").withColumnRenamed("count", "toxic_count")
toxic_ratio = top_countries.join(toxic_samples, on="country", how="left") \
                           .withColumn("toxic_ratio", col("toxic_count") / col("total_count")) \
                           .select("country", "toxic_count", "total_count", "toxic_ratio") \
                           .orderBy(desc("toxic_ratio"))
toxic_ratio.show(10, truncate=False)
"""
+--------------+-----------+-----------+--------------------+
|country       |toxic_count|total_count|toxic_ratio         |
+--------------+-----------+-----------+--------------------+
|Germany       |14299      |102080     |0.1400764106583072  |
|France        |10574      |77116      |0.13711810778567354 |
|United States |75960      |799324     |0.0950303006040104  |
|Australia     |5282       |68636      |0.07695669910833965 |
|Russia        |13053      |201050     |0.06492414822183537 |
|United Kingdom|10235      |165132     |0.0619807184555386  |
|Canada        |4491       |95118      |0.047215038163123696|
|India         |2006       |106384     |0.01885621898029779 |
|China         |1132       |106300     |0.010649106302916274|
|Hong Kong     |884        |93346      |0.009470143337689885|
+--------------+-----------+-----------+--------------------+
"""
# PII analysis group by country
total_samples = df_utterance.groupBy("country").count().alias("total_count").withColumnRenamed("count", "total_count").orderBy(desc("total_count"))
top_countries = total_samples.orderBy(desc("total_count")).limit(10)

pii_samples = df_utterance.filter(col("redacted") == True).groupBy("country").count().alias("pii_count").withColumnRenamed("count", "pii_count")
pii_ratio = top_countries.join(pii_samples, on="country", how="left") \
                           .withColumn("pii_ratio", col("pii_count") / col("total_count")) \
                           .select("country", "pii_count", "total_count", "pii_ratio") \
                           .orderBy(desc("pii_ratio"))
pii_ratio.show(10, truncate=False)
"""
+--------------+---------+-----------+---------------------+
|country       |pii_count|total_count|pii_ratio            |
+--------------+---------+-----------+---------------------+
|United States |6305     |799324     |0.007887915288418713 |
|India         |520      |106384     |0.004887953075650474 |
|Australia     |256      |68636      |0.003729821085144822 |
|United Kingdom|574      |165132     |0.003476007073129375 |
|Hong Kong     |293      |93346      |0.0031388597261800184|
|Canada        |232      |95118      |0.0024390756744254506|
|China         |221      |106300     |0.002079021636876764 |
|France        |151      |77116      |0.0019580891125058355|
|Russia        |323      |201050     |0.001606565530962447 |
|Germany       |123      |102080     |0.0012049373040752352|
+--------------+---------+-----------+---------------------+
"""