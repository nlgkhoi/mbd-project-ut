# What languages are most commonly used, and how often do language switches occur?
# load the data
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,desc,split,regexp_extract, row_number, when,explode, lag, sum, avg,count, lit
from pyspark.sql.window import Window
from pyspark.ml.feature import Tokenizer,HashingTF, IDF, StandardScaler, StopWordsRemover
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import time
# traditional module
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score, precision_score, recall_score

# spark = SparkSession.builder.appName("language_use").getOrCreate()
spark = SparkSession.builder \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

wildchat_path = "/user/s3505235/WildChat-1M"# "../data/WildChat-1M" # full version of the data, replace it with the HDFS path
lmsys_path = "/user/s3505235/lmsys-chat-1m" # "../data/lmsys-chat-1m" # this is not neccessary, we can just use the wildchat data
# load the data
df_wildchat = spark.read.parquet(wildchat_path)
df_lmsys = spark.read.parquet(lmsys_path)
# print(f"Number of rows in df1 before deduplication: {df_wildchat.count()}")
# print(f"Number of rows in df2 before deduplication: {df_lmsys.count()}")

# -------start to process df_wild---------
# dftm_wildchat = df_wildchat.withColumn("turn_identifier", col("conversation")[0]["turn_identifier"])
# dftm_wildchat = dftm_wildchat.dropDuplicates(["turn_identifier"])
# dftm_wildchat = dftm_wildchat.drop("turn_identifier")
# df_wildchat = dftm_wildchat
# ----------end of the process------

# df_lmsys = df_lmsys.dropDuplicates(["conversation_id"])
# print(f"Number of rows in df1 after deduplication: {df_wildchat.count()}")
# print(f"Number of rows in df2 after deduplication: {df_lmsys.count()}")
common_columns = list(set(df_wildchat.columns).intersection(set(df_lmsys.columns)))
# in the very beginning, we will just count the most common languages used in the conversation
df1_wildchat = df_wildchat.select(common_columns).drop("openai_moderation","conversation")
df1_lmsys = df_lmsys.select(common_columns).drop("openai_moderation","conversation")

# print(df_wildchat.columns)
# print(df_lmsys.schema)
df = df1_wildchat.unionByName(df1_lmsys)
# df.show(5)

# first, we will directly use the language col of merged data to see the most commonly used languages
# language_count = df.groupBy("language").count().sort(desc("count"))
# top_language = language_count.limit(5)
# top_language.show()
# top_languages_pd = top_language.toPandas()
# # plot the top 5 languages
# plt.figure(figsize=(10, 6))
# plt.bar(top_languages_pd["language"], top_languages_pd["count"])
# plt.xlabel("Language")
# plt.ylabel("Count")
# plt.title("Top 5 Most Frequent Languages")
# plt.show()

# extract model from the model col
df_wildchat = df_wildchat.withColumn('model_tag', regexp_extract('model', r'(gpt-\d(?:\.\d)?)', 1))

# df_wildchat.show(5)
# language_count = df_wildchat.groupBy("model_tag", "language").count().orderBy("count", ascending=False)

# window_spec = Window.partitionBy("model_tag").orderBy(desc("count"))
# top_language = language_count.withColumn("rank", row_number().over(window_spec)).filter(col("rank") <= 5)

# top_language.show()
# top_language_pd = top_language.toPandas()
# plot the top 5 languages

# pivot_df = top_language_pd.pivot(index='model_tag', columns='language', values='count')
# pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
# plt.title('Top 5 Languages by Model (Stacked Bar Chart)', fontsize=16)
# plt.xlabel('Model Label', fontsize=12, )
# plt.ylabel('Language Count', fontsize=12)
# plt.legend(title='Language', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# draw pie chart for each model
# model_tags = top_language_pd['model_tag'].unique()
# for model in model_tags:
#     model_data = top_language_pd[top_language_pd['model_tag'] == model]
    
#     languages = model_data['language']
#     counts = model_data['count']
    
#     plt.figure(figsize=(8, 6))
#     plt.pie(counts, labels=languages, autopct='%1.1f%%', startangle=140)
#     plt.title(f"Top 5 Languages for {model}")
#     plt.show()

# In the section, we will find the users' languages(English, Chinese, Russian, French, Spanish, German) in different countries(areas)
# country_count = df_wildchat.groupBy("country").count().sort(desc("count"))
# top_country = country_count.limit(20)
# top_country.show()
"""
+---------------+------+
|        country| count|
+---------------+------+
|  United States|170520|
|         Russia|113881|
|          China|101156|
|      Hong Kong| 46955|
| United Kingdom| 30808|
|        Germany| 28975|
|         France| 26065|
|          Japan| 17567|
|          India| 16676|
|         Canada| 16389|
|          Egypt| 14310|
|      Singapore| 13401|
|         Taiwan| 11927|
|         Brazil| 11621|
|       DR Congo|  9874|
|    Philippines|  9293|
|        Türkiye|  9035|
|          Italy|  8432|
|The Netherlands|  8262|
|        Vietnam|  8184|
+---------------+------+
"""
# df_filtered = df_wildchat.filter(col("country").isin([
#     "United States", "Russia", "China", "Hong Kong", "United Kingdom", 
#     "Germany", "France", "Japan", "India", "Canada", 
#     "Egypt", "Singapore", "Taiwan", "Brazil", "DR Congo", 
#     "Philippines", "Türkiye", "Italy", "The Netherlands", "Vietnam"
# ]))
# languages_to_keep = ["English", "Chinese", "Russian", "French", "Spanish", "German", "Dutch", "Japanese", "Portuguese"]
# df_filtered = df_filtered.withColumn(
#     "language",
#     when(col("language").isin(languages_to_keep), col("language")).otherwise("others")
# )

# df_filtered = df_filtered.groupBy("country", "language").count().sort(desc("count"))
# window_spec = Window.partitionBy("country").orderBy(desc("count"))
# top_language = df_filtered.withColumn("rank", row_number().over(window_spec)).filter(col("rank") <= 5)
# top_language.show(10)
# top_language_pd = top_language.toPandas()

# pivot columns plot ->
# pivot_data = top_language_pd.pivot(index="country", columns="language", values="count").fillna(0)

# pivot_data.plot(kind="bar", stacked=True, figsize=(14, 8))
# plt.title("Top Languages by Country/Region", fontsize=16)
# plt.xlabel("Country/Region", fontsize=14)
# plt.ylabel("Usage Count", fontsize=14)
# plt.xticks(rotation=45)
# plt.legend(title="Language", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# np.random.seed(42)

# country_coords = {
#     "United States": (37.0902, -95.7129),
#     "Russia": (61.5240, 105.3188),
#     "China": (35.8617, 104.1954),
#     "Hong Kong": (22.3193, 114.1694),
#     "United Kingdom": (55.3781, -3.4360),
#     "Germany": (51.1657, 10.4515),
#     "France": (46.6034, 1.8883),
#     "Japan": (36.2048, 138.2529),
#     "India": (20.5937, 78.9629),
#     "Canada": (56.1304, -106.3468),
#     "Egypt": (26.8206, 30.8025),
#     "Singapore": (1.3521, 103.8198),
#     "Taiwan": (23.6978, 120.9605),
#     "Brazil": (-14.2350, -51.9253),
#     "DR Congo": (-4.0383, 21.7587),
#     "Philippines": (12.8797, 121.7740),
#     "Türkiye": (38.9637, 35.2433),
#     "Italy": (41.8719, 12.5674),
#     "The Netherlands": (52.1326, 5.2913),
#     "Vietnam": (14.0583, 108.2772)
# }
# top_language_pd["lat"] = top_language_pd["country"].map(lambda x: country_coords[x][0])
# top_language_pd["lon"] = top_language_pd["country"].map(lambda x: country_coords[x][1])
# top_language_pd["lat"] += np.random.uniform(-1.5, 1.5, size=len(top_language_pd))
# top_language_pd["lon"] += np.random.uniform(-1.5, 1.5, size=len(top_language_pd))

# fig = px.scatter_geo(
#     top_language_pd,
#     lat="lat",  
#     lon="lon",  
#     size="count",  
#     color="language",  
#     hover_name="country",  
#     hover_data=["language", "count"],  
#     title="Language Usage by Country (Adjusted Points)",
#     projection="natural earth"  
# )

# fig.update_traces(marker=dict(sizeref=100, sizemin=5))
# fig.show()

# In this section, we will find the language used and time, as the time is UTC, we will not consider every time zone
# for example, we can consider time zone in Nettherlands. This is not my part
# then, we will exact the language use from convesation and analyse(count) the common language used by the users and assitants
# print(df_wildchat.columns)

# language switches from the conversation column
df_exploded = df_wildchat.select(col("model_tag"), explode(col("conversation")).alias("utterance"))

# df_parsed = df_exploded.select(
#     col("model_tag"),
#     col("utterance.turn_identifier").alias("turn_identifier"),
#     col("utterance.language").alias("language"),
#     col("utterance.role").alias("role")
# )

# df_parsed = df_parsed.withColumn(
#     "order_col",
#     when(col("role") == "user", 1).otherwise(2)
# )

# window_spec = Window.partitionBy("turn_identifier").orderBy("order_col")

# df_parsed = df_parsed.withColumn(
#     "prev_language",
#     lag("language").over(window_spec)
# ).withColumn(
#     "prev_role",
#     lag("role").over(window_spec)
# )

# df_parsed = df_parsed.withColumn(
#     "language_switch",
#     when(
#         (col("role") != col("prev_role")) & (col("language") != col("prev_language")),
#         1
#     ).otherwise(0)
# )

# df_filter = df_parsed.filter(col("role") == "assistant")

# switch_sum = df_parsed.groupBy("model_tag").agg(sum("language_switch").alias("language_switches"), count("turn_identifier").alias("total_turns"))
# # switch_sum.show()
# # we also want know that what language by user will result in a language switch
# df_switch = df_parsed.filter(col("language_switch") == 1)
# window_spec = Window.partitionBy("model_tag").orderBy(col("count").desc())
# df_switch = (
#     df_switch.groupBy("model_tag","prev_language")
#     .count()
#     .withColumn("rank", row_number().over(window_spec))
# )

# top_switches = df_switch.filter(col("rank") <= 5)
# top_switches.show()
# df_switch.groupBy("model_tag", "prev_language").count().orderBy("model_tag","count", ascending=False).show()
# we will group the language model


# Second, we will analyse the PII data in the conversation, 
# we also want apply machine learning to train classification model in our big data contexts
# df_exploded = df_exploded.filter(col("language") == "English") # filter the English language(optional)


df_parsed = df_exploded.select(
    col("utterance.turn_identifier").alias("turn_identifier"),
    col("utterance.role").alias("role"),
    col("utterance.content").alias("content"),
    col("utterance.redacted").alias("redacted")
)
# print("point 1")

# df_parsed = df_exploded.select(
#     col("utterance.turn_identifier").alias("turn_identifier"),
#     col("utterance.role").alias("role"),
#     col("utterance.content").alias("content"),
#     col("utterance.redacted").alias("redacted"),
#     col("utterance.language").alias("language")
# ).filter(col("language") == "English")
# df_parsed = df_parsed.drop("language")

# df_parsed.show(5, truncate=True)
# df_parsed.groupBy("redacted","language").count().orderBy("count", ascending=False).show()
# filter english language (optional)

# df_parsed.groupBy("redacted").count().show()
# we want to use logistic regression to predict the redacted column
train_num = 5000
test_num = 1000

df_false = df_parsed.filter(col("redacted") == False)
train_false = df_false.sample(False, fraction=train_num / df_false.count()).limit(train_num)
remaining_false = df_false.subtract(train_false)
test_false = remaining_false.sample(False, fraction=test_num / remaining_false.count()).limit(test_num)

df_true = df_parsed.filter(col("redacted") == True)
train_true = df_true.sample(False, fraction=train_num / df_true.count()).limit(train_num)
remaining_true = df_true.subtract(train_true)
test_true = remaining_true.sample(False, fraction=test_num / remaining_true.count()).limit(test_num)

train_false = train_false.withColumn("label", lit(0))
test_false = test_false.withColumn("label", lit(0))
train_true = train_true.withColumn("label", lit(1))
test_true = test_true.withColumn("label", lit(1))

train_data = train_false.union(train_true)
test_data = test_false.union(test_true)
# remover = StopWordsRemover(inputCol="content", outputCol="clean_content")

# tokenization
tokenizer = Tokenizer(inputCol="content", outputCol="words")
train_data = tokenizer.transform(train_data)
test_data = tokenizer.transform(test_data)

hashingTF = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=10000)
train_data = hashingTF.transform(train_data)
test_data = hashingTF.transform(test_data)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(train_data)
train_data = idf_model.transform(train_data)
test_data = idf_model.transform(test_data)

# standardization
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=False, withStd=True)
scaler_model = scaler.fit(train_data)
train_data = scaler_model.transform(train_data)
test_data = scaler_model.transform(test_data)

train_data.show(5)
df_filtered = df_parsed.filter(col("role") == "user")

start = time.time()
model = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
model = model.fit(train_data)
end = time.time()
print(f"pyspark Training time: {end - start}")

# evaluate the model
predictions = model.transform(test_data)
evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_accuracy.evaluate(predictions)

evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
precision = evaluator_precision.evaluate(predictions)

evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
recall = evaluator_recall.evaluate(predictions)

print(f"Test set accuracy = {accuracy}, precision = {precision}, recall = {recall}")

# build a param grid
model = LogisticRegression(featuresCol="features", labelCol="label")
paramGrid = ParamGridBuilder() \
    .addGrid(model.maxIter, [10, 50, 100]) \
    .addGrid(model.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(model.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
crossval = CrossValidator(estimator=model,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

cv_model = crossval.fit(train_data)

best_model = cv_model.bestModel
# print the best model
print("Best maxIter: ", best_model._java_obj.getMaxIter())
print("Best regParam: ", best_model._java_obj.getRegParam())
print("Best elasticNetParam: ", best_model._java_obj.getElasticNetParam())
best_auc = evaluator.evaluate(best_model.transform(test_data))
print("Best ROC AUC on test data: ", best_auc)
accuracy = evaluator_accuracy.evaluate(best_model.transform(test_data))
precision = evaluator_precision.evaluate(best_model.transform(test_data))
recall = evaluator_recall.evaluate(best_model.transform(test_data))
print(f"Best model test set accuracy = {accuracy}, precision = {precision}, recall = {recall}")


'''
# There are much more data in redacted false than redacted true, so we want to test the model in the this type of data
data_test = df_parsed.filter(col("redacted") == False)
data_test = data_test.withColumn("label", lit(0))

data_test = tokenizer.transform(data_test)
data_test = hashingTF.transform(data_test)
idf_model = idf.fit(data_test)
data_test = idf_model.transform(data_test)
scaler_model = scaler.fit(data_test)
data_test = scaler_model.transform(data_test)
data_test = model.transform(data_test)

accuracy = evaluator.evaluate(data_test)
precision = evaluator_precision.evaluate(data_test)
recall = evaluator_recall.evaluate(data_test)
print(f"Full data for redacted false set accuracy = {accuracy}, precision = {precision}, recall = {recall}")
'''
# Then I want to test the model in the data in sckit-learn we found it is useless because the data is too large
