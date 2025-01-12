from pyspark.sql import SparkSession
from pyspark.sql.functions import col,desc,split,regexp_extract, row_number, when,explode, lag, sum, avg,count, lit
from pyspark.sql.window import Window
from pyspark.ml.feature import Tokenizer,HashingTF, IDF, StandardScaler, StopWordsRemover
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

import time
import numpy as np

spark = SparkSession.builder.appName("language_use").getOrCreate()

wildchat_path = "/user/s3505235/WildChat-1M"# "../data/WildChat-1M" # full version of the data, replace it with the HDFS path
lmsys_path = "/user/s3505235/lmsys-chat-1m" # "../data/lmsys-chat-1m" # this is not neccessary, we can just use the wildchat data

df_wildchat = spark.read.parquet(wildchat_path)
df_lmsys = spark.read.parquet(lmsys_path)

df_wildchat = df_wildchat.withColumn('model_tag', regexp_extract('model', r'(gpt-\d(?:\.\d)?)', 1))

# optional: filter english only
# df_wildchat = df_wildchat.filter(col("language") == "English")

# language switches from the conversation column
df_exploded = df_wildchat.select(col("model_tag"), explode(col("conversation")).alias("utterance"))


df_parsed = df_exploded.select(
    col("utterance.turn_identifier").alias("turn_identifier"),
    col("utterance.role").alias("role"),
    col("utterance.content").alias("content"),
    col("utterance.redacted").alias("redacted")
)

# we want to use logistic regression to predict the redacted column
train_num = 5000
test_num = 1000
df_parsed = df_parsed.withColumn("content_array", split(col("content"), " ")) # only for English

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

remover = StopWordsRemover(inputCol="content_array", outputCol="clean_content")
train_data = remover.transform(train_data)
test_data = remover.transform(test_data)

input_col = "content" # "clean_content" / "content"

# tokenization
if input_col == "content":
    tokenizer = Tokenizer(inputCol=input_col, outputCol="words")
    train_data = tokenizer.transform(train_data)
    test_data = tokenizer.transform(test_data)
    hashingTF = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=10000)
else:
    hashingTF = HashingTF(inputCol=input_col, outputCol="raw_features", numFeatures=10000)

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

# train_data.show(5)
# df_filtered = df_parsed.filter(col("role") == "user")

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