import time
import numpy as np
from pyspark.sql.functions import col,desc,split,regexp_extract, rand, when,explode, lag, sum as spark_sum, avg,count, lit
from pyspark.sql import SparkSession
from pyspark.sql.functions import posexplode, element_at
from pyspark.ml.feature import Tokenizer,HashingTF, IDF, StandardScaler, StopWordsRemover, CountVectorizer,Word2Vec

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall

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
    # col("model_tag"),
    col("country"),
    # col("utterance.turn_identifier").alias("turn_identifier"),
    col("utterance.content").alias("content"),
    # col("utterance.role").alias("role"),
    col("utterance.toxic").alias("toxic"),
    # col("utterance.redacted").alias("redacted"),
    col("moderation.categories.harassment").alias("harassment"),
    col("moderation.categories.hate").alias("hate"),
    col("moderation.categories.`self-harm`").alias("self_harm"),
    col("moderation.categories.sexual").alias("sexual"),
    col("moderation.categories.violence").alias("violence")
)

features = ["harassment", "hate", "self_harm", "sexual", "violence"]
for feature in features:
    df_utterance = df_utterance.withColumn(
        feature, when(col(feature) == True, 1).otherwise(0)
    )

n = 1000
toxic_false_samples = df_utterance.filter(col("toxic") == False).limit(n)
harassment_samples = df_utterance.filter(col("harassment") == 1).limit(n)
hate_samples = df_utterance.filter(col("hate") == 1).limit(n)
self_harm_samples = df_utterance.filter(col("self_harm") == 1).limit(n)
sexual_samples = df_utterance.filter(col("sexual") == 1).limit(n)
violence_samples = df_utterance.filter(col("violence") == 1).limit(n)

df_utterance = toxic_false_samples \
    .union(harassment_samples) \
    .union(hate_samples) \
    .union(self_harm_samples) \
    .union(sexual_samples) \
    .union(violence_samples)

print("New DataFrame sample count:", df_utterance.count())
      
# df_utterance.show(truncate=False)
df_utterance = df_utterance.withColumn("content_array", split(col("content"), " "))
remover = StopWordsRemover(inputCol="content_array", outputCol="clean_content")
df_utterance = remover.transform(df_utterance)

df_utterance = df_utterance.sample(withReplacement=False, fraction=0.1, seed=42)
word2vec = Word2Vec(inputCol="clean_content", outputCol="features", vectorSize=100, minCount=5)
model = word2vec.fit(df_utterance)
data = model.transform(df_utterance)

# train_num = 8000
# test_num = 1000
label_cols = features
# data = data.limit(3000)
train_data, validation_data, test_data = data.randomSplit([0.8,0.1,0.1], seed=42)

def spark_df_to_tf_dataset_binary(spark_df, feature_col="features", label_cols=features, batch_size=32, shuffle=True):

    if label_cols is None:
        raise ValueError("label_cols can not be null")
    
    rdd = spark_df.rdd.map(
        lambda row: (
            row[feature_col].toArray().astype(np.float32),
            {label: np.array(row[label], dtype=np.float32) for label in label_cols}
        )
    )
    
    def generator():
        for features_np, labels in rdd.toLocalIterator():
            yield features_np, labels
    
    output_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.float32),  
        {label: tf.TensorSpec(shape=(), dtype=tf.float32) for label in label_cols}  
    )
    
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    
    ds = ds.batch(batch_size)
    
    return ds


train_ds = spark_df_to_tf_dataset_binary(train_data, batch_size=16, label_cols=label_cols)
validation_ds = spark_df_to_tf_dataset_binary(validation_data, batch_size=16, shuffle=False, label_cols=label_cols)
test_ds = spark_df_to_tf_dataset_binary(test_data, batch_size=16, shuffle=False, label_cols=label_cols)

input_dim = 100

inputs = Input(shape=(input_dim,), name="inputs")

x = Dense(256, activation='relu')(inputs)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)

outputs = {}
for label in label_cols:
    outputs[label] = Dense(1, activation='sigmoid', name=label)(x)


model = Model(inputs=inputs, outputs=outputs)


losses = {label: "binary_crossentropy" for label in label_cols}
metrics = {
    label: [
        "accuracy",
        Precision(name=f"{label}_precision"),
        Recall(name=f"{label}_recall")
    ] for label in label_cols
}

model.compile(optimizer='adam', loss=losses, metrics=metrics)
model.summary()

EPOCHS = 500

model.fit(train_ds, epochs=EPOCHS, validation_data=validation_ds)

# eval_results = model.evaluate(test_ds)

# 1.0, 1.0, 1.0, 1.0, 0.9659090638160706
eval_results = model.evaluate(test_ds, verbose=1)
for i, label in enumerate(label_cols):
    print(f"Metrics for {label}:")
    print(f"  Loss: {eval_results[i * 3]:.4f}")
    print(f"  Accuracy: {eval_results[i * 3 + 1]:.4f}")
    print(f"  Precision: {eval_results[i * 3 + 2]:.4f}")
    print(f"  Recall: {eval_results[i * 3 + 3]:.4f}")


#--------------------- toxic analysis ---------------------
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

# to be continued: train some models to predict classify fine-grained toxic categories
