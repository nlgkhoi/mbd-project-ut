from pyspark.sql import SparkSession

# Initialize a Spark session
spark = SparkSession.builder \
    .appName("Word Count Example") \
    .getOrCreate()

# Create an RDD from a text file
text_file = spark.sparkContext.textFile("example.txt")

# Perform a word count
word_counts = text_file.flatMap(lambda line: line.split(" ")) \
    .map(lambda word: (word, 1)) \
    .reduceByKey(lambda a, b: a + b)

# Print the word count results
for word, count in word_counts.collect():
    print(f"{word}: {count}")

# Stop the Spark session
spark.stop()
