from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow
import mlflow.spark

DATA_PATH = "hdfs:///data/sentiment140/training.1600000.processed.noemoticon.csv"
MODEL_PATH = "hdfs:///models/sentiment140_lr"

def main():
    spark = SparkSession.builder.appName("Sentiment140-Train").getOrCreate()

    mlflow.set_experiment("sentiment140-experiment")

    df = spark.read.csv(DATA_PATH, header=False, inferSchema=True)
    df = df.toDF("target", "ids", "date", "flag", "user", "text")
    df = df.withColumn("label", col("target"))

    df = df.withColumn("clean_text", lower(col("text")))
    df = df.withColumn("clean_text", regexp_replace(col("clean_text"), r"http\S+|www\S+", " "))
    df = df.withColumn("clean_text", regexp_replace(col("clean_text"), r"[^a-zA-Z0-9\s]", ""))

    tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    tf = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=2**18)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    lr = LogisticRegression(maxIter=30, regParam=0.01)

    pipeline = Pipeline(stages=[tokenizer, remover, tf, idf, lr])

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    with mlflow.start_run():
        model = pipeline.fit(train_df)
        pred = model.transform(test_df)

        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        acc = evaluator.evaluate(pred)
        print("ACCURACY =", acc)

        mlflow.log_metric("accuracy", acc)
        mlflow.spark.log_model(model, "sentiment_model")
        model.write().overwrite().save(MODEL_PATH)

    spark.stop()

if __name__ == "__main__":
    main()
