"""Optional: the cleaning + language-id step done in Spark / Spark NLP, for when the raw
data no longer fits comfortably in pandas.

NOTE: I have not been able to run this one against a real Spark cluster in the dev
environment - treat it as a starting point and check the Spark NLP model names against
https://sparknlp.org/models before running.

    pip install pyspark==3.5.1 spark-nlp==5.3.3
    python -m socintel.spark_stage data/processed/00_normalized.jsonl data/processed/spark_clean
"""

import sys


def run(inp, out):
    import sparknlp
    from pyspark.ml import Pipeline
    from pyspark.sql import functions as F
    from sparknlp.annotator import LanguageDetectorDL
    from sparknlp.base import DocumentAssembler

    spark = sparknlp.start()
    df = spark.read.json(inp)

    # regex clean-up in pure Spark SQL
    df = df.withColumn("clean_text", F.col("text"))
    df = df.withColumn("clean_text", F.regexp_replace("clean_text", r"https?://\S+|www\.\S+", " "))
    df = df.withColumn("clean_text", F.regexp_replace("clean_text", r"(?<!\w)(u/|r/|@)\w+", " @user "))
    df = df.withColumn("clean_text", F.regexp_replace("clean_text", r"\s+", " "))
    df = df.withColumn("clean_text", F.trim("clean_text")).filter(F.length("clean_text") >= 3)

    doc = DocumentAssembler().setInputCol("clean_text").setOutputCol("document")
    lang = (LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_21", "xx")
            .setInputCols(["document"]).setOutputCol("lang").setThreshold(0.5).setCoalesceSentences(True))
    pipe = Pipeline(stages=[doc, lang])
    res = pipe.fit(df).transform(df)
    res = res.withColumn("language", F.col("lang.result")[0]).drop("document", "lang")
    res.write.mode("overwrite").json(out)
    print("rows:", res.count())


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
