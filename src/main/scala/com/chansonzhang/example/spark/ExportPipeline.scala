package com.chansonzhang.example.spark

import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport.SparkTransformerOps
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.feature.{Binarizer, StringIndexer}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, SparkSession}
import resource.managed

/**
 * Copyright 2021 Zhang, Chen. All Rights Reserved.
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @since 2021/5/31 17:20
 * @author Zhang Chen(ChansonZhang)
 */
object ExportPipeline {
  def main(args: Array[String]): Unit = {
    val datasetName = "./spark-demo.csv"
    val spark = SparkSession.builder().master("local").getOrCreate()

    val dataframe: DataFrame = spark.sqlContext.read.format("csv")
      .option("header", true)
      .load(datasetName)
      .withColumn("test_double", col("test_double").cast("double"))

    // User out-of-the-box Spark transformers like you normally would
    val stringIndexer = new StringIndexer().
      setInputCol("test_string").
      setOutputCol("test_index")

    val binarizer = new Binarizer().
      setThreshold(0.5).
      setInputCol("test_double").
      setOutputCol("test_bin")

    val pipelineEstimator = new Pipeline()
      .setStages(Array(stringIndexer, binarizer))

    val pipeline = pipelineEstimator.fit(dataframe)

    // then serialize pipeline
    val sbc = SparkBundleContext().withDataset(pipeline.transform(dataframe))
    for (bf <- managed(BundleFile("jar:file:/Users/zhangchen/code/mleap-examples/tmp/simple-spark-pipeline.zip"))) {
      pipeline.writeBundle.save(bf)(sbc).get
    }

  }
}
