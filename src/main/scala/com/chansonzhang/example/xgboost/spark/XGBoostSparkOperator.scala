package com.chansonzhang.example.xgboost.spark


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
 * @since 2021/6/20 15:26
 * @author Zhang Chen(ChansonZhang)
 */

import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport.SparkTransformerOps
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}
import resource.managed

object XGBoostSparkOperator {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local[3]")
      .getOrCreate()
    val irisDf = spark.read
      .option("header", value = true)
      .option("inferSchema", value = true)
      .csv("src/main/resources/iris.csv")
    println("data length: " + irisDf.count())
    irisDf.show(false)
    val stringIndexerModel = new StringIndexer()
      .setInputCol("variety")
      .setOutputCol("varietyIndex")
      .fit(irisDf)

    val Array(train, test) = stringIndexerModel.transform(irisDf)
      .randomSplit(Array(0.8, 0.2), 123456)
    println("train length: " + train.count())
    println("test length: " + test.count())

    val xgBoostModel = operator(
      train,
      Array("sepallength", "sepalwidth", "petallength", "petalwidth"),
      labelCol = "varietyIndex",
      stringIndexerModel
    )

    val predict = xgBoostModel.transform(test)
    predict.show(false)

    // then serialize pipeline
    val sbc = SparkBundleContext().withDataset(predict)
    for (bf <- managed(BundleFile("jar:file:/Users/zhangchen/code/mleap-examples/tmp/xgboost-pipeline.zip"))) {
      xgBoostModel.writeBundle.save(bf)(sbc).get
    }
  }

  def operator(df: DataFrame,
               inputCols: Array[String],
               labelCol: String,
               stringIndexerModel: StringIndexerModel
              ): PipelineModel = {
    val vectorAssembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("features")

    val xgbClassifier = new XGBoostClassifier()
      .setFeaturesCol("features").
      setLabelCol(labelCol)
      .setProbabilityCol("probability")
      .setPredictionCol("prediction")
      .setObjective("multi:softprob")
      .setMaxDepth(2)
      .setNumClass(3).
      setNumRound(6).
      setNumWorkers(3)
      .setTimeoutRequestWorkers(6000)

    val labelConvertor = new IndexToString()
      .setLabels(stringIndexerModel.labels)
      .setInputCol("prediction")
      .setOutputCol("predictVariety")

    val pipeline = new Pipeline()
      .setStages(Array(vectorAssembler, xgbClassifier, labelConvertor))

    pipeline.fit(df)
  }
}
