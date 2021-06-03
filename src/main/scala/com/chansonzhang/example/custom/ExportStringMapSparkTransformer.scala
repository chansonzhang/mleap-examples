package com.chansonzhang.example.custom

import com.chansonzhang.example.custom.transformer.{StringMapModel, StringMapSparkTransformer}
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
 * @since 2021/5/31 17:20
 * @author Zhang Chen(ChansonZhang)
 */
object ExportStringMapSparkTransformer {
  def main(args: Array[String]): Unit = {
    val model = StringMapModel(Map[String, Double]("A" -> 1.0, "B" -> 2.0))
    val stringMapSparkTransformer = StringMapSparkTransformer("StringMapSparkTransformer",model)
    stringMapSparkTransformer.setInputCol("input")
    stringMapSparkTransformer.setOutputCol("output")

    // then serialize transformer
    for (bf <- managed(BundleFile("jar:file:/Users/zhangchen/code/mleap-examples/tmp/simple-spark-transformer.zip"))) {
      stringMapSparkTransformer.writeBundle
        .save(bf).get
    }

  }
}
