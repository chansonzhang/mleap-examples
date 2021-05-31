package com.chansonzhang.mleap.example.spark

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
 * @since 2021/5/31 17:07
 * @author Zhang Chen(ChansonZhang)
 */

import ml.combust.bundle.BundleFile
import ml.combust.mleap.runtime.MleapSupport._
import resource._
import ml.combust.mleap.runtime.frame.{DefaultLeapFrame, Row}
import ml.combust.mleap.core.types._
import ml.combust.mleap.runtime.MleapContext
import org.apache.spark.ml.bundle.SparkBundleContext
import breeze.storage.Zero

object LoadPipeline {
  implicit val sbc:MleapContext = MleapContext()
  def main(args: Array[String]): Unit = {

    // load the Spark pipeline we saved in the previous section
    val bundle = (for (bundleFile <- managed(BundleFile("jar:file:/Users/zhangchen/code/mleap-examples/tmp/simple-spark-pipeline.zip")))
      yield {
        bundleFile.loadMleapBundle().get
      }).opt.get

    // create a simple LeapFrame to transform
    // MLeap makes extensive use of monadic types like Try
    val schema = StructType(
      StructField("test_string", ScalarType.String),
      StructField("test_double", ScalarType.Double)).get
    val data = Seq(Row("hello", 0.6), Row("MLeap", 0.2))
    val frame = DefaultLeapFrame(schema, data)

    // transform the dataframe using our pipeline
    val mleapPipeline = bundle.root
    val frame2 = mleapPipeline.transform(frame).get
    val data2 = frame2.dataset

    frame2.show()

    // get data from the transformed rows and make some assertions
    assert(data2(0).getDouble(2) == 1.0) // string indexer output
    assert(data2(0).getDouble(3) == 1.0) // binarizer output

    // the second row
    assert(data2(1).getDouble(2) == 2.0)
    assert(data2(1).getDouble(3) == 0.0)

  }
}
