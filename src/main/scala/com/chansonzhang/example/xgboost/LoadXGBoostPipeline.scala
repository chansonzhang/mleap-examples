package com.chansonzhang.example.xgboost

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
import ml.combust.mleap.core.types._
import ml.combust.mleap.runtime.MleapContext
import ml.combust.mleap.runtime.MleapSupport._
import ml.combust.mleap.runtime.frame.{DefaultLeapFrame, Row}
import resource._
import org.apache.spark.ml.bundle.SparkBundleContext
import breeze.storage.Zero
import ml.dmlc.xgboost4j.scala.spark.mleap.XGBoostClassificationModelOp

object LoadXGBoostPipeline {
  implicit val sbc: MleapContext = MleapContext()

  def main(args: Array[String]): Unit = {

    // load the Spark pipeline we saved in the previous section
    val bundle = (for (bundleFile <- managed(BundleFile("jar:file:/Users/zhangchen/code/mleap-examples/tmp/xgboost-pipeline.zip")))
      yield {
        bundleFile.loadMleapBundle().get
      }).opt.get

    // create a simple LeapFrame to transform
    // MLeap makes extensive use of monadic types like Try
    val schema = StructType(
      StructField("sepallength", ScalarType.Double),
      StructField("sepalwidth", ScalarType.Double),
      StructField("petallength", ScalarType.Double),
      StructField("petalwidth", ScalarType.Double)).get
    //5.1,3.5,1.4,.2,"Setosa"
    //6.9,3.1,4.9,1.5,"Versicolor"
    //5.7,2.5,5,2,"Virginica"
    val data = Seq(
      Row(5.1, 3.5, 1.4, .2),
      Row(6.9, 3.1, 4.9, 1.5),
      Row(5.7, 2.5, 5.0, 2.0))
    val frame = DefaultLeapFrame(schema, data)

    // transform the dataframe using our pipeline
    val mleapPipeline = bundle.root
    val frame2 = mleapPipeline.transform(frame).get
    frame2.show()

  }
}
