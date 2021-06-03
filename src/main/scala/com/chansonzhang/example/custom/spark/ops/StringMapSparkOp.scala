package com.chansonzhang.example.custom.spark.ops

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
 * @since 2021/6/3 14:07
 * @author Zhang Chen(ChansonZhang)
 */

import com.chansonzhang.example.custom.transformer.{CustomOps, StringMapModel, StringMapSparkTransformer}
import ml.combust.bundle.BundleContext
import ml.combust.bundle.dsl._
import ml.combust.bundle.op.{OpModel, OpNode}
import org.apache.spark.ml.bundle.SparkBundleContext

class StringMapSparkOp extends OpNode[SparkBundleContext, StringMapSparkTransformer, StringMapModel] {
  override val Model: OpModel[SparkBundleContext, StringMapModel] = new OpModel[SparkBundleContext, StringMapModel] {
    override val klazz: Class[StringMapModel] = classOf[StringMapModel]

    override def opName: String = CustomOps.feature.string_map

    override def store(model: Model, obj: StringMapModel)(implicit context: BundleContext[SparkBundleContext]): Model = {
      // unzip our label map so we can store the label and the value
      // as two parallel arrays, we do this because MLeap Bundles do
      // not support storing data as a map
      val (labels, values) = obj.labels.toSeq.unzip

      // add the labels and values to the Bundle model that
      // will be serialized to our MLeap bundle
      model.withValue("labels", Value.stringList(labels)).
        withValue("values", Value.doubleList(values))

    }

    override def load(model: Model)(implicit context: BundleContext[SparkBundleContext]): StringMapModel = {
      // retrieve our list of labels
      val labels = model.value("labels").getStringList

      // retrieve our list of values
      val values = model.value("values").getDoubleList

      // reconstruct the model using the parallel labels and values
      StringMapModel(labels.zip(values).toMap)

    }
  }
  override val klazz: Class[StringMapSparkTransformer] = classOf[StringMapSparkTransformer]

  override def name(node: StringMapSparkTransformer): String = node.uid

  override def model(node: StringMapSparkTransformer): StringMapModel = node.model

  override def shape(node: StringMapSparkTransformer)(implicit context: BundleContext[SparkBundleContext]): NodeShape = {
    NodeShape().withStandardIO(node.getInputCol, node.getOutputCol)
  }

  override def load(node: Node, model: StringMapModel)(implicit context: BundleContext[SparkBundleContext]): StringMapSparkTransformer = {
    StringMapSparkTransformer(uid = node.name, model = model).
      setInputCol(node.shape.standardInput.name).
      setOutputCol(node.shape.standardOutput.name)
  }
}
