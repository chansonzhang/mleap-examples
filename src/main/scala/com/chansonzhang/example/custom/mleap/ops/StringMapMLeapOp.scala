package com.chansonzhang.example.custom.mleap.ops

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
 * @since 2021/6/3 13:43
 * @author Zhang Chen(ChansonZhang)
 */

import com.chansonzhang.example.custom.transformer.{CustomOps, StringMapMLeapTransformer, StringMapModel}
import ml.combust.bundle.BundleContext
import ml.combust.bundle.dsl._
import ml.combust.bundle.op.OpModel
import ml.combust.mleap.bundle.ops.MleapOp
import ml.combust.mleap.runtime.MleapContext

class StringMapMLeapOp extends MleapOp[StringMapMLeapTransformer, StringMapModel] {
  override val Model: OpModel[MleapContext, StringMapModel] = new OpModel[MleapContext, StringMapModel] {
    override val klazz: Class[StringMapModel] = classOf[StringMapModel]

    override def opName: String = CustomOps.feature.string_map


    override def store(model: Model, obj: StringMapModel)(implicit context: BundleContext[MleapContext]): Model = {
      val (labels, values) = obj.labels.toSeq.unzip
      model.withValue("labels", Value.stringList(labels))
        .withValue("values", Value.doubleList(values))
    }

    override def load(model: Model)(implicit context: BundleContext[MleapContext]): StringMapModel = {
      val labels = model.value("labels").getStringList
      val values = model.value("values").getDoubleList

      StringMapModel(labels.zip(values).toMap)
    }
  }

  override def model(node: StringMapMLeapTransformer): StringMapModel = node.model
}
