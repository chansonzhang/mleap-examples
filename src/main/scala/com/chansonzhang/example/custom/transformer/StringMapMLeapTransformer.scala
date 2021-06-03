package com.chansonzhang.example.custom.transformer

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
 * @since 2021/6/3 11:34
 * @author Zhang Chen(ChansonZhang)
 */

import ml.combust.mleap.core.types.NodeShape
import ml.combust.mleap.runtime.function.UserDefinedFunction
import ml.combust.mleap.runtime.frame.{SimpleTransformer, Transformer}

case class StringMapMLeapTransformer(override val uid: String = Transformer.uniqueName(CustomOps.feature.string_map),
                     override val shape: NodeShape,
                     override val model: StringMapModel) extends SimpleTransformer {
  override val exec: UserDefinedFunction = (label: String) => model(label)
}

