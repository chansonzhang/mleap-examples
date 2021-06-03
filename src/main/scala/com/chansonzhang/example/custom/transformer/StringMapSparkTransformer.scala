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
 * @since 2021/6/3 11:37
 * @author Zhang Chen(ChansonZhang)
 */

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types._

case class StringMapSparkTransformer(override val uid: String,
                                     model: StringMapModel) extends Transformer
  with HasInputCol
  with HasOutputCol {
  def this(model: StringMapModel) = this(uid = Identifiable.randomUID(CustomOps.feature.string_map), model = model)

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val stringMapUdf = udf {
      (label: String) => model(label)
    }

    dataset.withColumn($(outputCol), stringMapUdf(dataset($(inputCol))))
  }

  override def copy(extra: ParamMap): Transformer = copyValues(StringMapSparkTransformer(uid, model), extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    require(schema($(inputCol)).dataType.isInstanceOf[StringType],
      s"Input column must be of type StringType but got ${schema($(inputCol)).dataType}")
    val inputFields = schema.fields
    require(!inputFields.exists(_.name == $(outputCol)),
      s"Output column ${$(outputCol)} already exists.")

    StructType(schema.fields :+ StructField($(outputCol), DoubleType))
  }
}

