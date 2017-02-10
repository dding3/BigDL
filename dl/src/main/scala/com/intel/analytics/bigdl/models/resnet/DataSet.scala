/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.models.resnet

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image._
import org.apache.spark.SparkContext

trait ResNetDataSet {
  def trainDataSet(path: String, batchSize: Int, size: Int)
  : DataSet[MiniBatch[Float]]
  def valDataSet(path: String, batchSize: Int, size: Int)
  : DataSet[MiniBatch[Float]]
  def valDataSet(path: String, sc: SparkContext, imageSize: Int,
    batchSize: Int): DataSet[MiniBatch[Float]]
  def trainDataSet(path: String, sc: SparkContext, imageSize: Int,
    batchSize: Int): DataSet[MiniBatch[Float]]
}

object Cifar10DataSet extends ResNetDataSet {

  val trainMean = (0.4913996898739353, 0.4821584196221302, 0.44653092422369434)
  val trainStd = (0.24703223517429462, 0.2434851308749409, 0.26158784442034005)
  val testMean = (0.4942142913295297, 0.4851314002725445, 0.45040910258647154)
  val testStd = (0.2466525177466614, 0.2428922662655766, 0.26159238066790275)

  override def trainDataSet(path: String, batchSize: Int, size: Int)
  : DataSet[MiniBatch[Float]] = {

    DataSet.array(Utils.loadTrain(path))
      .transform(BytesToBGRImg())
      .transform(BGRImgNormalizer(trainMean, trainStd))
      .transform(HFlip(0.5))
      .transform(BGRImgRdmCropper(cropWidth = 32, cropHeight = 32, padding = 4))
      .transform(BGRImgToBatch(batchSize))
  }

  override def valDataSet(path: String, batchSize: Int, size: Int)
  : DataSet[MiniBatch[Float]] = {

    DataSet.array(Utils.loadTest(path))
      .transform(BytesToBGRImg())
      .transform(BGRImgNormalizer(testMean, testStd))
      .transform(BGRImgToBatch(batchSize))
  }

  override def valDataSet(path: String, sc: SparkContext, imageSize: Int,
    batchSize: Int)
  : DataSet[MiniBatch[Float]] = {

    DataSet.array(Utils.loadTest(path), sc)
      .transform(BytesToBGRImg())
      .transform(BGRImgNormalizer(trainMean, trainStd))
      .transform(BGRImgToBatch(batchSize))
  }

  override def trainDataSet(path: String, sc: SparkContext, imageSize: Int,
    batchSize: Int)
  : DataSet[MiniBatch[Float]] = {

    DataSet.array(Utils.loadTrain(path), sc)
      .transform(BytesToBGRImg())
      .transform(BGRImgNormalizer(testMean, testStd))
      .transform(HFlip(0.5))
      .transform(BGRImgRdmCropper(cropWidth = 32, cropHeight = 32, padding = 4))
      .transform(BGRImgToBatch(batchSize))
  }
}
