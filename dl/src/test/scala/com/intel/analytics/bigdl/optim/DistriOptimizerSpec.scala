/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.parameters.ParameterManager
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

object DistriOptimizerSpec {
  val input1: Tensor[Double] = Tensor[Double](Storage[Double](Array(0.0, 1.0, 0.0, 1.0)))
  val output1 = 0.0
  val input2: Tensor[Double] = Tensor[Double](Storage[Double](Array(1.0, 0.0, 1.0, 0.0)))
  val output2 = 1.0
  var plusOne = 0.0
  val nodeNumber = 1
  val partitionNum = 4
  Engine.init(nodeNumber, partitionNum, true)

  val batchSize = 2

  val prepareData: Int => (MiniBatch[Double]) = index => {
    val input = Tensor[Double]().resize(batchSize, 4)
    val target = Tensor[Double]().resize(batchSize)
    var i = 0
    while (i < batchSize) {
      if (i % 2 == 0) {
        target.setValue(i + 1, output1 + plusOne)
        input.select(1, i + 1).copy(input1)
      } else {
        target.setValue(i + 1, output2 + plusOne)
        input.select(1, i + 1).copy(input2)
      }
      i += 1
    }
    MiniBatch(input, target)
  }
}

object DistriOptimizerSpecModel {
  def mse: Module[Double] = {
    val mlp = new Sequential[Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new Sigmoid)
    mlp.add(new Linear(2, 1))
    mlp.add(new Sigmoid)
    mlp
  }

  def cre: Module[Double] = {
    val mlp = new Sequential[Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new LogSoftMax)
    mlp
  }
}

class DistriOptimizerSpec extends FlatSpec with Matchers with BeforeAndAfter {

  import DistriOptimizerSpec._
  import DistriOptimizerSpecModel._

  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)

  var sc: SparkContext = null

  var dataSet: DistributedDataSet[MiniBatch[Double]] = null

  before {
    sc = new SparkContext("local[1]", "RDDOptimizerSpec")

    val rdd = sc.parallelize(1 to (256 * nodeNumber), partitionNum).map(prepareData)

    dataSet = new DistributedDataSet[MiniBatch[Double]] {
      override def originRDD(): RDD[_] = rdd

      override def data(train : Boolean): RDD[MiniBatch[Double]] = rdd

      override def size(): Long = 256 * nodeNumber

      override def shuffle(): Unit = {}
    }

    plusOne = 0.0
    System.setProperty("bigdl.check.singleton", false.toString)
    Engine.model.setPoolSize(1)
  }

  after {
    if (sc != null) {
      sc.stop()
      ParameterManager.clear()
    }
  }

  "Train with MSE and LBFGS" should "be good" in {
    RandomGenerator.RNG.setSeed(10)
    val optimizer = new DistriOptimizer(
      mse,
      dataSet,
      new MSECriterion[Double]())
      .setOptimMethod(new LBFGS)
    val model = optimizer.optimize()
println("model: " + model.getParameters()._1)
    val result1 = model.forward(input1).asInstanceOf[Tensor[Double]]
    result1(Array(1)) should be(0.0 +- 1e-2)

    val result2 = model.forward(input2).asInstanceOf[Tensor[Double]]
    result2(Array(1)) should be(1.0 +- 1e-2)
  }

  "Train with MSE and SGD" should "be trained with good result" in {
    val mm = mse
    mm.getParameters()._1.fill(0.125)
    val optimizer = new DistriOptimizer[Double](mm, dataSet, new MSECriterion[Double]())
      .setState(T("learningRate" -> 20.0))
      .setEndWhen(Trigger.maxEpoch(1))
    val model = optimizer.optimize()

    val result1 = model.forward(input1).asInstanceOf[Tensor[Double]]
    result1(Array(1)) should be(0.0 +- 5e-2)

    val result2 = model.forward(input2).asInstanceOf[Tensor[Double]]
    result2(Array(1)) should be(1.0 +- 5e-2)
  }

//  it should "be same compare to ref optimizer" in {
//    RandomGenerator.RNG.setSeed(10)
//    val optimizer = new DistriOptimizer(
//      mse,
//      dataSet,
//      new MSECriterion[Double]())
//    val model = optimizer.optimize()
//
//    RandomGenerator.RNG.setSeed(10)
//    val optimizerRef = new RefDistriOptimizer(
//      mse,
//      dataSet,
//      new MSECriterion[Double]()
//    )
//    val modelRef = optimizerRef.optimize()
//    
//    model.getParameters()._1 should be(modelRef.getParameters()._1)
//  }

  "An Artificial Neural Network with Cross Entropy and LBFGS" should
    "be trained with good result" in {
    plusOne = 1.0
    val optimizer = new DistriOptimizer[Double](cre, dataSet,
      new ClassNLLCriterion[Double]())
      .setEndWhen(Trigger.maxEpoch(3)).setOptimMethod(new LBFGS)
    val model = optimizer.optimize()

    val result1 = model.forward(input1).asInstanceOf[Tensor[Double]]
    result1.max(1)._2(Array(1)) should be(1.0)

    val result2 = model.forward(input2).asInstanceOf[Tensor[Double]]
    result2.max(1)._2(Array(1)) should be(2.0)
  }

  "An Artificial Neural Network with Cross Entropy and SGD" should
    "be trained with good result" in {
    plusOne = 1.0
    RandomGenerator.RNG.setSeed(10)
    val optimizer = new DistriOptimizer[Double](cre, dataSet,
      new ClassNLLCriterion[Double]())
      .setState(T("learningRate" -> 20.0))
    val model = optimizer.optimize()

    val result1 = model.forward(input1).asInstanceOf[Tensor[Double]]
    result1.max(1)._2(Array(1)) should be(1.0)

    val result2 = model.forward(input2).asInstanceOf[Tensor[Double]]
    result2.max(1)._2(Array(1)) should be(2.0)
  }

//  it should "be same compare to ref optimizer" in {
//    plusOne = 1.0
//    RandomGenerator.RNG.setSeed(10)
//    val optimizer = new DistriOptimizer[Double](
//      cre,
//      dataSet,
//      new ClassNLLCriterion[Double]()
//    ).setState(T("learningRate" -> 20.0))
//    val model = optimizer.optimize()
//
//    RandomGenerator.RNG.setSeed(10)
//    val optimizerRef = new RefDistriOptimizer(
//      cre,
//      dataSet,
//      new ClassNLLCriterion[Double]()
//    ).setState(T("learningRate" -> 20.0))
//    val modelRef = optimizerRef.optimize()
//
//    println("model:" + model.getParameters()._1)
//    model.getParameters()._1 should be(modelRef.getParameters()._1)
//
//  }
}
