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
package com.intel.analytics.bigdl.models.inception

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, T, Table}
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object TrainInceptionV1 {
  LoggerFilter.redirectSparkInfoLogs()
//  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.DEBUG)
  val logger = Logger.getLogger(getClass)

  import Options._

  def decay(epoch: Int): Double = {
    if (epoch > 30) {
      1
    } else if (epoch > 60) {
      2
    } else if (epoch > 80) {
      3
    } else {
      0.0
    }
  }

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      val imageSize = 224
      val conf = Engine.createSparkConf()
        .setAppName(s"BigDL InceptionV1 Train Example batchSize" +
          s" ${param.batchSize} with global gradientClip${param.gradientClipMax}")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      val trainSet = ImageNet2012(
        param.folder + "/train",
        sc,
        imageSize,
        param.batchSize,
        Engine.nodeNumber(),
        Engine.coreNumber(),
        param.classNumber,
        1281167
      )
      val valSet = ImageNet2012Val(
        param.folder + "/val",
        sc,
        imageSize,
        param.batchSize,
        Engine.nodeNumber(),
        Engine.coreNumber(),
        param.classNumber,
        50000
      )

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else if (param.graphModel) {
        Inception_v1_NoAuxClassifier.graph(classNum = param.classNumber)
      } else {
        Inception_v1_NoAuxClassifier(classNum = param.classNumber)
      }

      val baseLr = param.learningRate

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else if (param.maxEpoch.isDefined) {
        val iterationsPerEpoch = math.ceil(1281167.toFloat / param.batchSize).toInt
        val warmUpIteration = iterationsPerEpoch * param.warmUpEpoch
        val maxLr = param.maxLr
        val delta = (maxLr - baseLr) / warmUpIteration

        logger.info(s"warmUpIteraion: $warmUpIteration, startLr: ${param.learningRate}, " +
          s"maxLr: $maxLr, " +
          s"delta: $delta, nesterov: ${param.nesterov}")

        val sgd = new SGD[Float](learningRate = param.learningRate,
          learningRateDecay = 0.0,
          weightDecay = param.weightDecay, momentum = 0.9,
          dampening = 0.0, nesterov = param.nesterov,
          learningRateSchedule =
 //           SGD.EpochDecayWithWarmUp(warmUpIteration, delta, decay))
         SGD.PolyWithWarmUp(warmUpIteration, delta,
           0.5, math.ceil(1281167.toDouble / param.batchSize).toInt * param.maxEpoch.get),
          gradientClipMax = param.gradientClipMax
//          gradientMax = param.gradientMax,
//          gradientMin = param.gradientMin
        )
        if (param.resumeEpoch.isDefined) {
          val resumeEpoch = param.resumeEpoch.get
          val neval = (resumeEpoch - 1) * iterationsPerEpoch + 1
          sgd.setState(T(
            "epoch" -> resumeEpoch,
            "neval" -> neval,
            "evalCounter" -> (neval - 1)
          ))
        }
        sgd
      } else {
        new SGD[Float](learningRate = param.learningRate, learningRateDecay = 0.0,
          weightDecay = param.weightDecay, momentum = 0.9,
          dampening = 0.0, nesterov = param.nesterov,
          learningRateSchedule = SGD.Poly(0.5, param.maxIteration))
      }

      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = new ClassNLLCriterion[Float]()
      )

      val (checkpointTrigger, testTrigger, endTrigger) = if (param.maxEpoch.isDefined) {
        (Trigger.everyEpoch, Trigger.everyEpoch, Trigger.maxEpoch(param.maxEpoch.get))
      } else {
        (
          Trigger.severalIteration(param.checkpointIteration),
          Trigger.severalIteration(param.checkpointIteration),
          Trigger.maxIteration(param.maxIteration)
          )
      }

      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, checkpointTrigger)
      }

      if (param.overWriteCheckpoint) {
        optimizer.overWriteCheckpoint()
      }

      val logdir = "imagenet"
      val appName = s"${sc.applicationId}"
//      val trainSummary = TrainSummary(logdir, appName)
//      trainSummary.setSummaryTrigger("LearningRate", Trigger.severalIteration(1))
//      trainSummary.setSummaryTrigger("gradientNorm2", Trigger.severalIteration(1))
//      trainSummary.setSummaryTrigger("Parameters", Trigger.severalIteration(10))
//      val validationSummary = ValidationSummary(logdir, appName)

      optimizer
        .setOptimMethod(optimMethod)
//        .setTrainSummary(trainSummary)
//        .setValidationSummary(validationSummary)
        .setValidation(testTrigger,
          valSet, Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
        .setEndWhen(endTrigger)
        .optimize()
      sc.stop()
    })
  }
}
