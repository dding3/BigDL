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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch, DataSet => DataSource}
import com.intel.analytics.bigdl.parameters.{CompressedTensor, ParameterManager}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import org.apache.log4j.Logger
import org.apache.spark.{SparkEnv, TaskContext}
import org.apache.spark.rdd.{RDD, ZippedPartitionsWithLocalityRDD}

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.concurrent.Future
import scala.reflect.ClassTag

object DistriOptimizer {
  import Optimizer._

  val logger = Logger.getLogger(getClass)

  /**
   * Optimizer cache some metadata on each executor
   *
   * @param localModels cached models
   * @param modelWeights weights of the cached models
   * @param modelGradients gradients of the cached models
   * @param localCriterions cached criterion
   * @param localStates cached state
   * @param gradient tensor buffer
   * @tparam T
   */
  case class Cache[T](
    localModels: Array[Module[T]],
    modelWeights: Array[Tensor[T]],
    modelGradients: Array[Tensor[T]],
    localCriterions: Array[Criterion[T]],
    localStates: Array[Table],
    gradient: Tensor[T],
    var moduleTimeList: Array[Long] = null
  )

  private[optim] def optimize[T: ClassTag](
    dataset: DistributedDataSet[MiniBatch[T]],
    state: Table,
    endWhen: Trigger,
    metrics: Metrics,
    models: RDD[Cache[T]],
    optimMethod: OptimMethod[T],
//    parameters: AllReduceParameter[T],
    parameters: ParameterManager,
    validationTrigger: Option[Trigger],
    validationDataSet: Option[DataSet[MiniBatch[T]]],
    validationMethods: Option[Array[ValidationMethod[T]]],
    cacheTrigger: Option[Trigger],
    cachePath: Option[String],
    isOverWrite: Boolean
    //    dummyData: RDD[Array[Int]]
  )(implicit ev: TensorNumeric[T]): Unit = {
    val sc = dataset.originRDD().sparkContext
    val partitionNum = dataset.originRDD().partitions.length
    var wallClockTime = 0L
    var lastEpochTime = 0L
    val driverState = T("epoch" -> state.get[Int]("epoch").getOrElse(1),
      "neval" -> state.get[Int]("neval").getOrElse(1))
    val _subModelNumber = Engine.getEngineType match {
      case MklBlas => Engine.coreNumber()
      case _ => throw new IllegalArgumentException()
    }
    var accumulateCount = 0
    val shuffleBefore = System.nanoTime()
    logger.info(s"config $state")
    logger.info(s"Shuffle data")
    dataset.shuffle()
    val shuffleEnd = System.nanoTime()
    logger.info(s"Shuffle data complete. Takes ${(shuffleEnd - shuffleBefore) / 1e9}s")

    var tasks: ArrayBuffer[Future[_]] = new ArrayBuffer()
    var threshold = Long.MaxValue
    var timeout = Long.MaxValue
    var iteration = 0
    val dropPercentage = state.get[Double]("dropPercentage").get
    val warmupIterationNum = state.get[Int]("warmupIterationNum").get
    val comupteThresholdbatchSize = state.get[Int]("comupteThresholdbatchSize").get
    val maxDropPercentage = state.get[Double]("maxDropPercentage").get
    val driverSubModelNum = partitionNum * _subModelNumber
    var dropModelNumBatch = 0
    var lossArray = new Array[Double](_subModelNumber)
        
    var epochStart = System.nanoTime()
    var dataRDD = dataset.data(train = true)
    while (!endWhen(driverState)) {
      val _header = header(driverState[Int]("epoch"), accumulateCount, dataset.size(),
        driverState[Int]("neval"), wallClockTime)
      val lossSum = sc.accumulator(0.0, "loss sum")
      val recordsNum = sc.accumulator(0, "record number")
      metrics.set("data prepare average", 0.0, sc, partitionNum)
//      metrics.set("computing time for each node", mutable.ArrayBuffer[Double](), sc)
      metrics.set("computing time average", 0.0, sc, partitionNum)
      metrics.set("aggregate gradient time", 0.0, sc, partitionNum)
      metrics.set("get weights average", 0.0, sc, partitionNum)
      metrics.set("get weights executor average", 0.0, sc, Engine.nodeNumber().get)
//      metrics.set("get weights for each node", mutable.ArrayBuffer[Double](), sc)
      metrics.set("send gradient partition", 0.0, sc, partitionNum)
      metrics.set("aggregate local gradient executor", 0.0, sc, Engine.nodeNumber().get)
      metrics.set("aggregate local gradient", 0.0, sc, partitionNum)
//      metrics.set("put gradients for each node", mutable.ArrayBuffer[Double](), sc)
//      metrics.set("put weights for each node", mutable.ArrayBuffer[Double](), sc)
      metrics.set("aggregrateGradientParition1 average executor", 0.0, sc, Engine.nodeNumber().get)
      metrics.set("aggregrateGradientParition2 average executor", 0.0, sc, Engine.nodeNumber().get)
      metrics.set("compute weight average", 0.0, sc, partitionNum)
//      metrics.set("put weights-executor average", 0.0, sc, partitionNum)
//      metrics.set("aggregate and put local gradients average", 0.0, sc, partitionNum)
      metrics.set("send weights average", 0.0, sc, partitionNum)
      
      val driverMetrics = metrics
      val start = System.nanoTime()

      // Sync weight
//      dataRDD.mapPartitions(data => {
//        val syWStart = System.nanoTime()
//        ParameterManager.get().getWeights()
//        val weightSyncTime = System.nanoTime() - syWStart
//        driverMetrics.add("get weights average", weightSyncTime)
//        //          driverMetrics.add("get weights for each node", weightSyncTime)
//        Iterator.empty
//      }).count()
      
      // Training
      val finishedModelNum = dataRDD.zipPartitions(
        models, true)(
        (data, modelIter) => {
          val cached = modelIter.next()
          val parameters = ParameterManager.get()
          val syWStart = System.nanoTime()
          ParameterManager.synchronized {
            if (!parameters.done) {
              val t = System.nanoTime()
              parameters.syncWeights(cached.modelWeights.head)
              parameters.done = true
              driverMetrics.add("get weights executor average", System.nanoTime()-t)
              println("weight after sync: " + cached.modelWeights.head)
            }
          }
          val weightSyncTime = System.nanoTime() - syWStart
          driverMetrics.add("get weights average", weightSyncTime)
//          parameters.getWeights(cached.modelWeights.head)
          val tensorConstruct = System.nanoTime()
          val tensorBuffer = new Array[(Tensor[T], Tensor[T])](_subModelNumber)
          tasks += Engine.default.invoke(() => {
            val batch = data.next()
            var b = 0
            require(batch.data.size(1) == batch.labels.size(1),
              "data and label batch size not match")
            require(batch.data.size(1) >= _subModelNumber,
              "total batch size should be divided by spark.task.cpus")
            val stackSize = batch.data.size(1) / _subModelNumber
            while (b < _subModelNumber) {
              tensorBuffer(b) = (batch.data.narrow(1, b * stackSize + 1, stackSize),
                batch.labels.narrow(1, b * stackSize + 1, stackSize))
              b += 1
            }
          })
          Engine.default.sync(tasks)
          driverMetrics.add("data prepare average", System.nanoTime() - tensorConstruct)
          
          tasks.clear()
          
          // ======================Start train models===================================
          var time = System.nanoTime()
          if(dropPercentage > 0 && iteration > warmupIterationNum + comupteThresholdbatchSize - 1) {
            timeout = threshold - weightSyncTime
//            timeout = threshold
          }
          val pre = (iteration % comupteThresholdbatchSize) * _subModelNumber
          tensorBuffer.foreach { x =>
            println("tensor buffer input: " + x._1)
            println("tensor buffer target: " + x._2)
          }
          val trainingThreads = Engine.default.invokeAndWait2((0 until _subModelNumber).map(i =>
            () => {
              val trainStart = System.nanoTime()
              val localModel = cached.localModels(i)
              localModel.training()
              val localCriterion = cached.localCriterions(i)
              val (input, target) = tensorBuffer(i)
              val output = localModel.forward(input)
              lossArray(i) = ev.toType[Double](localCriterion.forward(output, target))
              val errors = localCriterion.backward(output, target)
              localModel.backward(input, errors)
              cached.moduleTimeList(i + pre) = System.nanoTime() - trainStart + weightSyncTime
//              cached.moduleTimeList(i + pre) = System.nanoTime() - trainStart
              i
            }
          ), timeout)
          val computingTime = System.nanoTime() - time
          driverMetrics.add("computing time average", computingTime)
//          driverMetrics.add("computing time for each node", computingTime)

          time = System.nanoTime()
          val finishedThreads = trainingThreads.filter(!_.isCancelled).map(_.get())
          recordsNum += finishedThreads.size * tensorBuffer.head._2.size(1)
          var i = 0
          while (i < finishedThreads.size) {
            lossSum += lossArray(finishedThreads(i))
            i += 1
          }

          if (finishedThreads.size > 0) {
//            time = System.nanoTime()
            val gradLength = cached.modelGradients(0).nElement()
            val taskSize = gradLength / _subModelNumber
            val extraTask = gradLength % _subModelNumber

            (0 until _subModelNumber).diff(finishedThreads).foreach(i =>
              cached.modelGradients(i).zero()
            )

            // copy multi-model gradient to the buffer
            val parallelNum = if (taskSize == 0) extraTask else _subModelNumber
            Engine.default.invokeAndWait((0 until parallelNum).map(tid => () => {
              val offset = tid * taskSize + math.min(tid, extraTask)
              val length = taskSize + (if (tid < extraTask) 1 else 0)
              var i = 0
              while (i < cached.modelGradients.length) {
                if (i == 0) {
                  cached.gradient.narrow(1, offset + 1, length)
                    .copy(cached.modelGradients(i).narrow(1, offset + 1, length))
                } else {
                  cached.gradient.narrow(1, offset + 1, length)
                    .add(cached.modelGradients(i).narrow(1, offset + 1, length))
                }
                i += 1
              }
            }))
          }
          driverMetrics.add("aggregate gradient time", System.nanoTime() - time)
          
          time = System.nanoTime()
//          parameters.putGradients(cached.gradient)
          parameters.sendGradientPartition(cached.gradient, TaskContext.getPartitionId())
//          println("taskid: " + TaskContext.getPartitionId()+" table:" + cached.localStates.head)
          
          tasks ++= Engine.default.invoke((0 until _subModelNumber).map(i => () => {
            cached.localModels(i).training()
            cached.localModels(i).zeroGradParameters()
          }))
          driverMetrics.add("send gradient partition", System.nanoTime() - time)
//          println("cached.gradient: " + cached.gradient)
//println("gradients: " + cached.gradient)
//          println("taskId: " + TaskContext.getPartitionId())
          
          time = System.nanoTime()
          parameters.synchronized {
            parameters.finishedTaskNumber += 1
            if (parameters.taskIds.length == parameters.finishedTaskNumber) {
              val t = System.nanoTime()
              val gradient = parameters.aggregateLocalGradient()
//              println("gradients: " + gradient)
              parameters.putGradients(gradient)
              parameters.finishedTaskNumber = 0
              parameters.done = false
              println("pm.gradient: " + gradient)
              driverMetrics.add("aggregate local gradient executor", System.nanoTime() - t)
            }
          }
          driverMetrics.add("aggregate local gradient", System.nanoTime() - time)
          
//          driverMetrics.add("task1 time for each node", System.nanoTime() - syWStart)
          Iterator(finishedThreads.size)
        }).reduce(_ + _)

      dropModelNumBatch += (driverSubModelNum - finishedModelNum)
      if (dropPercentage == 0 || finishedModelNum >= driverSubModelNum * (1-maxDropPercentage)) {
        val value = lossSum.value / finishedModelNum

//        dummyData.mapPartitions(dataIter => {
//          val data = dataIter.next()
//          val pm = ParameterManager.get()
//
//          val getG1 = System.nanoTime()
//          val gradient = pm.aggregateLocalGradient()
//          pm.putGradients(gradient)
//          driverMetrics.add("aggregate and put local gradients average", System.nanoTime() - getG1)
//          Iterator.empty
//        }).count()
        
//        dummyData.mapPartitions(dataIter => {
//          val data = dataIter.next()
//          val pm = ParameterManager.get()
//          val params = new Array[CompressedTensor[T]](Engine.nodeNumber().get)
//          val getG1 = System.nanoTime()
//          pm.aggregrateGradientParition1(params)
//          driverMetrics.add("get gradients1 average", System.nanoTime() - getG1)
//
//          val getG2 = System.nanoTime()
//          pm.aggregrateGradientParition2(params)
//          driverMetrics.add("get gradients2 average", System.nanoTime() - getG2)
//
//          Iterator.empty
//        }).count()
        
        val nodeNumber = Engine.nodeNumber().get
        models.mapPartitions(modelIter => {
          val modelCache = modelIter.next()
          val parameters = ParameterManager.get()
//          parameters.registerPartition(TaskContext.getPartitionId())
          
          ParameterManager.synchronized {
            if (!parameters.done) {
//              val gradients = parameters.aggregateLocalGradient()
              val params = new Array[CompressedTensor[T]](nodeNumber)
              val getG1 = System.nanoTime()
              parameters.aggregrateGradientParition1(params)
              driverMetrics.add("aggregrateGradientParition1 average executor", System.nanoTime() - getG1)

              val getG2 = System.nanoTime()
              parameters.aggregrateGradientParition2(params)
              driverMetrics.add("aggregrateGradientParition2 average executor", System.nanoTime() - getG2)
              parameters.done = true
              println("gradientExecutor: " + parameters.readGradientExecutor())
              println("finishedModelNum: " + finishedModelNum + "value: " + value)
            }
          }
          
          val t = System.nanoTime()
          val taskId = TaskContext.getPartitionId
//          println("taskId: " + taskId + " modelCache.localStates: " + modelCache.localStates.head)
          val gradients = parameters.readGradientPartition[T](taskId)
          println("taskId: " + taskId + " gradientPartition: " + gradients)
          gradients.div(ev.fromType(finishedModelNum))
          println("taskId: " + taskId + " gradientPartition after: " + gradients)
          
          val weights = parameters.readWeightPartition[T](taskId)
          println("taskId: " + taskId + " weightPartition: " + parameters.readWeightPartition(taskId))
          modelCache.localStates.head("neval") = driverState[Int]("neval")
          modelCache.localStates.head("epoch") = driverState[Int]("epoch")
//          optimMethod.optimize(_ => (ev.fromType(value), parameters.gradientPartition),
//            parameters.weightPartition, modelCache.localStates.head, modelCache.localStates.head)
          optimMethod.optimize(_ => (ev.fromType(value), gradients),
            weights, modelCache.localStates.head, modelCache.localStates.head)
          driverMetrics.add("compute weight average", System.nanoTime() - t)
          
          val time = System.nanoTime()
          parameters.sendWeightPartition(weights, taskId)
          // aggreate local weights
          parameters.synchronized {
            parameters.finishedTaskNumber += 1
            if (parameters.taskIds.length == parameters.finishedTaskNumber) {
              parameters.sendWeightExecutor()
              parameters.finishedTaskNumber = 0
              parameters.done = false
              println("weightExecutor: " + parameters.readWeightExecutor())
            }
          }
          println("taskId: " + taskId + " weightPartition: " + parameters.readWeightPartition(taskId))
          driverMetrics.add("send weights average", System.nanoTime() - time)
//          driverMetrics.add("put weights for each node", System.nanoTime() - time)
          Iterator.empty
        }).count()
        
//        models.mapPartitions(iter => {
//          ParameterManager.synchronized {
//            if (!parameters.done) {
//              parameters.sendWeight(partitionNum)
//            }
//          }
//          Iterator.empty
//        }).count()
        
//        dummyData.mapPartitions(dataIter => {
//          val data = dataIter.next()
//          val pm = ParameterManager.get()
//          val time = System.nanoTime()
//          pm.sendWeight(partitionNum)
//          driverMetrics.add("put weights-executor average", System.nanoTime() - time)
//          Iterator.empty
//        }).count()
        
        accumulateCount += recordsNum.value
        val end = System.nanoTime()
        wallClockTime += end - start
        logger.info(s"${_header} Train ${recordsNum.value} in ${(end - start) / 1e9}seconds. " +
          s"Throughput is ${recordsNum.value / ((end - start) / 1e9)} records/second. Loss is ${
            lossSum.value / finishedModelNum
          }. ")

        logger.info("\n" + metrics.summary())
        logger.debug("Dropped modules: " + (driverSubModelNum - finishedModelNum))
        lossArray = new Array[Double](_subModelNumber)

        // compute threshold
        iteration += 1
        if (dropPercentage > 0 && iteration > warmupIterationNum &&
          iteration % comupteThresholdbatchSize == 0) {
          val moduleTimeList = models.mapPartitions { iter =>
            iter.next().moduleTimeList.iterator
          }.collect()

          val k = (dropPercentage * comupteThresholdbatchSize * driverSubModelNum).toInt
          if (k > dropModelNumBatch) {
            threshold = Util.kthLargest(moduleTimeList, 0, moduleTimeList.length-1,
              k - dropModelNumBatch)
          } else {
            threshold = (threshold * 1.01).toLong
          }
          logger.info("threshold: " + threshold)

          // clear moduleTimeList in each node
          models.mapPartitions { iter =>
            val timeList = iter.next.moduleTimeList
            var i = 0
            while (i < timeList.length) {
              timeList(i) = 0
              i += 1
            }
            Iterator.empty
          }.count()
          dropModelNumBatch = 0
        }

        driverState("neval") = driverState[Int]("neval") + 1
        if (accumulateCount >= dataset.size()) {
          val epochEnd = System.nanoTime()
          wallClockTime = lastEpochTime + epochEnd - epochStart
          lastEpochTime = wallClockTime
          epochStart = System.nanoTime()
          logger.info(s"${_header} Epoch finished. Wall clock time is ${wallClockTime / 1e6}ms")

          driverState("epoch") = driverState[Int]("epoch") + 1
          dataset.shuffle()
          dataRDD = dataset.data(train = true)
          accumulateCount = 0
        }
        validate(
          validationTrigger,
          validationDataSet,
          validationMethods,
          models,
          wallClockTime,
          driverState
        )

        checkpoint(
          cacheTrigger,
          cachePath,
          isOverWrite,
          wallClockTime,
          models,
          driverState,
          parameters
        )
      } else {
        logger.info(s"Warning!!! Ignore this iteration as more than maxDropPercentage " +
          s"module is dropped!! Finished modules number: ${finishedModelNum}")
      }
    }

//    models.mapPartitions { iter =>
//      println("weights:" + iter.next.modelWeights.head)
//      Iterator.empty
//    }.count()
  }


  private def checkpoint[T: ClassTag](
    cacheTrigger: Option[Trigger],
    cachePath: Option[String],
    isOverWrite: Boolean,
    wallClockTime: Long,
    models: RDD[Cache[T]],
    state: Table,
//    parameters: AllReduceParameter[T])
    parameters: ParameterManager)
  : Unit = {
    if (cacheTrigger.isDefined) {
      val trigger = cacheTrigger.get
      if (trigger(state) && cachePath.isDefined) {
        println(s"[Wall Clock ${wallClockTime / 1e9}s] Save model to ${cachePath.get}")
        saveModel(getModel(models, parameters), cachePath, isOverWrite,
          s".${state[Int]("neval")}")
        saveState(models.map(_.localStates.head).first(), cachePath, isOverWrite, s"" +
          s".${state[Int]("neval")}")
      }
    }
  }

  private def initThreadModels[T: ClassTag](
    model: Module[T],
    dataset: DistributedDataSet[MiniBatch[T]],
    criterion: Criterion[T],
    state: Table,
    nodeNumber: Int,
    checkSingleton: Boolean,
    pm: ParameterManager
//    parameters: AllReduceParameter[T],
//    dummyData: RDD[Array[Int]]
    )(implicit ev: TensorNumeric[T]) = {
    val sc = dataset.originRDD().sparkContext
    val _subModelNumber = Engine.getEngineType match {
      case MklBlas => Engine.coreNumber()
      case _ => throw new IllegalArgumentException
    }
    val partitionNum = dataset.originRDD().partitions.length
    val comupteThresholdbatchSize = state.get[Int]("comupteThresholdbatchSize").get
    val parameterSize = model.getParameters()._1.nElement()

    val executorIdList = dataset.originRDD().mapPartitions { iter =>
      ParameterManager.synchronized {
          Iterator(SparkEnv.get.executorId)
      }
    }.collect().distinct
    
    require(executorIdList.size == nodeNumber)
    val executorIdMap = new HashMap[String, Int]()
    var i = 0
    while (i < nodeNumber) {
      executorIdMap(executorIdList(i)) = i
      i += 1
    }
    
    dataset.originRDD().mapPartitions { iter =>
      ParameterManager.synchronized {
        if (ParameterManager.get() == null) {
          val executorId = SparkEnv.get.executorId
          ParameterManager.createParameterManager(executorIdMap(executorId), nodeNumber,
            partitionNum, parameterSize)
        }
        val parameter = ParameterManager.get()
        parameter.registerPartition(TaskContext.getPartitionId())
        Iterator.empty
      }
    }.count()
        
    val taskDistribution = dataset.originRDD().mapPartitions { iter =>
      val parameter = ParameterManager.get()
//      println(s"executorid: ${parameter.executorId} size: ${parameter.taskIds.size}")
      Iterator.single(Map(parameter.executorId ->
        parameter.taskIds.size))
    }.reduce(_ ++ _)

//    taskDistribution.foreach(x => println(s"taskdistribution: key: ${x._1} value: ${x._2}"))
    
    val taskSize = parameterSize / partitionNum
    val remainSize = (parameterSize - taskSize * partitionNum) / nodeNumber
    val extraSize = (parameterSize - taskSize * partitionNum) % nodeNumber
//println(s"tasksize: $taskSize remainsize: $remainSize extrasize: $extraSize")
    val sizeMap = new HashMap[Int, (Int, Int)]()
    var index = 0
    var start = 0
    var length = 0
      while (index < nodeNumber) {
//        println("index: " + index)
        length = taskSize * (taskDistribution(index)) + remainSize + (if (index < extraSize) 1 else 0)
        sizeMap.put(index, (start, length))
        start += length
        index += 1
      }
    
//    sizeMap.foreach(x => println(s"key: ${x._1} value: ${x._2}"))
    
    
    val broadcast = sc.broadcast((model, criterion, state))
    val models = dataset.originRDD().mapPartitions(_ => {
      val (broadcastModel, broadcastCriterion, broadcastState)
         = broadcast.value
      if (!Engine.checkSingleton()) {
        if (checkSingleton) {
          require(Engine.checkSingleton(), "Detect multi-task run on one Executor/Container. " +
            "Please disable singleton check or repartition data")
        } else {
          logger.warn("Detect multi-task run on one Executor/Container.")
        }
      }
      val cached = (0 until _subModelNumber).map { _ =>
        val localModel = broadcastModel.cloneModule()
        val localCriterion = broadcastCriterion.cloneCriterion()
        val localState = broadcastState.clone()
        val (weights, grads) = localModel.getParameters()
        (localModel, weights, grads, localCriterion, localState)
      }.toArray

//      val weights = cached.head._2
//      cached.map(c =>
//        if (!c._2.eq(weights)) {
//          c._2.storage().set(weights.storage())
//        }
//      )

      logger.info("model thread pool size is " + Engine.model.getPoolSize)
//      parameters.init(weights)

      // init parameter manager
      val weights = cached.head._2
      val parameter = ParameterManager.get()
      ParameterManager.synchronized {
        if (!parameter.done) {
//          val executorId = SparkEnv.get.executorId
//          val id = if (!executorId.equals("driver")) executorId.toInt else 0
//          val pm2 = ParameterManager.createParameterManager(id, nodeNumber,
//            partitionNum, weights.nElement)
          parameter.init(weights, sizeMap)
          parameter.done = true
        }
      }
      
      cached.map(c =>
        c._2.storage().set(parameter.getWeight[T]().storage())
      )
      
      ParameterManager.synchronized {
        parameter.finishedTaskNumber += 1
        if (parameter.finishedTaskNumber == parameter.taskIds.size) {
          parameter.finishedTaskNumber = 0
          parameter.done = false
        }
      }
      
      Iterator(Cache(
        cached.map(_._1), // models
        cached.map(_._2), // weights
        cached.map(_._3), // gradients
        cached.map(_._4), // criterions
        cached.map(_._5), // states
        cached.head._2.clone(), // a tensor buffer
        new Array[Long](_subModelNumber * comupteThresholdbatchSize)
      ))
    }).persist()
    models.setName("Thread Model RDD")
    logger.info("Cache thread models...")
    models.count()
    logger.info("Cache thread models... done")
    models
  }


  private def validate[T](
    validationTrigger: Option[Trigger],
    validationDataSet: Option[DataSet[MiniBatch[T]]],
    validationMethods: Option[Array[ValidationMethod[T]]],
    models: RDD[Cache[T]],
    wallClockTime: Long,
    state: Table
  ): Unit = {
    if (validationTrigger.isEmpty || validationDataSet.isEmpty) {
      return
    }
    val trigger = validationTrigger.get
    if (!trigger(state)) {
      return
    }
    val vMethods = validationMethods.get
    val validateRDD = validationDataSet.get.toDistributed().data(train = false)
    logger.info(s"[Wall Clock ${wallClockTime / 1e9}s] Validate model...")
    val _subModelNumber = Engine.getEngineType match {
      case MklBlas => Engine.coreNumber()
      case _ => throw new IllegalArgumentException
    }
    ZippedPartitionsWithLocalityRDD(models, validateRDD)((modelIter, dataIter) => {
      val cached = modelIter.next()
      val criterion = cached.localCriterions
      val workingModels = cached.localModels

      workingModels.foreach(_.evaluate())
      dataIter.map(batch => {
        require(batch.data.size(1) == batch.labels.size(1))
        val stackSize = batch.data.size(1) / _subModelNumber
        val extraSize = batch.data.size(1) % _subModelNumber
        val parallelism = if (stackSize == 0) extraSize else _subModelNumber
        Engine.default.invokeAndWait(
          (0 until parallelism).map(b =>
            () => {
              val offset = b * stackSize + math.min(b, extraSize)
              val length = stackSize + (if (b < extraSize) 1 else 0)
              val input = batch.data.narrow(1, offset + 1, length)
              val target = batch.labels.narrow(1, offset + 1, length)
              val output = workingModels(b).forward(input)
              vMethods.map(validation => {
                validation(output, target, criterion(b))
              })
            }
          )
        ).reduce((left, right) => {
          left.zip(right).map { case (l, r) =>
            l + r
          }
        })
      })
    }).reduce((left, right) => {
      left.zip(right).map { case (l, r) =>
        l + r
      }
    }).zip(vMethods).foreach(r => {
      logger.info(s"${r._2} is ${r._1}")
    })
  }


  private def getModel[T: ClassTag](models: RDD[Cache[T]],
//      parameters: AllReduceParameter[T])
    parameters: ParameterManager)
  : Module[T] = {
    val partitionNum = models.partitions.length
    val trainedModel = models.map(_.localModels.head.clearState()).first()
    val weights = models.mapPartitions(iter => {
      val cached = iter.next()
//      val curPartitionId = TaskContext.getPartitionId()
//      Iterator.single(Map(curPartitionId -> parameters.weightPartition))
//      Iterator.single(Map(curPartitionId ->
//        ParameterManager.get().readWeightPartition[T](curPartitionId)))
      val parameter = ParameterManager.get()
      Iterator.single(Map(parameter.executorId ->
        parameter.readWeightExecutor[T]()))
    }).reduce(_ ++ _)
    
    val sizeMap = models.mapPartitions(iter => {
      Iterator.single(ParameterManager.get().sizeMap)
    }).first()

    val parameter = trainedModel.getParameters()._1
    val parameterLength = parameter.nElement()
//    val taskSize = parameterLength / partitionNum
//    require(taskSize != 0, "parameter length should not less than partition number")
//    val extraSize = parameterLength % partitionNum

//    (0 until partitionNum).map(pid => {
//      val start = pid * taskSize + math.min(pid, extraSize)
//      val length = taskSize + (if (pid < extraSize) 1 else 0)
//      parameter.narrow(1, start + 1, length).copy(weights(pid))
//    })

//    println("weightsize: " + weights.size)
//    println("parameterlen: " + parameterLength)
//    val taskSize = parameterLength / weights.size
//    val extraSize = parameterLength % weights.size

    (0 until weights.size).map(pid => {
//      val start = pid * taskSize + math.min(pid, extraSize)
//      val length = taskSize + (if (pid < extraSize) 1 else 0)
      val start = sizeMap(pid)._1
      val length = sizeMap(pid)._2
      parameter.narrow(1, start + 1, length).copy(weights(pid))
    })

    trainedModel
  }
}

//class DistriOptimizer[T: ClassTag] private[optim](
class DistriOptimizer[T: ClassTag](
  model: Module[T],
  dataset: DistributedDataSet[MiniBatch[T]],
  criterion: Criterion[T]
)(implicit ev: TensorNumeric[T])
  extends Optimizer[T, MiniBatch[T]](
    model, dataset, criterion) {

  val metrics = new Metrics

  private var models: RDD[DistriOptimizer.Cache[T]] = null

  override def optimize(): Module[T] = {
    this.assertEngineInited()

    optimMethod.clearHistory(state)
    state("dropPercentage") = dropPercentage
    state("warmupIterationNum") = warmupIterationNum
    state("comupteThresholdbatchSize") = comupteThresholdbatchSize
    state("maxDropPercentage") = maxDropPercentage

//    require(Engine.nodeNumber().isDefined, "Node number is not set")
//    val nodeNumber = Engine.nodeNumber().get

    val partitionNum = dataset.originRDD().partitions.length
    val size = model.getParameters()._1.nElement()
//    val parameters = AllReduceParameter.newParameter(partitionNum, size)
//    val parameters = ParameterManager.createParameterManager(nodeNumber, partitionNum, size)
val parameters: ParameterManager = null
//    val dummyData = dataset.originRDD().sparkContext
//      .parallelize(0 to 100000, nodeNumber * Engine.coreNumber)
//      // Keep this line, or the array will be send to worker every time
//      .coalesce(nodeNumber, true)
//      .mapPartitions(iter => {
//        Iterator.single(iter.toArray)
//      }).setName("dummyData dataset").cache()
//    dummyData.count()
//
    val actualNodeNumber = dataset.originRDD().mapPartitions { iter =>
        Iterator(SparkEnv.get.executorId)
      }.collect().distinct.size
    Engine.setNodeNumber(Some(actualNodeNumber))
    dataset.originRDD().sparkContext.getConf.setExecutorEnv("DL_NODE_NUMBER", actualNodeNumber.toString)
    
    models = DistriOptimizer.initThreadModels(model, dataset, criterion, state,
      actualNodeNumber, checkSingleton, parameters)

    DistriOptimizer.optimize(
      dataset,
      state,
      endWhen,
      metrics,
      models,
      optimMethod,
      parameters,
      validationTrigger,
      validationDataSet,
      validationMethods,
      checkpointTrigger,
      checkpointPath,
      isOverWrite
    )

    DistriOptimizer.getModel(models, parameters)
  }
}


