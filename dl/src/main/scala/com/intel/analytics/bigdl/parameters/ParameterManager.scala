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
package com.intel.analytics.bigdl.parameters

import java.util.concurrent.{Callable, Executors, Future, ThreadFactory}

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.log4j.Logger
import org.apache.spark.sparkExtension.SparkExtension
import org.apache.spark.{SparkEnv, TaskContext}
import org.apache.spark.storage.{BlockId, BlockManagerWrapper, StorageLevel}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.reflect._

object ParameterManager {
  val logger = Logger.getLogger(getClass)

  @volatile private var pm: ParameterManager = _
  
  def get[T: ClassTag](): ParameterManager = {
    pm
  }

  def set[T: ClassTag](p : ParameterManager): Unit = {
    pm = p
  }
  
  def createParameterManager(executorId: String, executorNum: Int, size: Int):
    ParameterManager = {
    val p = new ParameterManager(executorId, executorNum, size)
    set(p)
    p
  }
}

class ParameterManager(executorId: String, executorNum: Int, size: Int) {
  import ParameterManager._

  private val syncPoolSize: Int = System.getProperty(
    "bigdl.Parameter.syncPoolSize", "4").toInt
  
  val syncPool = Executors.newFixedThreadPool(syncPoolSize, new ThreadFactory {
    override def newThread(r: Runnable): Thread = {
      val t = Executors.defaultThreadFactory().newThread(r)
      t.setDaemon(true)
      t
    }
  })

  val blockIds = new ArrayBuffer[BlockId]()

  val bm = SparkEnv.get.blockManager
  
  def getExecutorNum() : Int = {
    executorNum
  }

  def init[T: ClassTag](parameter: Tensor[T])(implicit ev: TensorNumeric[T]): Unit = {
    val _classTag = classTag[T]

    val taskSize = size / executorNum
    val extraSize = size % executorNum
    val start = executorIdToInt(executorId) * taskSize +
      math.min(executorIdToInt(executorId), extraSize)
    val length = taskSize + (if (executorIdToInt(executorId) < extraSize) 1 else 0)

    val _weights = Tensor[T](length)(_classTag, ev).copy(parameter.narrow(1,
      start + 1, length))
    BlockManagerWrapper.putSingle(getWeightPartitionId(),
      _weights, StorageLevel.MEMORY_AND_DISK, tellMaster = false)

    val weights = Tensor[T](parameter.nElement())(_classTag, ev).copy(parameter)
    BlockManagerWrapper.putSingle(getWeightId(),
      weights, StorageLevel.MEMORY_AND_DISK, tellMaster = false)

    val _gradients = Tensor[T](length)(_classTag, ev)
    BlockManagerWrapper.putSingle(getGradientPartitionId,
      _gradients, StorageLevel.MEMORY_AND_DISK, tellMaster = false)
    
    val blockId = getWeightBlockId(executorIdToInt(executorId).toString)
    val fp16param = new FP16CompressedTensor[T](length)(_classTag)
    fp16param.compress(0, parameter, start, length)
    BlockManagerWrapper.putBytes(blockId, fp16param.bytes(), StorageLevel.MEMORY_ONLY_SER)
  }
  
  def aggregateLocalGradient[T: ClassTag]() : Tensor[T] = {
    val gradientBuffer = new Array[Tensor[T]](blockIds.length)
    Engine.default.invokeAndWait2((0 until blockIds.length).map(pid => () => {
        val blockId = blockIds(pid)
        gradientBuffer(pid) =
          BlockManagerWrapper.getLocal(blockId).map(_.data.next()) match {
            case Some(x) =>
              x.asInstanceOf[Tensor[T]]
    
            case None =>
              throw new Exception("Please initialize AllReduceParameter first!!")
          }
      }))

    blockIds.clear()
    val poolSize = Engine.default.getPoolSize
    val innerTaskSize = size / poolSize
    val innerExtraSize = size % poolSize
    val availableTask = if (innerTaskSize == 0) innerExtraSize else poolSize
      Engine.default.invokeAndWait2((0 until availableTask).map(tid => () => {
      val innerStart = tid * innerTaskSize + math.min(innerExtraSize, tid)
      val innerLength = innerTaskSize + (if (tid < innerExtraSize) 1 else 0)
      var i = 1
      while (i < gradientBuffer.length) {
        gradientBuffer(0).narrow(1, innerStart + 1, innerLength)
            .add(gradientBuffer(i).narrow(1, innerStart + 1, innerLength))
        i += 1
      }
      tid
    }))
    
    gradientBuffer(0)
  }

  def putGradients[T: ClassTag](parameter: Tensor[T]): Unit = {
    var pid = 0
    val parameterBuffer = new FP16SplitsCompressedTensor[T](size,
      executorNum).asInstanceOf[CompressedTensor[T]]
    parameterBuffer.compress(parameter)
    val taskSize = size / executorNum
    val extraSize = size % executorNum
    while (pid < executorNum) {
      val start = pid * taskSize + math.min(pid, extraSize)
      val length = taskSize + (if (pid < extraSize) 1 else 0)
      val blockId = getGradientBlockId(executorIdToInt(executorId).toString, pid.toString)
      BlockManagerWrapper.putBytes(
        blockId, parameterBuffer.bytes(start, length),
        StorageLevel.MEMORY_ONLY_SER)
      pid += 1
    }
  }

  def aggregrateGradientParition1[T: ClassTag](params: Array[CompressedTensor[T]]): Unit = {
    val sgThreads = (0 until executorNum).map(pid => {
      new Callable[Int] {
        override def call(): Int = {
          try {
            val blockId = getGradientBlockId(pid.toString, executorIdToInt(executorId).toString)
            val tmp = BlockManagerWrapper.byteBufferConvert(bm.getLocalBytes(blockId)
              .getOrElse(bm.getRemoteBytes(blockId).get))
            params(pid) = SerializerInstance.serialize(tmp)
            BlockManagerWrapper.unlock(blockId)
            pid
          } catch {
            case t : Throwable =>
              logger.error("Error: " + ExceptionUtils.getStackTrace(t))
              throw t
          }
        }
      }
    })
    syncPool.invokeAll(sgThreads.asJava)
  }

  def aggregrateGradientParition2[T: ClassTag](params: Array[CompressedTensor[T]]): Unit = {
    val taskSize = size / executorNum
    val extraSize = size % executorNum
    val length = taskSize + (if (executorIdToInt(executorId) < extraSize) 1 else 0)
    val poolSize = Engine.default.getPoolSize
    val innerTaskSize = length / poolSize
    val innerExtraSize = length % poolSize
    val availableTask = if (innerTaskSize == 0) innerExtraSize else poolSize
    Engine.default.invokeAndWait2((0 until availableTask).map(tid => () => {
      val innerStart = tid * innerTaskSize + math.min(innerExtraSize, tid)
      val innerLength = innerTaskSize + (if (tid < innerExtraSize) 1 else 0)
      params.reduce((l, r) => l.add(r.bytes(innerStart, innerLength), innerStart,
        innerLength))
      tid
    }))

    val gradientId = getGradientPartitionId()
    val gradient = BlockManagerWrapper.getLocal(gradientId).map(_.data.next()) match {
      case Some(x) =>
        x.asInstanceOf[Tensor[T]]

      case None =>
        throw new Exception("Please initialize AllReduceParameter first!!")
    }
    params.head.deCompress(gradient)
    BlockManagerWrapper.removeBlock(gradientId)
    BlockManagerWrapper.putSingle(gradientId, gradient, StorageLevel.MEMORY_AND_DISK, false)
  }

  def getWeights[T: ClassTag](): Unit = {
    val weight = BlockManagerWrapper.getLocal(getWeightId)
      .map(_.data.next()) match {
      case Some(x) =>
        x.asInstanceOf[Tensor[T]]

      case None =>
        throw new Exception("Please initialize AllReduceParameter first!!")
    }
    val taskSize = size / executorNum
    val extraSize = size % executorNum
    val tasks = (0 until executorNum).map(pid => {
      new Callable[Int] {
        override def call(): Int = {
          try {
            val blockId = getWeightBlockId(pid.toString)
            val localBuffer = BlockManagerWrapper.byteBufferConvert(
              bm.getLocalBytes(blockId).getOrElse(bm.getRemoteBytes(blockId)
                .get))
            val start = pid * taskSize + math.min(pid, extraSize)
            val length = taskSize + (if (pid < extraSize) 1 else 0)
            require(localBuffer.array().length == length * 2)
            SerializerInstance.serialize(localBuffer).deCompress(0, weight, start, length)
            BlockManagerWrapper.unlock(blockId)
            pid
          } catch {
            case t : Throwable =>
              logger.error("Error: " + ExceptionUtils.getStackTrace(t))
              throw t
          }
        }
      }
    })
    syncPool.invokeAll(tasks.asJava)
    
    val weigthId = getWeightId()
    BlockManagerWrapper.removeBlock(weigthId)
    BlockManagerWrapper.putSingle(weigthId, weight, StorageLevel.MEMORY_AND_DISK, false)
  }

  def sendWeight[T: ClassTag](partitionNum: Int) : Unit = {
    val weight = BlockManagerWrapper.getLocal(getWeightPartitionId)
      .map(_.data.next()) match {
      case Some(x) =>
        x.asInstanceOf[Tensor[T]]

      case None =>
        throw new Exception("Please initialize AllReduceParameter first!!")
    }
    
    val taskSize = size / partitionNum
    val extraSize = size % partitionNum
    Engine.default.invokeAndWait2((0 until blockIds.length).map(pid => () => {
      val blockId = blockIds(pid)
      val weightPartition =
        BlockManagerWrapper.getLocal(blockId).map(_.data.next()) match {
          case Some(x) =>
            x.asInstanceOf[Tensor[T]]
  
          case None =>
            throw new Exception("Please initialize AllReduceParameter first!!")
        }
      val offset = pid * taskSize + math.min(pid, extraSize)
      val length = taskSize + (if (pid < extraSize) 1 else 0)
      weight.narrow(1, offset + 1, length).set(weightPartition)
    }))
    blockIds.clear()
    val blockId = getWeightBlockId(executorIdToInt(executorId).toString)
    BlockManagerWrapper.removeBlock(blockId)
    BlockManagerWrapper.putBytes(blockId,
      SerializerInstance.serialize(weight).bytes(), StorageLevel.MEMORY_ONLY_SER)
  }
  
  def getGradientBlockId(pidFrom : String, pidTo : String): BlockId = {
    SparkExtension.getLocalBlockId("parameterManager" + pidTo + "gradientBytes" + pidFrom)
  }

  def getWeightBlockId(pid : String): BlockId = {
    SparkExtension.getLocalBlockId("parameterManager" +
      executorIdToInt(executorId) + "weightBytes" + pid)
  }

  def getWeightPartitionId(): BlockId = {
    SparkExtension.getLocalBlockId("parameterManager weights" + executorIdToInt(executorId))
  }

  def getGradientPartitionId(): BlockId = {
    SparkExtension.getLocalBlockId("parameterManager gradients" + executorIdToInt(executorId))
  }

  def getWeightId(): BlockId = {
    SparkExtension.getLocalBlockId("parameterManager weightsTest" + executorIdToInt(executorId))
  }
  
  def executorIdToInt(executorId: String): Int = {
    var id = 0
    if (!executorId.equals("driver")) {
      println("executorid: " + executorId)
      throw new Exception("executor")
    }
      
    id
  }
}
