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

import java.util.concurrent.atomic.{AtomicInteger, AtomicLong}
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
import scala.collection.mutable.HashMap
import scala.reflect._

object ParameterManager {
  val logger = Logger.getLogger(getClass)

  private val nextId = new AtomicInteger(0)

  private var pm: ParameterManager = _

//  def get[T: ClassTag](id: Int, executorId: Int): ParameterManager = {
//    pm(id<<16 + executorId.toInt)
//  }
  def get(): ParameterManager = {
    pm
  }
  
  def createParameterManager[T: ClassTag](executorId: Int,
    executorNum: Int, partitionNum: Int, size: Int): ParameterManager = {
    val id = nextId.getAndIncrement()
    val p = new ParameterManager(id, executorId, executorNum, partitionNum, size)
    pm = p
    p
  }
  
  def clear(): Unit = {
    pm = null
  }
}

class ParameterManager(val id: Int, val executorId: Int,
  executorNum: Int, partitionNum: Int, size: Int) {
  import ParameterManager._

  private val syncPoolSize: Int = System.getProperty(
    "bigdl.Parameter.syncPoolSize", "4").toInt

  private val computePoolSize: Int = System.getProperty(
    "bigdl.Parameter.computePoolSize", "28").toInt
  
  val syncPool = Executors.newFixedThreadPool(syncPoolSize, new ThreadFactory {
    override def newThread(r: Runnable): Thread = {
      val t = Executors.defaultThreadFactory().newThread(r)
      t.setDaemon(true)
      t
    }
  })

  val computePool = Executors.newFixedThreadPool(computePoolSize, new ThreadFactory {
    override def newThread(r: Runnable): Thread = {
      val t = Executors.defaultThreadFactory().newThread(r)
      t.setDaemon(true)
      t
    }
  })
  
//  val blockIds = new ArrayBuffer[BlockId]()
  
//  @transient var executorId: Int = 0
//  @transient private var partitionId: Int = 0
  var done: Boolean = false

//  private def readObject(in: java.io.ObjectInputStream) = {
//    in.defaultReadObject()
//    executorId = executorIdToInt(SparkEnv.get.executorId)
//    partitionId = TaskContext.getPartitionId()
//    done = false
//  }

  def init[T: ClassTag](parameter: Tensor[T])(implicit ev: TensorNumeric[T]): Unit = {
    val _classTag = classTag[T]

//    println("executorid: " + executorId)
    val taskSize = size / executorNum
    val extraSize = size % executorNum
    val start = executorId * taskSize + math.min(executorId, extraSize)
    val length = taskSize + (if (executorId < extraSize) 1 else 0)
//    println("size: " + size)
//    println("executorNum: " + executorNum)
    val _weightsExecutor = Tensor[T](length)(_classTag, ev).copy(parameter.narrow(1,
      start + 1, length))
//    val _weights = Tensor[T](parameter.nElement())(_classTag, ev).copy(parameter)
    BlockManagerWrapper.putSingle(getWeightExecutorId(),
      _weightsExecutor, StorageLevel.MEMORY_AND_DISK, tellMaster = false)

    val weights = Tensor[T](parameter.nElement())(_classTag, ev).copy(parameter)
    BlockManagerWrapper.putSingle(getWeightId(),
      weights, StorageLevel.MEMORY_AND_DISK, tellMaster = false)

    val _gradientsExecutor = Tensor[T](length)(_classTag, ev)
//    val _gradients = Tensor[T](parameter.nElement())(_classTag, ev)
    BlockManagerWrapper.putSingle(getGradientExecutorId(),
      _gradientsExecutor, StorageLevel.MEMORY_AND_DISK, tellMaster = false)
    
    val blockId = getWeightBlockId(executorId)
    val fp16param = new FP16CompressedTensor[T](length)(_classTag)
    fp16param.compress(0, parameter, start, length)
    BlockManagerWrapper.putBytes(blockId, fp16param.bytes(), StorageLevel.MEMORY_ONLY_SER)
  }
  
  def aggregateLocalGradient[T: ClassTag]() : Tensor[T] = {
//    val gradientBuffer = new Array[Tensor[T]](blockIds.length)
//    Engine.default.invokeAndWait2((0 until blockIds.length).map(pid => () => {
    val gradientBuffer = new Array[Tensor[T]](finishedTaskNumber)
    val threads2 = (0 until finishedTaskNumber).map(pid => {
      new Callable[Int] {
        override def call(): Int = {
          val blockId = getGradientPartitionId(taskIds(pid))
          gradientBuffer(pid) =
            BlockManagerWrapper.getLocal(blockId).map(_.data.next()) match {
              case Some(x) =>
                x.asInstanceOf[Tensor[T]]

              case None =>
                throw new Exception("Please initialize AllReduceParameter first!!")
            }
          pid
        }
      }
    }).asJava
    computePool.invokeAll(threads2)

//    blockIds.clear()
//    val poolSize = Engine.default.getPoolSize
    val poolSize = 28
    val innerTaskSize = size / poolSize
    val innerExtraSize = size % poolSize
    val availableTask = if (innerTaskSize == 0) innerExtraSize else poolSize
    
    val threads = (0 until availableTask).map(tid => {
      new Callable[Int] {
        override def call(): Int = {
          val innerStart = tid * innerTaskSize + math.min(innerExtraSize, tid)
          val innerLength = innerTaskSize + (if (tid < innerExtraSize) 1 else 0)
          var i = 1
          while (i < gradientBuffer.length) {
            gradientBuffer(0).narrow(1, innerStart + 1, innerLength)
              .add(gradientBuffer(i).narrow(1, innerStart + 1, innerLength))
            i += 1
          }
          tid
        }
      }
    }).asJava
    computePool.invokeAll(threads)
    
    gradientBuffer(0)
  }

  def putGradients[T: ClassTag](parameter: Tensor[T]): Unit = {
//    println("executorid: " + executorId)
    var pid = 0
    val parameterBuffer = new FP16SplitsCompressedTensor[T](size,
      executorNum).asInstanceOf[CompressedTensor[T]]
    parameterBuffer.compress(parameter)
    val taskSize = size / executorNum
    val extraSize = size % executorNum
    while (pid < executorNum) {
      val start = pid * taskSize + math.min(pid, extraSize)
      val length = taskSize + (if (pid < extraSize) 1 else 0)
      val blockId = getGradientBlockId(executorId, pid)
      BlockManagerWrapper.putBytes(
        blockId, parameterBuffer.bytes(start, length),
        StorageLevel.MEMORY_ONLY_SER)
      pid += 1
    }
  }

  def aggregrateGradientParition1[T: ClassTag](params: Array[CompressedTensor[T]]): Unit = {
    val bm = SparkEnv.get.blockManager
//    println("executorid: " + executorId)
    val sgThreads = (0 until executorNum).map(pid => {
      new Callable[Int] {
        override def call(): Int = {
          try {
            val blockId = getGradientBlockId(pid, executorId)
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

//  def aggregrateGradientParition2[T: ClassTag](params: Array[CompressedTensor[T]]): Unit = {
  def aggregrateGradientParition2[T: ClassTag](params: Array[CompressedTensor[T]]): Unit = {
    val taskSize = size / executorNum
    val extraSize = size % executorNum
//    println("executorid: " + executorId)
    val length = taskSize + (if (executorId < extraSize) 1 else 0)
//    val poolSize = Engine.default.getPoolSize
val poolSize = 28
    val innerTaskSize = length / poolSize
    val innerExtraSize = length % poolSize
    val availableTask = if (innerTaskSize == 0) innerExtraSize else poolSize

    val threads = (0 until availableTask).map(tid => {
      new Callable[Int] {
        override def call(): Int = {
          val innerStart = tid * innerTaskSize + math.min(innerExtraSize, tid)
          val innerLength = innerTaskSize + (if (tid < innerExtraSize) 1 else 0)
          params.reduce((l, r) => l.add(r.bytes(innerStart, innerLength), innerStart,
            innerLength))
          tid
        }
      }
    }).asJava
    computePool.invokeAll(threads)

    val gradientId = getGradientExecutorId()
    val gradientExecutor = BlockManagerWrapper.getLocal(gradientId).map(_.data.next()) match {
      case Some(x) =>
        x.asInstanceOf[Tensor[T]]

      case None =>
        throw new Exception("Please initialize AllReduceParameter first!!")
    }
    params.head.deCompress(gradientExecutor)
//    println("gradientExecutor: " + gradientExecutor)
//    BlockManagerWrapper.removeBlock(gradientId)
//    BlockManagerWrapper.putSingle(gradientId, gradientExecutor, StorageLevel.MEMORY_AND_DISK, false)
//    BlockManagerWrapper.removeBlock(gradientId)
//    BlockManagerWrapper.putSingle(gradientId, gradient, StorageLevel.MEMORY_AND_DISK, false)
  }

  def syncWeights[T: ClassTag](localParameter: Tensor[T]): Unit = {
    val taskSize = size / executorNum
    val extraSize = size % executorNum
    val bm = SparkEnv.get.blockManager
    val tasks = (0 until executorNum).map(pid => {
      new Callable[Int] {
        override def call(): Int = {
          try {
            val blockId = getWeightBlockId(pid)
            val localBuffer = BlockManagerWrapper.byteBufferConvert(
              bm.getLocalBytes(blockId).getOrElse(bm.getRemoteBytes(blockId)
                .get))
            val start = pid * taskSize + math.min(pid, extraSize)
            val length = taskSize + (if (pid < extraSize) 1 else 0)
            require(localBuffer.array().length == length * 2)
            SerializerInstance.serialize[T](localBuffer)
              .deCompress(0, localParameter, start, length)
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
    
//    val weigthId = getWeightId()
//    BlockManagerWrapper.removeBlock(weigthId)
//    BlockManagerWrapper.putSingle(weigthId, weight, StorageLevel.MEMORY_AND_DISK, false)
  }

  def sendWeightExecutor[T: ClassTag]() : Unit = {
    val weightExecutor = BlockManagerWrapper.getLocal(getWeightExecutorId())
      .map(_.data.next()) match {
      case Some(x) =>
        x.asInstanceOf[Tensor[T]]

      case None =>
        throw new Exception("Please initialize AllReduceParameter first!!")
    }

    val size = weightExecutor.nElement()
    val taskSize = size / taskIds.size
    val extraSize = size % taskIds.size

    val threads = (0 until finishedTaskNumber).map(pid => {
      new Callable[Int] {
        override def call(): Int = {
          val blockId = getWeightPartitionId(taskIds(pid))
          val weightPartition =
            BlockManagerWrapper.getLocal(blockId).map(_.data.next()) match {
              case Some(x) =>
                x.asInstanceOf[Tensor[T]]

              case None =>
                throw new Exception("Please initialize AllReduceParameter first!!")
            }
          val offset = pid * taskSize + math.min(pid, extraSize)
          val length = taskSize + (if (pid < extraSize) 1 else 0)
          //      println("weightpartition:" + weightPartition)
          weightExecutor.narrow(1, offset + 1, length).set(weightPartition)
          pid
        }
      }
    }).asJava
    computePool.invokeAll(threads)
    
//    blockIds.clear()
//    println("weightExecutor:" + weightExecutor)
//    println("executorid: " + executorId)
    val blockId = getWeightBlockId(executorId)
    BlockManagerWrapper.removeBlock(blockId)
    BlockManagerWrapper.putBytes(blockId,
      SerializerInstance.serialize(weightExecutor).bytes(), StorageLevel.MEMORY_ONLY_SER)
  }
  
  def getGradientBlockId(pidFrom : Int, pidTo : Int): BlockId = {
    SparkExtension.getLocalBlockId("parameterManager" + pidTo + "gradientBytes" + pidFrom)
  }

  def getWeightBlockId(pid : Int): BlockId = {
    SparkExtension.getLocalBlockId("parameterManager" +
      "weightBytes" + pid)
  }

  def getWeightExecutorId(): BlockId = {
    SparkExtension.getLocalBlockId("parameterManager weights executor" + executorId)
  }

  def getWeight[T: ClassTag](): Tensor[T] = {
    BlockManagerWrapper.getLocal(getWeightId()).map(_.data.next()) match {
      case Some(x) =>
        x.asInstanceOf[Tensor[T]]

      case None =>
        throw new Exception("Please initialize AllReduceParameter first!!")
    }
  }
  
  def getWeightId(): BlockId = {
    SparkExtension.getLocalBlockId("parameterManager weights" + executorId)
  }

  def getWeightPartitionId(pid: Int): BlockId = {
    SparkExtension.getLocalBlockId("parameterManager weights partition" + pid)
  }

  def getGradientPartitionId(pid: Int): BlockId = {
    SparkExtension.getLocalBlockId("parameterManager gradient partition" + pid)
  }

  def getGradientExecutorId(): BlockId = {
    SparkExtension.getLocalBlockId("parameterManager gradients" + executorId)
  }

//  def getWeightId(): BlockId = {
//    SparkExtension.getLocalBlockId("parameterManager weightsTest" + executorId)
//  }
  
//  @transient lazy val weightExecutor: Tensor[T] = readWeightExecutor()
//  @transient lazy val gradientExecutor: Tensor[T] = readGradientExecutor()
//  @transient lazy val weightPartition: Tensor[T] = readWeightPartition()
//  @transient lazy val gradientPartition: Tensor[T] = readGradientPartition()
  def readWeightExecutor[T: ClassTag](): Tensor[T] = {
    val blockId = getWeightExecutorId()
    BlockManagerWrapper.getLocal(blockId).map(_.data.next()) match {
      case Some(x) =>
        x.asInstanceOf[Tensor[T]]

      case None =>
        throw new Exception("Please initialize AllReduceParameter first!!")
    }
  }

  def readGradientExecutor[T: ClassTag](): Tensor[T] = {
    val blockId = getGradientExecutorId()
    BlockManagerWrapper.getLocal(blockId).map(_.data.next()) match {
      case Some(x) =>
        x.asInstanceOf[Tensor[T]]

      case None =>
        throw new Exception("Please initialize AllReduceParameter first!!")
    }
  }

  def readGradientPartition[T: ClassTag](pid: Int): Tensor[T] = {
    val gradientExecutor = BlockManagerWrapper.getLocal(getGradientExecutorId())
      .map(_.data.next()) match {
      case Some(x) =>
        x.asInstanceOf[Tensor[T]]

      case None =>
        throw new Exception("Please initialize AllReduceParameter first!!")
    }
    
//    println("gradientExecutor: " + gradientExecutor)
    val size = gradientExecutor.nElement()
    val taskSize = size / taskIds.size
    val extraSize = size % taskIds.size
    val innerPid = taskIdsMap(pid)
    val offset = innerPid * taskSize + math.min(innerPid, extraSize)
    val length = taskSize + (if (innerPid < extraSize) 1 else 0)

//    println("gradientExecutor size: " + gradientExecutor.nElement())
//    taskIdsMap.foreach(x => println(x._1, x._2))
//    println("pid: " + pid + "tasksize: " + taskSize + "extrasize:" + extraSize)
    gradientExecutor.narrow(1, offset + 1, length)
  }

  def readWeightPartition[T: ClassTag](pid: Int): Tensor[T] = {
    val weightExecutor = BlockManagerWrapper.getLocal(getWeightExecutorId())
      .map(_.data.next()) match {
      case Some(x) =>
        x.asInstanceOf[Tensor[T]]

      case None =>
        throw new Exception("Please initialize AllReduceParameter first!!")
    }
    
    val size = weightExecutor.nElement()
    val taskSize = size / taskIds.size
    val extraSize = size % taskIds.size
    val innerPid = taskIdsMap(pid)
    val offset = innerPid * taskSize + math.min(innerPid, extraSize)
    val length = taskSize + (if (innerPid < extraSize) 1 else 0)

    weightExecutor.narrow(1, offset + 1, length)
  }

  def sendWeightPartition[T: ClassTag](weight: Tensor[T], pid: Int): Unit = {
    val weightsId = getWeightPartitionId(pid)
//    require(weightPartition != null)
    BlockManagerWrapper.removeBlock(weightsId)
    BlockManagerWrapper.putSingle((weightsId),
      weight, StorageLevel.MEMORY_AND_DISK, tellMaster = false)
  }

  def sendGradientPartition[T: ClassTag](gradient: Tensor[T], pid: Int): Unit = {
    val gradientsId = getGradientPartitionId(pid)
    BlockManagerWrapper.removeBlock(gradientsId)
    BlockManagerWrapper.putSingle((gradientsId),
      gradient, StorageLevel.MEMORY_AND_DISK, tellMaster = false)
  }
  
  private val taskIdsMap = new HashMap[Int, Int]()
  val taskIds = new ArrayBuffer[Int]()
  var finishedTaskNumber = 0
  var tmp = 0
  def registerPartition(taskId: Int): Unit = {
    ParameterManager.synchronized {
        taskIds.append(taskId)
        taskIdsMap(taskId) = tmp
        tmp += 1
    }
  }
  
  def clear(): Unit = {
    taskIds.clear()
    taskIdsMap.clear()
    finishedTaskNumber = 0
    done = false
  }
}
