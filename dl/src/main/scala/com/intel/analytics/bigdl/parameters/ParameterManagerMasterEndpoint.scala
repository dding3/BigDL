

package org.apache.spark.sparkExtension

import java.util
import java.util.HashMap

import org.apache.spark.{SparkConf, SparkContext, SparkEnv}
import org.apache.spark.rpc._
import org.apache.spark.storage.{BlockId, BlockManagerId}
import org.apache.spark.util.{RpcUtils, Utils}
import org.apache.spark.SecurityManager

import scala.collection.mutable


case class GetExecutorBlockList(executorId: Int)
case class UpdateExecutorBlockList(executorId: Int, blockId: BlockId)
case class ClearExecutorBlockList(executorId: Int)
case class UpdateGradientPos(executorId: Int, pos: Int, length: Int)
case class SetLength(executorId: Int, length: Int)

/**
  * ParameterManagerMasterEndpoint is an [[ThreadSafeRpcEndpoint]] on the master node to track statuses
  * of all slaves' parameter managers.
  */

class ParameterManagerMasterEndpoint(
  override val rpcEnv: RpcEnv,
//  val isLocal: Boolean,
  conf: SparkConf)
//  listenerBus: LiveListenerBus)
  extends ThreadSafeRpcEndpoint {

  private val blocks = new HashMap[Int, mutable.HashSet[BlockId]]
  private val offsets = new HashMap[Int, mutable.HashSet[(Int, Int)]]
  private val remainLength = new HashMap[Int, Int]()
  
  override def receiveAndReply(context: RpcCallContext): PartialFunction[Any, Unit] = {
    case GetExecutorBlockList(executorId) =>
      context.reply(getExecutorBlockList(executorId))

    case UpdateExecutorBlockList(executorId, blockId) =>
      context.reply(updateExecutorBlockList(executorId, blockId))

    case ClearExecutorBlockList(executorId) =>
      context.reply(clearExecutorBlockList(executorId))

    case UpdateGradientPos(executorId, pos, length) =>
      context.reply(updateGradientPos(executorId, pos, length))

    case SetLength(executorId: Int, length: Int) =>
      context.reply(setLength(executorId, length))
  }

  private def getExecutorBlockList(executorId: Int): Seq[BlockId] = {
    if (blocks.containsKey(executorId)) {
      blocks.get(executorId).toSeq
    } else Seq.empty
  }

  private def updateExecutorBlockList(executorId: Int, blockId: BlockId): Unit = {
    if (blocks.containsKey(executorId)) blocks.get(executorId).add(blockId)
    else {
      val hashset = new mutable.HashSet[BlockId]()
      hashset.add(blockId)
      blocks.put(executorId, hashset)
    }
  }

  private def clearExecutorBlockList(executorId: Int) = {
    if (blocks.containsKey(executorId)) blocks.get(executorId).clear()
  }

  private def updateGradientPos(executorId: Int, pos: Int, length: Int): Int = {
    if (offsets.containsKey(executorId)) offsets.get(executorId).add(pos, length)
    val len = remainLength.get(executorId) - length
    remainLength.put(executorId, len)
    len
  }
  
  private def setLength(executorId: Int, length: Int) = {
    remainLength.put(executorId, length)
  }
}

class ParameterManagerMaster(
  var driverEndpoint: RpcEndpointRef,
  isDriver: Boolean)
{
  /** Get locations of the blockId from the driver */
  def getBlockId(executorId: Int): Seq[BlockId] = {
    val t = driverEndpoint.askWithRetry[Seq[BlockId]](GetExecutorBlockList(executorId))
//    t.foreach(println(_))
    t
  }

  /** Get locations of the blockId from the driver */
  def updateBlockId(executorId: Int, blockId: BlockId): Unit = {
//    println(s"executorId: $executorId, blockId: $blockId")
    driverEndpoint.askWithRetry[Unit](UpdateExecutorBlockList(executorId, blockId))
  }

  def clearBlockId(executorId: Int): Unit = {
    driverEndpoint.askWithRetry[Unit](ClearExecutorBlockList(executorId))
  }

  def updateGradientPos(executorId: Int, pos: Int, length: Int): Unit = {
    driverEndpoint.askWithRetry[Int](UpdateGradientPos(executorId, pos, length))
  }

  def setLength(executorId: Int, length: Int): Unit = {
    driverEndpoint.askWithRetry[Unit](SetLength(executorId, length))
  }
}

object ParameterManagerMaster {

  def createEnv(conf: SparkConf, isDriver: Boolean): ParameterManagerMaster = {
    val bindAddress = Utils.localHostName()
    var port = 7777

//    val bindAddress = "localhost" 
//    var port = 9999
        
    val systemName = if (isDriver) "BigDLDriver" else "BigDLExecutor"
    println(s"systemName: $systemName, bindAddress: $bindAddress, port: $port")
    val rpcEnv = RpcEnv.create(systemName, bindAddress, port, conf,
      new SecurityManager(conf), clientMode = !isDriver)

    def registerOrLookupEndpoint(name: String, isDriver: Boolean, endpointCreator: => RpcEndpoint):
    RpcEndpointRef = {
      if (isDriver) {
        rpcEnv.setupEndpoint(name, endpointCreator)
      } else {
        val driverHost = SparkEnv.get.blockManager.master.driverEndpoint.address.host
//val driverHost = "localhost"
//        println("hostname: " + driverHost)
        val driverPort = 7777
//val driverPort = 9999
//        println("port: " + driverPort)
        rpcEnv.setupEndpointRef(systemName, RpcAddress(driverHost, driverPort), name)
      }
    }

    new ParameterManagerMaster(registerOrLookupEndpoint(
      "ParameterManagerMaster", isDriver,
      new ParameterManagerMasterEndpoint(rpcEnv, conf)), isDriver)
  }
}

