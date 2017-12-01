///*
// * Copyright 2016 The BigDL Authors.
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//
//package com.intel.analytics.bigdl.optim
//
//import com.intel.analytics.bigdl.optim.SGD.{Default, LearningRateSchedule}
//import com.intel.analytics.bigdl.tensor.Tensor
//import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
//import org.apache.log4j.Logger
//
//import scala.reflect.ClassTag
//
//class LarsSgd[T: ClassTag](
//      learningRate: Double = 1e-3,
//      learningRateDecay: Double = 0.0,
//      weightDecay: Double = 0.0,
//      momentum: Double = 0.0,
//      dampening: Double = 0.0,
//      nesterov: Boolean = false,
//      learningRateSchedule: LearningRateSchedule = Default())(
//  implicit ev: TensorNumeric[T]) extends SGD(learningRate, learningRateDecay, weightDecay,
//      momentum, dampening, nesterov, learningRateSchedule) {
//  protected var indices: Array[Int] = _
//
//  def setParameterIndices(parameters: Array[Tensor[T]]): this.type = {
//    val lengths = parameters.map(_.nElement())
//    var i = 1
//    while (i < lengths.length) {
//      lengths(i) += lengths(i - 1)
//      i += 1
//    }
//    indices = Array(0) ++ lengths
//    this
//  }
//
//  protected def findIndex(indices: Array[Int], value: Int): Int = {
//    var index = 0
//    var i = 0
//    while(i < indices.length) {
//      if (indices(i) < value) index = i
//      i += 1
//    }
//    index
//  }
//
//  protected def view(indices: Array[Int], tensor: Tensor[T]): Array[Tensor[T]] = {
//    val offset = state.getOrElse[Int]("parameterOffset", tensor.storageOffset() - 1)
//    val length = state.getOrElse[Int]("parameterLength", tensor.nElement())
//    val startIndex = findIndex(indices, offset)
//    val endIndex = findIndex(indices, offset + length)
//    val result = new Array[Tensor[T]](endIndex - startIndex + 1)
//    if (startIndex == endIndex) {
//      result(0) = tensor.narrow(1, 1, length)
//    } else {
//      result(0) = tensor.narrow(1, 1, indices(startIndex + 1) - offset)
//      var i = startIndex + 1
//      while (i < endIndex) {
//        result(i - startIndex) =
//          tensor.narrow(1, indices(i) + 1 - offset, indices(i + 1) - indices(i))
//        i += 1
//      }
//      result(i - startIndex) =
//        tensor.narrow(1, indices(endIndex) + 1 - offset, length + offset - indices(endIndex))
//    }
//    result
//  }
//
//  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]), x: Tensor[T])
//  : (Tensor[T], Array[T]) = {
//
//    this.updateHyperParameter()
//    val wd = this.weightDecay
//    val mom = this.momentum
//    val damp = this.dampening
//    val nesterov = this.nesterov
//    val clr = ev.fromType(this.learningRateSchedule.currentRate)
//
//    require(!nesterov || (mom > 0 && damp == 0),
//      "Nesterov momentum requires a momentum and zero dampening")
//
//    val (fx, dfdx) = feval(x)
//
//    val (_x, _dfdx) =
//      if (state.get[Array[Tensor[T]]]("_x").isDefined) {
//        (state.get[Array[Tensor[T]]]("_x").get,
//          state.get[Array[Tensor[T]]]("_dfdx").get
//          )
//      } else {
//        (view(indices, x), view(indices, dfdx))
//      }
//
//    var i = 0
//    while (i < _x.length) {
//      val iThX = _x(i)
//      var iThDfdx = _dfdx(i)
//      val iThXNorm = iThX.norm(2)
//      val iThDfdxNorm = iThDfdx.norm(2)
//      val iThClr = ev.times(clr, ev.divide(iThXNorm,
//        ev.plus(iThDfdxNorm, ev.times(ev.fromType(wd), iThXNorm))))
//      LarsSgd.logger.info(s"${state("parameterOffset")} $i-th clr: ${iThClr}," +
//          s" current iteration ${state("neval")}")
//
//      if (wd != 0) {
//        require(!state.get[Boolean]("isLayerwiseScaled").getOrElse(false),
//          "SGD: Can't set layerwise scale and weight decay at the same time")
//      }
//      if (wd != 0) {
//        iThDfdx.add(ev.fromType[Double](wd), iThX)
//      }
//
//      if (mom != 0) {
//        val stateDFDX = state.get[Tensor[T]](s"${i}ThDfdx") match {
//          case None =>
//            val DFDX = Tensor[T]().resizeAs(iThDfdx).copy(iThDfdx)
//            DFDX.mul(iThClr)
//            state(s"${i}ThDfdx") = DFDX
//            DFDX
//          case s: Some[Tensor[T]] => s.get.mul(ev.fromType[Double](mom)).
//            add(ev.negative(iThClr), iThDfdx)
//        }
//        iThDfdx = stateDFDX
//      }
//
//      iThX.add(ev.negative(ev.one), iThDfdx)
//
//      i += 1
//    }
//
//    (x, Array(fx))
//  }
//
//}
//
//object LarsSgd {
//  val logger = Logger.getLogger(this.getClass)
//}
