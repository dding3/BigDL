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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class PairwiseDistanceSpec extends FlatSpec with Matchers {

  "hashcode()" should "behave correctly" in {
    val m1 = new PairwiseDistance[Double]()
    val m2 = new PairwiseDistance[Double]()
    val m3 = new PairwiseDistance[Double](3)
    val m4 = new PairwiseDistance[Double]()
    val log = new Log[Double]()
    com.intel.analytics.bigdl.tensor.Tensor
    val input1 = Tensor[Double](3, 3).randn()
    val input2 = Tensor[Double](3, 3).randn()
    val input = T(1 -> input1, 2 -> input2)
    m4.forward(input)


    m1.hashCode() should equal (m2.hashCode())
    m1.hashCode() should not equal null
    m1.hashCode() should not equal log.hashCode()
    m1.hashCode() should not equal m3.hashCode()
    m1.hashCode() should not equal m4.hashCode()
  }

  "equals()" should "behave correctly" in {
    val m1 = new PairwiseDistance[Double]()
    val m2 = new PairwiseDistance[Double]()
    val m3 = new PairwiseDistance[Double](3)
    val m4 = new PairwiseDistance[Double]()
    val log = new Log[Double]()
    com.intel.analytics.bigdl.tensor.Tensor
    val input1 = Tensor[Double](3, 3).randn()
    val input2 = Tensor[Double](3, 3).randn()
    val input = T(1 -> input1, 2 -> input2)
    m4.forward(input)


    m1 should equal (m2)
    m1 should not equal null
    m1 should not equal log
    m1 should not equal m3
    m1 should not equal m4
  }
}
