/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include "mutlass_unit_test.h"

#include <mutlass/trace.h>

#include <mute/pointer.hpp>
#include <mute/pointer_swizzle.hpp>

namespace {

template <class T>
void test_smem_swizzle_preserves_cv()
{
  using Swizzle = mute::Swizzle<4, 4>;

  T* p = nullptr;
  auto smem_p = mute::make_smem_ptr(p);
  auto swizzle_p = mute::make_swizzle_ptr(smem_p, Swizzle{});
  auto applied_p = decltype(swizzle_p)::apply_swizzle(smem_p);
  static_assert(mute::is_same_v<decltype(applied_p), mute::smem_ptr<T*>>);

  T const* const_p = nullptr;
  auto const_smem_p = mute::make_smem_ptr(const_p);
  auto const_swizzle_p = mute::make_swizzle_ptr(const_smem_p, Swizzle{});
  auto const_applied_p = decltype(const_swizzle_p)::apply_swizzle(const_smem_p);
  static_assert(mute::is_same_v<decltype(const_applied_p), mute::smem_ptr<T const*>>);
}

}  // namespace

TEST(MuTe_core, Pointer)
{
  using namespace mute;

  MUTLASS_TRACE_HOST("-------------------------------");
  MUTLASS_TRACE_HOST("MuTe pointer wrappers");
  MUTLASS_TRACE_HOST("-------------------------------");

  // Test T* overloads (T can be nonconst or const)
  {
    using T = float;
    using expected_type = mute::gmem_ptr<T*>;
    T* p = nullptr;

    // explicit template argument
    auto gmem_p0 = mute::make_gmem_ptr<T>(p);
    static_assert(mute::is_same_v<decltype(gmem_p0), expected_type>);

    // deduced template argument
    auto gmem_p1 = mute::make_gmem_ptr(p);
    static_assert(mute::is_same_v<decltype(gmem_p1), expected_type>);
  }
  {
    using T = float const;
    using expected_type = mute::gmem_ptr<T*>;
    T* p = nullptr;

    // explicit template argument
    auto gmem_p0 = mute::make_gmem_ptr<T>(p);
    static_assert(mute::is_same_v<decltype(gmem_p0), expected_type>);

    // deduced template argument
    auto gmem_p1 = mute::make_gmem_ptr(p);
    static_assert(mute::is_same_v<decltype(gmem_p1), expected_type>);
  }

  // Test void* and void const* overloads
  // (these require an explicit template argument)
  {
    using T = float;
    using expected_type = mute::gmem_ptr<T*>;
    void* p = nullptr;

    auto gmem_p0 = mute::make_gmem_ptr<T>(p);
    static_assert(mute::is_same_v<decltype(gmem_p0), expected_type>);
  }
  {
    using T = float const;
    using expected_type = mute::gmem_ptr<T*>;
    void const* p = nullptr;

    auto gmem_p0 = mute::make_gmem_ptr<T>(p);
    static_assert(mute::is_same_v<decltype(gmem_p0), expected_type>);
  }

  // Test nullptr_t overload.
  {
    using T = float;
    using expected_type = mute::gmem_ptr<T*>;

    auto gmem_p0 = mute::make_gmem_ptr<T>(nullptr);
    static_assert(mute::is_same_v<decltype(gmem_p0), expected_type>);
  }
  {
    using T = float const;
    using expected_type = mute::gmem_ptr<T*>;

    auto gmem_p0 = mute::make_gmem_ptr<T>(nullptr);
    static_assert(mute::is_same_v<decltype(gmem_p0), expected_type>);
  }

  // Test swizzle preserves source cv-qualifiers for vectorized smem pointers.
  {
    test_smem_swizzle_preserves_cv<float>();
    test_smem_swizzle_preserves_cv<half_t>();
    test_smem_swizzle_preserves_cv<bfloat16_t>();
    test_smem_swizzle_preserves_cv<float_e4m3_t>();
    test_smem_swizzle_preserves_cv<uint_bit_t<64>>();
    test_smem_swizzle_preserves_cv<uint_bit_t<128>>();
  }
}
