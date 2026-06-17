/*
 * Copyright (c) 2017-2021, 2023-2026 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ACL_SRC_CPU_OPERATORS_CPUCONV2D_H
#define ACL_SRC_CPU_OPERATORS_CPUCONV2D_H

#include "arm_compute/function_info/ActivationLayerInfo.h"

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuOperator.h"

namespace arm_compute
{
namespace cpu
{
/** Basic function to simulate a convolution layer. This function calls one of the following functions:
 * -# @ref CpuGemm     (executed only in case GEMM is required for the operation)
 * -# @ref CpuWinogradConv2d (executed only in case Winograd is required for the operation)
 * -# @ref CpuDirectConv2d   (executed only in case Direct Convolution is required for the operation)
 *
 *
 * The function selects one of the algorithms mentioned above based on:
 *      - The size of the kernel
 *      - Number of input/output feature maps
 *      - Amount of memory needed
 *
 * Generally GEMM-based convolution is executed when neither Winograd nor FFT nor Direct convolution can be performed.
 *
 * FP32 Algorithm| Filter Size                                        |   Input/Output feature maps               |
 * --------------|----------------------------------------------------|-------------------------------------------|
 * Winograd      | 3x3 1x3 3x1 5x1 1x5 5x5(fast maths) 7x1 1x7        |  Input channels is greater than 3         |
 * FFT           | Squared kernels and greater than 9x9               |  Input feature maps > Output feature maps |
 * DirectConv    | 9x9                                                |                                           |
 * GEMM          | Any size                                           |                                           |
 *
 * Winograd 5x5 requires fast maths enabled.
 *
 * FP16 Algorithm| Filter Size      |
 * --------------|------------------|
 * Winograd      | Not supported    |
 * FFT           | Not supported    |
 * DirectConv    | 9x9              |
 * GEMM          | Any size         |
 *
 *
 */
class CpuConv2d : public ICpuOperator
{
public:
    /** Constructor */
    CpuConv2d();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuConv2d);
    /** Default destructor */
    ~CpuConv2d();
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src0           |src1               |src2   |dst            |
     * |:--------------|:------------------|:------|:--------------|
     * |F16            |F16                |F16    |F16            |
     * |F32            |F32                |F32    |F32            |
     * |QASYMM8        |QASYMM8            |S32    |QASYMM8        |
     * |QASYMM8        |QASYMM8            |F32    |F32            |
     * |QASYMM8        |QASYMM8_SIGNED     |S32    |QASYMM8        |
     * |QASYMM8        |QSYMM8_PER_CHANNEL |S32    |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32    |QASYMM8_SIGNED |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED     |F32    |F32            |
     * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32    |QASYMM8_SIGNED |
     *
     * The QASYMM8_SIGNED→F32 row (F32 bias, F32 dst) is only supported when @p use_direct_i8_s8_f32 is true.
     * It requires NHWC layout and no dilation, and uses the single-kernel CpuGemmDirectConv2d path.
     *
     * The QASYMM8→F32 row (F32 bias, F32 dst) is only supported when @p use_direct_u8_u8_f32 is true.
     * It requires NHWC layout and no dilation, and uses the single-kernel CpuGemmDirectConv2d path.
     *
     * @param[in]  src                   Source tensor info.
     * @param[in]  weights               Weights tensor info.
     * @param[in]  biases                Biases tensor info.
     * @param[out] dst                   Destination tensor info.
     * @param[in]  conv_info             Contains padding and stride information.
     * @param[in]  weights_info         Weights reshape info.
     * @param[in]  dilation             (Optional) Dilation. Defaults to (1, 1).
     * @param[in]  act_info             (Optional) Fused activation.
     * @param[in]  enable_fast_math     (Optional) Enable fast math. Default is false.
     * @param[in]  num_groups           (Optional) Number of groups. num_groups != 1 is not supported.
     * @param[in]  use_direct_i8_s8_f32 (Optional) When true and input is QASYMM8_SIGNED with F32 output,
     *                                   route via the single-kernel direct-conv path (NHWC, no dilation). Default is false.
     * @param[in]  use_direct_u8_u8_f32 (Optional) When true and input is QASYMM8 with F32 output,
     *                                   route via the single-kernel direct-conv path (NHWC, no dilation). Default is false.
     */
    void configure(ITensorInfo               *src,
                   ITensorInfo               *weights,
                   const ITensorInfo         *biases,
                   ITensorInfo               *dst,
                   const PadStrideInfo       &conv_info,
                   const WeightsInfo         &weights_info         = WeightsInfo(),
                   const Size2D              &dilation             = Size2D(1U, 1U),
                   const ActivationLayerInfo &act_info             = ActivationLayerInfo(),
                   bool                       enable_fast_math     = false,
                   unsigned int               num_groups           = 1,
                   bool                       use_direct_i8_s8_f32 = false,
                   bool                       use_direct_u8_u8_f32 = false);
    /** Static function to check if given info will lead to a valid configuration of @ref CpuConv2d
     *
     * Similar to CpuConv2d::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo         *src,
                           const ITensorInfo         *weights,
                           const ITensorInfo         *biases,
                           const ITensorInfo         *output,
                           const PadStrideInfo       &conv_info,
                           const WeightsInfo         &weights_info         = WeightsInfo(),
                           const Size2D              &dilation             = Size2D(1U, 1U),
                           const ActivationLayerInfo &act_info             = ActivationLayerInfo(),
                           bool                       enable_fast_math     = false,
                           unsigned int               num_groups           = 1,
                           bool                       use_direct_i8_s8_f32 = false,
                           bool                       use_direct_u8_u8_f32 = false);
    /** Static function to check if given info will return the convolution called by @ref CpuConv2d
     *
     * @param[in] src                   Source tensor info.
     * @param[in] weights               Weights tensor info.
     * @param[in] dst                   Destination tensor info.
     * @param[in] conv_info             Contains padding and stride information.
     * @param[in] weights_info         Weights reshape info.
     * @param[in] dilation             (Optional) Dilation. Defaults to (1, 1).
     * @param[in] act_info             (Optional) Fused activation.
     * @param[in] enable_fast_math     (Optional) Enable fast math. Default is false.
     * @param[in] use_direct_i8_s8_f32 (Optional) Force GEMM_CONV2D for QASYMM8_SIGNED→F32. Default is false.
     * @param[in] use_direct_u8_u8_f32 (Optional) Force GEMM_CONV2D for QASYMM8→F32. Default is false.
     *
     * @return the Convolution Method Hint
     */
    static ConvolutionMethod get_convolution_method(const ITensorInfo         *src,
                                                    const ITensorInfo         *weights,
                                                    const ITensorInfo         *dst,
                                                    const PadStrideInfo       &conv_info,
                                                    const WeightsInfo         &weights_info         = WeightsInfo(),
                                                    const Size2D              &dilation             = Size2D(1U, 1U),
                                                    const ActivationLayerInfo &act_info             = ActivationLayerInfo(),
                                                    bool                       enable_fast_math     = false,
                                                    bool                       use_direct_i8_s8_f32 = false,
                                                    bool                       use_direct_u8_u8_f32 = false);
    // Inherited methods overridden:
    void                             run(ITensorPack &tensors) override;
    void                             prepare(ITensorPack &constants) override;
    experimental::MemoryRequirements workspace() const override;

private:
    std::unique_ptr<ICpuOperator>    _function;
    experimental::MemoryRequirements _aux_mem{};
};
} // namespace cpu
} // namespace arm_compute

#endif // ACL_SRC_CPU_OPERATORS_CPUCONV2D_H
