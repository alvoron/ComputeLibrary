/*
 * Copyright (c) 2021-2026 Arm Limited.
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
#include "src/cpu/operators/CpuGemmDirectConv2d.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/FunctionDescriptors.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/common/utils/Log.h"
#include "src/common/utils/profile/acl_profile.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/utils/CpuAuxTensorHandler.h"
#include "support/Cast.h"

#include <set>

namespace arm_compute
{
namespace cpu
{
using namespace arm_compute::experimental;
using namespace arm_compute::utils::cast;

namespace
{
inline bool is_direct_i8_s8_f32_path(const ITensorInfo *src, const ITensorInfo *dst, const Conv2dInfo &info)
{
    return info.use_direct_i8_s8_f32 && src->data_type() == DataType::QASYMM8_SIGNED &&
           dst->data_type() == DataType::F32;
}

inline bool is_direct_u8_u8_f32_path(const ITensorInfo *src, const ITensorInfo *dst, const Conv2dInfo &info)
{
    return info.use_direct_u8_u8_f32 && src->data_type() == DataType::QASYMM8 &&
           dst->data_type() == DataType::F32;
}

GEMMLowpOutputStageInfo calculate_output_stage_metadata(const ITensorInfo         *src,
                                                        const ITensorInfo         *weights,
                                                        const ITensorInfo         *dst,
                                                        const ActivationLayerInfo &act)
{
    // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
    // Extract and negate input and weights offset
    const QuantizationInfo        iqinfo    = src->quantization_info();
    const QuantizationInfo        wqinfo    = weights->quantization_info();
    const QuantizationInfo        oqinfo    = (dst->total_size() == 0) ? iqinfo : dst->quantization_info();
    const UniformQuantizationInfo uoqinfo   = oqinfo.uniform();
    const DataType                data_type = src->data_type();
    // Merge activation with output stage
    const std::set<ActivationLayerInfo::ActivationFunction> supported_acts = {
        ActivationLayerInfo::ActivationFunction::RELU, ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
        ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU};
    PixelValue type_min{};
    PixelValue type_max{};
    std::tie(type_min, type_max) = get_min_max(data_type);
    int32_t min_activation       = type_min.get<int32_t>();
    int32_t max_activation       = type_max.get<int32_t>();
    if (supported_acts.count(act.activation()) != 0)
    {
        std::tie(min_activation, max_activation) = get_quantized_activation_min_max(act, data_type, uoqinfo);
    }
    GEMMLowpOutputStageInfo os_info;
    os_info.type                     = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
    os_info.gemmlowp_offset          = uoqinfo.offset;
    os_info.gemmlowp_min_bound       = min_activation;
    os_info.gemmlowp_max_bound       = max_activation;
    os_info.is_quantized_per_channel = (weights->data_type() == DataType::QSYMM8_PER_CHANNEL);
    quantization::calculate_quantized_multipliers(iqinfo, wqinfo, oqinfo, os_info);
    return os_info;
}
cpu::AsmGemmInfo init_assembly_metadata(const Conv2dInfo &info, bool is_indirect)
{
    cpu::AsmGemmInfo asm_info;
    asm_info.method                  = is_indirect ? cpu::AsmConvMethod::Indirect : cpu::AsmConvMethod::Conv;
    asm_info.ps_info                 = info.conv_info;
    asm_info.activation_info         = info.act_info;
    asm_info.depth_output_gemm3d     = true;
    asm_info.reinterpret_input_as_3d = true;
    asm_info.padding_top             = info.conv_info.pad_top();
    asm_info.padding_left            = info.conv_info.pad_left();
    asm_info.padding_value           = 0.f;
    asm_info.negated_offsets         = false;
    asm_info.fast_mode               = info.enable_fast_math;
    asm_info.fixed_format            = info.weights_info.weight_format() != WeightFormat::UNSPECIFIED;
    asm_info.weight_format           = info.weights_info.weight_format();
    asm_info.use_fp32_acc            = info.use_fp32_acc;
    return asm_info;
}
} // namespace

CpuGemmDirectConv2d::CpuGemmDirectConv2d()
    : _gemm_asm_func(std::make_unique<CpuGemmAssemblyDispatch>()),
      _activation_func(std::make_unique<CpuActivation>()),
      _weights_permute_func(std::make_unique<CpuPermute>()),
      _weights_flip_func(nullptr),
      _aux_mem(AuxTensorIdx::Count),
      _perm_weights(),
      _flipped_weights(),
      _flip_weights(false),
      _run_activation(false),
      _is_prepared(false)
{
}

CpuGemmDirectConv2d::~CpuGemmDirectConv2d() = default;

void CpuGemmDirectConv2d::configure(const ITensorInfo *src,
                                    const ITensorInfo *weights,
                                    const ITensorInfo *biases,
                                    ITensorInfo       *dst,
                                    const Conv2dInfo  &info)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "CpuGemmDirectConv2d::configure");
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_ERROR_THROW_ON(
        CpuGemmDirectConv2d::validate(src, weights, biases != nullptr ? biases : nullptr, dst, info));
    ARM_COMPUTE_LOG_PARAMS(src, weights, biases, dst, info);

    _run_activation = info.act_info.enabled() && !_gemm_asm_func->is_activation_supported(info.act_info);
    _is_prepared    = false;
    _flip_weights   = is_direct_u8_u8_f32_path(src, dst, info);

    _weights_permute_func->configure(weights, &_perm_weights, PermutationVector{3, 0, 1, 2});

    // For the u8/u8→f32 path: the assembly kernel expects uint8 input + int8 weights.
    // Convert the permuted QASYMM8 weights to QASYMM8_SIGNED (subtract 128 per element)
    // during prepare().  The weight zero-point is adjusted accordingly.
    const ITensorInfo *asm_weights = &_perm_weights;
    if (_flip_weights)
    {
        _weights_flip_func = std::make_unique<kernels::CpuConvertQuantizedSignednessKernel>();
        // _perm_weights is QASYMM8; _flipped_weights will be QASYMM8_SIGNED with offset-128.
        _weights_flip_func->configure(&_perm_weights, &_flipped_weights);
        asm_weights = &_flipped_weights;
    }

    // Configure assembly dispatch
    cpu::AsmGemmInfo asm_info = init_assembly_metadata(info, false);
    if (is_direct_i8_s8_f32_path(src, dst, info))
    {
        // Provide the quantization zero-points via AsmGemmInfo so that create_arm_gemm_dequant
        // can bake them into the DequantizeFloat output stage.  The assembly kernel then handles
        // all offset corrections (col sums, row sums, cross-term) natively.
        asm_info.dequant_a_offset = src->quantization_info().uniform().offset;
        asm_info.dequant_b_offset = weights->quantization_info().uniform().offset;
    }
    else if (_flip_weights)
    {
        // u8/u8→f32: src stays as QASYMM8 (uint8), weights reinterpreted as QASYMM8_SIGNED.
        // The assembly uses the uint8+int8→float kernel (create_arm_gemm_dequant<uint8,int8,float>).
        // a_offset: raw QASYMM8 offset (range [0,255]).
        // b_offset: QASYMM8 offset adjusted by −128 (to match int8 reinterpretation of weights).
        asm_info.dequant_a_offset = src->quantization_info().uniform().offset;
        asm_info.dequant_b_offset = weights->quantization_info().uniform().offset - 128;
    }
    else if (is_data_type_quantized(src->data_type()))
    {
        asm_info.output_stage = calculate_output_stage_metadata(src, weights, dst, info.act_info);
    }
    _gemm_asm_func->configure(src, asm_weights, biases, dst, asm_info);

    // Configure activation (run after offset correction if needed)
    if (_run_activation)
    {
        _activation_func->configure(dst, nullptr, info.act_info);
    }

    // Add auxiliary memory requirements of the assembly dispatch
    const auto asm_mem_req = _gemm_asm_func->workspace();
    for (unsigned int slot = 0; slot < asm_mem_req.size(); ++slot)
    {
        _aux_mem[slot] = asm_mem_req[slot];
    }

    if (_aux_mem[Pretranspose].size > 0)
    {
        // Release permuted weights at end of prepare as they are further transposed by the assembly dispatch
        _aux_mem[PermutedWeights] =
            MemoryInfo(offset_int_vec(PermutedWeights), MemoryLifetime::Prepare, weights->total_size());
    }
    else
    {
        // We must permute weights if they are WeightFormat::UNSPECIFIED
        if (info.weights_info.weight_format() == WeightFormat::UNSPECIFIED)
            _aux_mem[PermutedWeights] =
                MemoryInfo(offset_int_vec(PermutedWeights), MemoryLifetime::Persistent, weights->total_size());
    }

    // For u8/u8→f32: flipped weights (QASYMM8_SIGNED) are needed at least through prepare().
    // They are released if pretranspose consumed them, or kept persistent otherwise.
    if (_flip_weights)
    {
        const MemoryLifetime lifetime =
            (_aux_mem[Pretranspose].size > 0) ? MemoryLifetime::Prepare : MemoryLifetime::Persistent;
        _aux_mem[FlippedWeights] =
            MemoryInfo(offset_int_vec(FlippedWeights), lifetime, _flipped_weights.total_size());
    }

}
Status CpuGemmDirectConv2d::validate(const ITensorInfo *src,
                                     const ITensorInfo *weights,
                                     const ITensorInfo *biases,
                                     const ITensorInfo *dst,
                                     const Conv2dInfo  &info)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "CpuGemmDirectConv2d::validate");
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::BFLOAT16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::QSYMM8_PER_CHANNEL, DataType::BFLOAT16,
                                                         DataType::F16, DataType::F32);
    if (!is_fixed_format(info.weights_info.weight_format()))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, weights);
    }
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.num_groups > 1, "Grouping (num_groups != 1) is not supported on Cpu");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src->data_layout() != DataLayout::NHWC, "Data layout supported is NHWC");
    const DataType    data_type = src->data_type();
    const TensorShape i_shape   = src->tensor_shape();
    const TensorShape w_shape   = weights->tensor_shape();
    ARM_COMPUTE_RETURN_ERROR_ON(w_shape[0] != i_shape[0]);
    ARM_COMPUTE_RETURN_ERROR_ON(info.dilation != Size2D(1U, 1U));
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);

    const bool direct_i8_f32 = is_direct_i8_s8_f32_path(src, dst, info);
    const bool direct_u8_f32 = is_direct_u8_u8_f32_path(src, dst, info);

    if (direct_i8_f32)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8_SIGNED);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QASYMM8_SIGNED);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::F32);
        // Offset correction handled inside the assembly kernel via DequantizeFloat output stage.
    }

    if (direct_u8_f32)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QASYMM8);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::F32);
        // Offset correction handled inside the assembly kernel via DequantizeFloat output stage.
        // Weights are reinterpreted as QASYMM8_SIGNED (subtract 128) before passing to the
        // uint8+int8→float assembly kernel.
    }

    // Validate Permute.  Build the expected output TensorInfo explicitly so that
    // subsequent validations have a properly initialised input.
    const TensorShape perm_shape =
        arm_compute::misc::shape_calculator::compute_permutation_output_shape(*weights, PermutationVector{3, 0, 1, 2});
    TensorInfo perm_weights = weights->clone()->set_tensor_shape(perm_shape);
    ARM_COMPUTE_RETURN_ON_ERROR(CpuPermute::validate(weights, &perm_weights, PermutationVector{3, 0, 1, 2}));

    // For u8/u8→f32: validate the signedness flip (QASYMM8→QASYMM8_SIGNED)
    if (direct_u8_f32)
    {
        TensorInfo flipped_weights;
        ARM_COMPUTE_RETURN_ON_ERROR(
            kernels::CpuConvertQuantizedSignednessKernel::validate(&perm_weights, &flipped_weights));
    }

    // Validate Activation
    const CpuGemmAssemblyDispatch gemm_asm_func;
    const bool run_activation = info.act_info.enabled() && !gemm_asm_func.is_activation_supported(info.act_info);
    if (run_activation)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CpuActivation::validate(dst, nullptr, info.act_info));
    }

    // Validate biases
    if (biases != nullptr)
    {
        if (direct_i8_f32 || direct_u8_f32)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::F32);
        }
        else if (is_data_type_quantized_asymmetric(data_type))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, biases);
        }
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(3));
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }

    if (!direct_i8_f32 && !direct_u8_f32)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    }

    cpu::AsmGemmInfo asm_info = init_assembly_metadata(info, false);
    if (direct_u8_f32)
    {
        // Assembly expects QASYMM8 input + QASYMM8_SIGNED weights → F32.
        // Construct the QASYMM8_SIGNED flipped-weights TensorInfo manually (same as
        // CpuConvertQuantizedSignednessKernel::configure would produce) so the assembly
        // dispatch can validate against a fully initialised TensorInfo.
        const auto src_qinfo  = perm_weights.quantization_info().uniform();
        const auto flipped_qi = QuantizationInfo(src_qinfo.scale, src_qinfo.offset - 128);
        TensorInfo flipped_weights(perm_weights.tensor_shape(), 1,
                                   DataType::QASYMM8_SIGNED, perm_weights.data_layout());
        flipped_weights.set_quantization_info(flipped_qi);
        ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuConvertQuantizedSignednessKernel::validate(
            &perm_weights, &flipped_weights));
        ARM_COMPUTE_RETURN_ON_ERROR(
            cpu::CpuGemmAssemblyDispatch::validate(src, &flipped_weights, biases, dst, asm_info));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(
            cpu::CpuGemmAssemblyDispatch::validate(src, weights, biases, dst, asm_info));
    }
    return Status{};
}
void CpuGemmDirectConv2d::run(ITensorPack &tensors)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "CpuGemmDirectConv2d::run");
    prepare(tensors);

    _gemm_asm_func->run(tensors);

    if (_run_activation)
    {
        ITensor    *io = tensors.get_tensor(ACL_DST);
        ITensorPack pack{{ACL_SRC, io}, {ACL_DST, io}};
        _activation_func->run(pack);
    }
}

void CpuGemmDirectConv2d::prepare(ITensorPack &tensors)
{
    if (!_is_prepared)
    {
        // If we are using fixed-format kernel the weights are already reshaped
        if (_gemm_asm_func && _gemm_asm_func->isVarWeightsKernel())
        {
            _gemm_asm_func->prepare(tensors);
            _is_prepared = true;
            return;
        }
        const ITensor *weights = tensors.get_const_tensor(ACL_SRC_1);
        ITensor       *weights_aux =
            utils::cast::polymorphic_cast<ITensor *>(tensors.get_tensor(offset_int_vec(PermutedWeights)));
        ARM_COMPUTE_ERROR_ON_NULLPTR(weights, weights_aux);

        CpuAuxTensorHandler permuted_weights(_perm_weights, *weights_aux);
        ITensorPack         permute_tensors{{ACL_SRC, weights}, {ACL_DST, permuted_weights.get()}};
        _weights_permute_func->run(permute_tensors);

        if (_flip_weights)
        {
            // Convert QASYMM8 permuted weights → QASYMM8_SIGNED (subtract 128 per element).
            // CpuAuxTensorHandler overlays _flipped_weights TensorInfo (correct 4D shape and
            // strides) onto the raw memory from the FlippedWeights workspace slot.
            // The handler must stay in scope until after _gemm_asm_func->prepare() returns.
            CpuAuxTensorHandler flipped_weights(offset_int_vec(FlippedWeights),
                                                _flipped_weights, tensors, false, false);
            ITensorPack flip_tensors{{ACL_SRC, permuted_weights.get()},
                                     {ACL_DST, flipped_weights.get()}};
            NEScheduler::get().schedule_op(_weights_flip_func.get(), Window::DimY,
                                           _weights_flip_func->window(), flip_tensors);
            tensors.add_const_tensor(ACL_SRC_1, flipped_weights.get());

            // Call prepare of assembly dispatch with flipped weights while handler is in scope.
            _gemm_asm_func->prepare(tensors);
        }
        else
        {
            tensors.add_const_tensor(ACL_SRC_1, permuted_weights.get());

            // Call prepare of assembly dispatch.
            _gemm_asm_func->prepare(tensors);
        }

        _is_prepared = true;
    }
}

experimental::MemoryRequirements CpuGemmDirectConv2d::workspace() const
{
    return _aux_mem;
}
} // namespace cpu
} // namespace arm_compute
