// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <string>
#include "lite/core/kernel.h"
#include "lite/operators/op_params.h"
#include "lite/utils/cp_logging.h"

#include "lite/api/paddle_place.h"
#include "lite/core/op_registry.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

// TODO(ysh329): add layout trans kernel
void TransHwcToChw(Tensor* dest, const Tensor* src) {}
void TransChwToHwc(Tensor* dest, const Tensor* src) {}

class LayoutComputeBufferChwToImage2DHwc
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNHWC)> {
 public:
  void Run() override {
    auto& param = Param<operators::LayoutParam>();
    param.y->mutable_data<float, cl::Image>(TARGET(kOpenCL));
    // TODO(ysh329): add layout trans kernel
    // TransChwToHwc(param.y, param.x);
  }

  std::string doc() const override { return "Trans Layout from NCHW to NHWC"; }
};

class LayoutComputeImage2DHwcToBufferChw
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  void Run() override {
    auto& param = Param<operators::LayoutParam>();
    param.y->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
    // TODO(ysh329): add layout trans kernel
    // TransChwToHwc(param.y, param.x);
  }

  std::string doc() const override { return "Trans Layout from NHWC to NCHW"; }
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// BufferChwToImage2DHwc
// [chw] -> [hwc]
REGISTER_LITE_KERNEL(
    layout,
    kOpenCL,
    kFloat,
    kNHWC,
    paddle::lite::kernels::opencl::LayoutComputeBufferChwToImage2DHwc,
    buffer_chw_to_image2d_hwc_opencl_fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

// [chw] -> [hwc]
REGISTER_LITE_KERNEL(
    layout_once,
    kOpenCL,
    kFloat,
    kNHWC,
    paddle::lite::kernels::opencl::LayoutComputeBufferChwToImage2DHwc,
    buffer_chw_to_image2d_hwc_opencl_fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

// Image2DHwcBufferChw
// [hwc] -> [chw]
REGISTER_LITE_KERNEL(
    layout,
    kOpenCL,
    kFloat,
    kNCHW,
    paddle::lite::kernels::opencl::LayoutComputeImage2DHwcToBufferChw,
    image2d_hwc_to_buffer_chw_opencl_fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

// [hwc] -> [chw]
REGISTER_LITE_KERNEL(
    layout_once,
    kOpenCL,
    kFloat,
    kNCHW,
    paddle::lite::kernels::opencl::LayoutComputeImage2DHwcToBufferChw,
    image2d_hwc_to_buffer_chw_opencl_fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
