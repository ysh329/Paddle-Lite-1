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

#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/operators/op_params.h"
#include "lite/utils/replace_stl/stream.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class ReluCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::ActivationParam;

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "buffer/relu_kernel.cl", build_options_);
  }

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    const auto& x_dims = param.X->dims();
    size_t count = x_dims.production();

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    auto* x_buf = param.X->data<float, cl::Buffer>();
    auto* out_buf = param.Out->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());
    VLOG(4) << TargetToStr(param.X->target());
    VLOG(4) << TargetToStr(param.Out->target());

    int arg_idx = 0;
    cl_int status = kernel.setArg(arg_idx, *x_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, (const int)count);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *out_buf);
    CL_CHECK_FATAL(status);

    auto global_work_size = cl::NDRange{count};
    status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        global_work_size,
        cl::NullRange,
        nullptr,
        event_.get());
    CL_CHECK_FATAL(status);
    context.cl_wait_list()->emplace(out_buf, event_);
  }

 private:
  std::string kernel_func_name_{"relu"};
  std::string build_options_{"-DCL_DTYPE=float -DRELU"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

class ReluComputeFloatNHWC
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::ActivationParam;

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "buffer/relu_kernel.cl", build_options_);
  }

  DDim InitImageDimInfoWith(const DDim& tensor_dim) {
    size_t new_dims[] = {1, 1, 1, 1};
    for (size_t j = 0; j < tensor_dim.size(); ++j) {
      new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
    }
    size_t N, C, H, W;
    N = new_dims[0];
    C = new_dims[1];
    H = new_dims[2];
    W = new_dims[3];
    size_t width = W * ((C + 3) / 4);
    size_t height = H * N;
    return DDim(
        std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                       static_cast<DDim::value_type>(height)}));
  }

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    const auto& x_dims = param.X->dims();
    auto image_shape = InitImageDimInfoWith(x_dims);
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    auto* x_buf = param.X->data<float, cl::Image2D>();
    auto* out_buf = param.Out->mutable_data<float, cl::Image2D>(
        TARGET(kOpenCL), image_shape[0], image_shape[1]);

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());
    VLOG(4) << TargetToStr(param.X->target());
    VLOG(4) << TargetToStr(param.Out->target());

    int arg_idx = 0;
    cl_int status = kernel.setArg(arg_idx, *x_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *out_buf);
    CL_CHECK_FATAL(status);

    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>(image_shape[0]),
                    static_cast<cl::size_type>(image_shape[1])};
    status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        global_work_size,
        cl::NullRange,
        nullptr,
        event_.get());
    CL_CHECK_FATAL(status);
    context.cl_wait_list()->emplace(out_buf, event_);
  }

 private:
  std::string kernel_func_name_{"relu_image"};
  std::string build_options_{"-DCL_DTYPE=float -DRELU"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// REGISTER_LITE_KERNEL(relu,
//                     kOpenCL,
//                     kFloat,
//                     kNCHW,
//                     paddle::lite::kernels::opencl::ReluCompute,
//                     def)
//    .BindInput("X", {LiteType::GetTensorTy(TARGET(kOpenCL))})
//    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kOpenCL))})
//    .Finalize();

REGISTER_LITE_KERNEL(relu,
                     kOpenCL,
                     kFloat,
                     kNHWC,
                     paddle::lite::kernels::opencl::ReluComputeFloatNHWC,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
