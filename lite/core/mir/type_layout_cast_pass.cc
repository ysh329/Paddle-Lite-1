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

#include "lite/core/mir/type_layout_cast_pass.h"
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {
namespace mir {

void TypeLayoutTransformPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // Start from inputs of the graph, those should have place set.
  VLOG(4) << "\n" << Visualize(graph.get());
  std::list<Node*> nodes;
  for (auto& node : graph->mutable_nodes()) {
    nodes.push_back(&node);
  }

  // 遍历graph中的类型为Stmt的node, 并取出其每1个输入(inlinks中的每个节点)并依次调用ComplementInputs
  // 补充: node有2种类型：arg表示参数如kernel的输入输出, stmt表示kernel
  LOG(INFO) << "nodes.size():" << nodes.size();
  for (auto& node : nodes) {
    LOG(INFO) << "!node->IsStmt():" << !node->IsStmt();
    if (!node->IsStmt()) continue;
    auto inlinks = node->inlinks;
    LOG(INFO) << "node->AsStmt().desc:" << node->AsStmt().desc
              << " inlinks.size():" << inlinks.size();
    for (auto* in : inlinks) {
      ComplementInputs(graph.get(), node, in);
    }
  }
  VLOG(4) << "\n" << Visualize(graph.get());
}

// SSAGraph* graph: 模型的图
// Node* inst_node: stmt类型的node(即kernel)
// Node* in: 该stmt类型的node的其中一个输入(inlinks中的一个元素)
void TypeLayoutTransformPass::ComplementInputs(SSAGraph* graph,
                                               Node* inst_node,
                                               Node* in) {
  // If this input is out of date.
  if (inst_node->inlinks.end() ==
      std::find(inst_node->inlinks.begin(), inst_node->inlinks.end(), in))
    return;

  CHECK(inst_node->IsStmt());
  auto& inst = inst_node->AsStmt();
  LOG(INFO) << "found Target tensor: " << in->AsArg().name;
  CHECK(in->IsRoleSet());
  CHECK(in->IsArg());
  
  // 根据in的arg名字, 取出stmt(即kernel, 同名kernel的选择在static_pick_kernel这个pass中完成)中, 
  //     以该名字为输入的Tensor所声明的类型信息(decl_arg_type)
  // 检查in的type(包含target/layout/precision/device)中的layout, 与Tensor声明的类型信息中的layout,
  //     若不一致(!DataLayoutCompatible), 
  //     则在该Tensor与该kernel(即stmt)中间插入Layout(即调用AddLayoutInst)
  // ??是否可以说: in是上一kernel的输出, decl_arg是当前kernel的输入
  auto in_arg_name = in->AsArg().name;
  std::string tmp;
  CHECK(inst.op_info()->GetInputArgname(in_arg_name, &tmp));
  auto decl_arg_type = inst.picked_kernel().GetInputDeclType(tmp);
  CHECK(in->AsArg().type);
  LOG(INFO) << "\n tmp:" << tmp << "\n in->AsArg().name:" << in->AsArg().name
            << "\n *in->AsArg().type:" << *in->AsArg().type
            << "\n *decl_arg_type:" << *decl_arg_type
            << "\n inst.op()->DebugString():" << inst.op()->DebugString();

  if (!DataLayoutCompatible(*in->AsArg().type, *decl_arg_type)) {
    LOG(INFO) << "found Layout unmatched tensor: " << in->AsArg().name
              << " for kernel " << inst.op()->DebugString() << " "
              << *in->AsArg().type << " -> " << *decl_arg_type;
    AddLayoutInst(*in->AsArg().type,
                  *decl_arg_type,
                  in,
                  graph,
                  inst_node,
                  graph->valid_places());
  }
}

// const Type& from: stmt的inlinks的1个(上一个函数中的in)的type信息(包含target/layout/precison/device)
// const Type& to: 根据from(上一函数中的in)的arg name, 在stmt中找的声明的Tensor的type
// Node* in: 上一个函数中的in, 跟前两个参数(from, to)都有关系：
//           from -> in的AsArg().type
//           to -> 根据in AsArg().name 在stmt中inst.picked_kernel().GetInputDeclType
// SSAGraph* graph: 同上一个函数
// Node* inst_node: 同上一个函数
// const std::vector<Place>& valid_places: graph->valid_places()
void TypeLayoutTransformPass::AddLayoutInst(
    const Type& from,
    const Type& to,
    Node* in,
    SSAGraph* graph,
    Node* inst_node,
    const std::vector<Place>& valid_places) {

  CHECK(!valid_places.empty()) << "valid_place should be set";
  CHECK(in->IsArg());

  // 创建layout inst node(后面创建)所对应的output node, 即Arg node
  //     其output的name参数的确定基于: in->AsArg().name.c_str(), node_id()
  //     其output的type参数的确定基于: from.target(), from.precision(), to.layout()
  auto node_id = [&] { return graph->nodes().size(); };
  auto layout_output_name =
      string_format("%s/layout_trans/%d", in->AsArg().name.c_str(), node_id());
  auto* layout_output_arg = graph->NewArgumentNode(layout_output_name);
  layout_output_arg->AsArg().type =
      LiteType::GetTensorTy(from.target(), from.precision(), to.layout());

  // 创建layout的inst node
  //     layout_type(layout|layout_once)基于: in->AsArg().is_weight || in->AsArg().is_persist
  //                                          该参数用于layout_op和layout_inst
  auto* layout_inst = graph->NewInstructNode();
  bool in_persist = in->AsArg().is_weight || in->AsArg().is_persist;
  std::string layout_type = in_persist ? "layout_once" : "layout";

  // 基于layout_type创建layout_op
  auto layout_op = LiteOpRegistry::Global().Create(layout_type);
  CHECK(layout_op) << "create op [" << layout_op << "] failed";
  layout_output_arg->AsArg().is_persist = in_persist;

  // 基于layout_output_name手动创建var, 并填入inst_node对应op中的scope, 作为一个var
  //     即layout的output的var
  inst_node->AsStmt().op()->scope()->Var(layout_output_name);

  // Create Layout Instruction.
  cpp::OpDesc op_desc;
  op_desc.SetType(layout_type);
  op_desc.SetInput("Input", {in->AsArg().name});
  op_desc.SetOutput("Out", {layout_output_name});

  layout_op->Attach(op_desc, inst_node->AsStmt().op()->scope());
  auto kernels = layout_op->CreateKernels(valid_places);
  std::vector<std::unique_ptr<KernelBase>> selected_kernels;
  bool is_found = false;
  for (auto& kernel : kernels) {
    const Type* in_arg_ty = kernel->GetInputDeclType("Input");
    const Type* out_arg_ty = kernel->GetOutputDeclType("Out");
#ifdef LITE_WITH_OPENCL
    // ignore [layout check] for layout trans from image2d to buffer
    if (TargetCompatibleTo(*in_arg_ty, from) &&
        PrecisionCompatibleTo(*in_arg_ty, from) &&
        DeviceCompatibleTo(*in_arg_ty, from)) {
#else
    if (TypeCompatible(*in_arg_ty, from) &&
        out_arg_ty->layout() == to.layout()) {
#endif
      is_found = true;
      selected_kernels.emplace_back(std::move(kernel));
      // we pick the kernel
      layout_inst->AsStmt(layout_type, std::move(selected_kernels), layout_op);
      break;
    }
  }
  CHECK(is_found) << "Can't find a layout  kernel for layout op: " << from
                  << ":" << in->AsArg().name << "->" << to << ":"
                  << inst_node->AsStmt().op_info()->Type();
  LOG(INFO) << "========= final picked kernel [info]:"
            << layout_inst->AsStmt().picked_kernel().name()
            << " [summary]:" << layout_inst->AsStmt().picked_kernel().summary()
            << "\n";

  // Remove the old link
  RemoveDirectedLink(in, inst_node);

  // Update the original instruction OpDesc.
  // Update its input to the layout_output_name
  // Add new link, var -> new_inst, new_inst->newarg, newarg->inst
  DirectedLink(in, layout_inst);
  DirectedLink(layout_inst, layout_output_arg);
  DirectedLink(layout_output_arg, inst_node);

  // reset opdesc and update kernel information
  UpdateInputTo(inst_node->AsStmt().op()->mutable_op_info(),
                in->AsArg().name,
                layout_output_name);
  auto original_selected_kernel =
      std::move(inst_node->AsStmt().kernels().front());
  auto update_op_info = *inst_node->AsStmt().op_info();
  // ResetOp() will change the Stmt op_info_ value,
  // after that the old op_info_ value will be nullified.
  // So, we can't pass `*inst_node->AsStmt().op_info()` into ResetOp.
  // `update_op_info` is the copy of `*inst_node->AsStmt().op_info().
  // Whenever update the op_info of a stmt, we should call its ResetOp().
  inst_node->AsStmt().ResetOp(update_op_info, graph->valid_places());
  inst_node->AsStmt().kernels().clear();
  inst_node->AsStmt().kernels().emplace_back(
      std::move(original_selected_kernel));

  std::string tmp;
  if (inst_node->AsStmt().op_info()->GetInputArgname("a", &tmp)) {
    CHECK(false) << "get old a " << tmp;
  }

  for (auto& kernel : inst_node->AsStmt().kernels()) {
    inst_node->AsStmt().op()->AttachKernel(kernel.get());
  }

  graph->CheckValid();
}

void TypeLayoutTransformPass::SetValidPlaces(
    const std::vector<Place>& valid_places) {
  CHECK(!valid_places.empty());
  valid_places_ = valid_places;
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(type_layout_cast_pass,
                  paddle::lite::mir::TypeLayoutTransformPass)
    .BindTargets({TARGET(kAny)})
    .BindKernel("layout_once")
    .BindKernel("layout");
