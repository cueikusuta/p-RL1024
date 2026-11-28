// Synthetic benchmark native dispatch; uses original ATen arithmetic.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <tuple>
using at::Tensor;
using Out=std::tuple<Tensor,Tensor,Tensor>;
namespace {
inline Tensor relu(Tensor t,bool reuse) { return reuse ? at::relu_(t) : at::relu(t); }
Out battle(const Tensor& s,const Tensor& a,const Tensor& h,at::TensorList w,bool reuse) {
  TORCH_CHECK(w.size()==14,"Bad parameter schema");
  auto state=relu(at::linear(s,w[0],w[1]),reuse);
  auto next=at::gru_cell(state,h,w[2],w[3],w[4],w[5]);
  auto repeated=next.unsqueeze(1).expand({-1,a.size(1),-1});
  auto input=at::cat({repeated,a},-1);
  auto policy=relu(at::linear(input,w[6],w[7]),reuse);
  auto logits=at::linear(policy,w[8],w[9]).squeeze(-1);
  auto latent=relu(at::linear(next,w[10],w[11]),reuse);
  auto value=at::linear(latent,w[12],w[13]);
  value=reuse?at::tanh_(value):at::tanh(value);
  return {logits,value.squeeze(-1),next};
}
}
TORCH_LIBRARY(speed_native_battle,m) {
  m.def("forward(Tensor state, Tensor actions, Tensor memory, Tensor[] weights, bool reuse) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(speed_native_battle,CompositeImplicitAutograd,m) {m.impl("forward",battle);}
