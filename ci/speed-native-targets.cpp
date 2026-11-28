// Exact-order CPU implementation of the per-decision two-seat trace targets.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <array>
#include <cmath>
#include <vector>
#include <tuple>
using at::Tensor;
using Tensors=std::vector<Tensor>;
namespace {
std::tuple<Tensors,Tensor> calculate(at::TensorList policy,at::TensorList ref,const Tensor& meta,double eta,double rho_cap,double c_cap) {
 const auto n=meta.size(0);
 TORCH_CHECK(n>0 && n==int64_t(policy.size()) && policy.size()==ref.size() && meta.size(1)==6);
 TORCH_CHECK(eta>=0 && rho_cap>0 && c_cap>0);
 const auto input=meta.accessor<double,2>();
 auto binary=at::zeros({n,7},meta.options());
 auto metrics=binary.accessor<double,2>();
 Tensors results(n);
 std::array<double,2> trace{}, next{}, opponent_reward{}, reach{1,1};
 for(int64_t i=n-1;i>=0;--i) {
   const auto k=int(input[i][0]), o=1-k;
   const auto old=input[i][3], prob=input[i][4];
   const auto action=int64_t(input[i][5]);
   TORCH_CHECK((k==0||k==1) && prob>0 && prob<=1 && std::isfinite(static_cast<float>(old)) && std::abs(old)<1e6);
   auto p=policy[i].detach().flatten().to(at::kDouble).contiguous();
   auto m=ref[i].detach().flatten().to(at::kDouble).contiguous();
   TORCH_CHECK(p.numel()>0 && m.numel()==p.numel() && action>=0 && action<p.numel());
   auto pa=p.const_data_ptr<double>(), ma=m.const_data_ptr<double>();
   for(int64_t j=0;j<p.numel();++j) TORCH_CHECK(std::isfinite(pa[j]) && std::isfinite(ma[j]) && pa[j]>0 && ma[j]>0);
   TORCH_CHECK(std::abs(p.sum().item<double>()-1)<=1e-5 && std::abs(m.sum().item<double>()-1)<=1e-5);
   const auto importance=pa[action]/prob;
   auto regularizer=at::log(at::div(p,m));
   const auto observed=regularizer.const_data_ptr<double>()[action];
   std::array<double,2> reward{input[i][1],input[i][2]};
   TORCH_CHECK(std::abs(reward[0]+reward[1])<=1e-6);
   reward[k]-=eta*observed; reward[o]+=eta*observed;
   TORCH_CHECK(std::abs(reward[0]+reward[1])<=1e-8);
   opponent_reward[o]=reward[o]+importance*opponent_reward[o];
   reach[o]*=importance;
   const auto corrected=importance*reach[k];
   const auto rho=std::min(rho_cap,corrected), c=std::min(c_cap,corrected);
   const auto previous=trace[k];
   const auto delta=rho*(reward[k]+importance*opponent_reward[k]+next[k]-old);
   const auto v=old+delta+c*(previous-next[k]);
   auto q=at::add(at::mul(regularizer,-eta),old);
   const auto update=(reward[k]+eta*observed+importance*(opponent_reward[k]+previous)-old)/prob;
   q.data_ptr<double>()[action]+=update;
   for(int64_t j=0;j<q.numel();++j) TORCH_CHECK(std::isfinite(q.const_data_ptr<double>()[j]));
   TORCH_CHECK(std::isfinite(static_cast<float>(v)));
   results[i]=q.to(at::kFloat);
   metrics[i][0]=v; metrics[i][1]=rho; metrics[i][2]=c;metrics[i][3]=importance;
   metrics[i][4]=k; metrics[i][5]=reward[0];metrics[i][6]=reward[1];
   trace[k]=v; next[k]=old; opponent_reward[k]=0;reach[k]=1;
 }
 return {results,binary};
}
}
TORCH_LIBRARY(speed_native_targets,m) {
 m.def("targets(Tensor[] policy, Tensor[] reference, Tensor metadata, float eta, float rho_cap, float c_cap) -> (Tensor[], Tensor)");
}
TORCH_LIBRARY_IMPL(speed_native_targets,CPU,m) {m.impl("targets",calculate);}
