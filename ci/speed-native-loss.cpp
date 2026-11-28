// Small checked CPU loss and validation primitives for generated timing only.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
using at::Tensor;
namespace {
bool finite(const Tensor& x) {
 if(!x.device().is_cpu() || x.scalar_type()!=at::kFloat || !x.is_contiguous())
   return at::isfinite(x).all().item<bool>();
 const float* data=x.const_data_ptr<float>();
 for(int64_t i=0;i<x.numel();++i) if(!std::isfinite(data[i])) return false;
 return true;
}
bool both_finite(const Tensor& a,const Tensor& b) {return finite(a)&&finite(b);}
Tensor policy_loss(const Tensor& logits,const Tensor& q_targets,double beta,double q_clip,double step,double importance) {
 TORCH_CHECK(step>0 && step<=1 && beta>=0 && q_clip>0 && importance>=0 && importance<=1);
 TORCH_CHECK(logits.dim()==1 && logits.sizes()==q_targets.sizes() && logits.numel()>0);
 TORCH_CHECK(both_finite(logits,q_targets));
 auto q=at::clamp(q_targets.detach().to(logits.options()),-q_clip,q_clip);
 auto probs=at::softmax(logits.detach(),-1);
 auto advantage=at::mul(at::sub(q,at::sum(at::mul(probs,q))),importance);
 auto centered=at::sub(logits,at::mean(logits));
 auto force=advantage;
 if(beta>0) {
   auto proposed=at::add(centered.detach(),at::mul(advantage,step));
   force=at::where(at::le(at::abs(proposed),beta),advantage,at::zeros_like(advantage));
 }
 return at::div(at::neg(at::sum(at::mul(centered,force))),logits.numel());
}
}
TORCH_LIBRARY(speed_native_loss,m) {
 m.def("both_finite(Tensor a, Tensor b) -> bool");
 m.def("policy_loss(Tensor logits, Tensor q, float beta, float clip, float step, float importance) -> Tensor");
}
TORCH_LIBRARY_IMPL(speed_native_loss,CompositeImplicitAutograd,m) {
 m.impl("both_finite",both_finite);
 m.impl("policy_loss",policy_loss);
}
