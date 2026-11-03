// Optional independent reference for token-ID checks; not linked into Strata.
// Build against an external llama.cpp installation with its matching headers.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
int main(int argc,char** argv){
 if(argc<3)return 2;llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;auto* model=llama_model_load_from_file(argv[1],mp);if(!model)return 3;
 auto cp=llama_context_default_params();cp.n_ctx=128;cp.n_batch=128;cp.n_threads=6;cp.n_threads_batch=6;auto* ctx=llama_init_from_model(model,cp);if(!ctx)return 4;
 for(int i=2;i<argc;i++){llama_token tok=std::atoi(argv[i]);if(llama_decode(ctx,llama_batch_get_one(&tok,1)))return 5;}
 const float* logits=llama_get_logits_ith(ctx,-1);int n=llama_vocab_n_tokens(llama_model_get_vocab(model)),best=0;for(int i=1;i<n;i++)if(logits[i]>logits[best])best=i;
 std::printf("oracle_next_token=%d logit=%.9g candidate_5371=%.9g\n",best,logits[best],logits[5371]);llama_free(ctx);llama_model_free(model);llama_backend_free();
}
