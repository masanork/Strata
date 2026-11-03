// Optional diagnostics only. Compile and link against a separate matching llama.cpp build.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
struct Trace { const char* dir; int pos=0; };
static bool callback(ggml_tensor* t,bool ask,void* ptr) {
 const char* names[]={"l_out","attn_norm","ffn_inp","ffn_norm","ffn_moe_out","Qcur","Kcur","Vcur"};const char* selected=nullptr;
 for(auto n:names)if(std::strncmp(t->name,n,std::strlen(n))==0 && t->name[std::strlen(n)]=='-'){selected=n;break;}
 if(!selected)return false;
 if(ask)return true;
 auto* trace=static_cast<Trace*>(ptr);int layer=-1;
 layer=std::atoi(t->name+std::strlen(selected)+1);
 if(!ggml_is_contiguous(t)||t->type!=GGML_TYPE_F32||(ggml_nelements(t)!=2560 && ggml_nelements(t)!=512 && ggml_nelements(t)!=5120))return true;
 std::vector<float> v(ggml_nelements(t));ggml_backend_tensor_get(t,v.data(),0,v.size()*4);
 auto path=std::string(trace->dir)+"/"+std::to_string(trace->pos)+"_"+std::to_string(layer)+"_"+selected+".bin";
 FILE* f=std::fopen(path.c_str(),"wb");if(!f)std::abort();auto n=std::fwrite(v.data(),4,v.size(),f);std::fclose(f);if(n!=v.size())std::abort();
 return true;
}
int main(int argc,char** argv){
 if(argc<4)return 2;llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;auto* model=llama_model_load_from_file(argv[1],mp);if(!model)return 3;
 Trace trace{argv[2]};auto cp=llama_context_default_params();cp.n_ctx=128;cp.n_batch=128;cp.n_threads=6;cp.n_threads_batch=6;cp.cb_eval=callback;cp.cb_eval_user_data=&trace;
 auto* ctx=llama_init_from_model(model,cp);if(!ctx)return 4;
 for(int i=3;i<argc;i++){trace.pos=i-3;llama_token tok=std::atoi(argv[i]);if(llama_decode(ctx,llama_batch_get_one(&tok,1)))return 5;}
 llama_free(ctx);llama_model_free(model);llama_backend_free();
}
