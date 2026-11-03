// Metadata-only inventory for the experimental LLM-jp engine. Not an inference path.
#include "strata/artifact/gguf_reader.hpp"
#include <cstdio>
#include <exception>
#include <map>
#include <string>
int main(int argc,char** argv) {
 if(argc!=2){std::fprintf(stderr,"usage: llm-jp-inspect MODEL.gguf\n");return 2;}
 try {
  strata::GgufFile f(argv[1]); const auto* arch=f.get("general.architecture");
  if(!arch || arch->s!="qwen3moe"){std::fprintf(stderr,"expected qwen3moe GGUF architecture\n");return 3;}
  std::printf("architecture: %s\n",arch->s.c_str());
  for(auto name:{"block_count","embedding_length","attention.head_count","attention.head_count_kv","attention.key_length","expert_count","expert_used_count","expert_feed_forward_length","context_length"}){
   std::string key=std::string("qwen3moe.")+name;auto* v=f.get(key);if(v && v->is_num())std::printf("%s: %llu\n",key.c_str(),(unsigned long long)v->u);else std::printf("%s: absent\n",key.c_str());
  }
  std::map<std::string,size_t> types; size_t missing_payload=0;
  for(const auto& t:f.tensors()) {
   ++types[t.type_name()];
   const uint64_t bytes=strata::tensor_payload_bytes(t);
   if(!bytes || t.offset>f.file_size()-f.data_start() || bytes>f.file_size()-f.data_start()-t.offset)
    ++missing_payload;
  }
  std::printf("tensors: %zu\n",f.tensors().size());for(auto& [type,count]:types)std::printf("  %s: %zu\n",type.c_str(),count);
  for(auto name:{"token_embd.weight","output_norm.weight","output.weight","blk.0.attn_q.weight","blk.0.attn_k.weight","blk.0.attn_v.weight","blk.0.attn_output.weight","blk.0.attn_q_norm.weight","blk.0.attn_k_norm.weight","blk.0.ffn_gate_inp.weight","blk.0.ffn_gate_exps.weight","blk.0.ffn_up_exps.weight","blk.0.ffn_down_exps.weight"}){
   auto* t=f.find(name);if(!t){std::printf("%s: missing\n",name);continue;}std::printf("%s: %s [",name,t->type_name());for(size_t i=0;i<t->shape.size();i++)std::printf("%s%llu",i?",":"",(unsigned long long)t->shape[i]);std::puts("]");
  }
  if(missing_payload){std::fprintf(stderr,"%zu tensors have unknown or incomplete payloads; not ready for inference\n",missing_payload);return 4;}
 }catch(const std::exception& e){std::fprintf(stderr,"GGUF error: %s\n",e.what());return 1;}
}
