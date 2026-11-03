#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
namespace strata::llm_jp {
using ProfileClock=std::chrono::steady_clock;
struct Profile {
    bool enabled=std::getenv("STRATA_LLMJP_PROFILE")!=nullptr;
    double attention=0,moe=0,route=0,stage=0,expert=0,head=0,layer_other=0;
    unsigned long layers=0,heads=0;
    static double elapsed(ProfileClock::time_point start) {
        return std::chrono::duration<double,std::milli>(ProfileClock::now()-start).count();
    }
    void report() const {
        if(enabled)std::fprintf(stderr,"profile layers=%lu heads=%lu attention_ms=%.2f moe_ms=%.2f route_ms=%.2f stage_enqueue_ms=%.2f expert_remaining_ms=%.2f layer_other_ms=%.2f head_ms=%.2f\n",layers,heads,attention,moe,route,stage,expert,layer_other,head);
    }
};
inline Profile profile;
}
