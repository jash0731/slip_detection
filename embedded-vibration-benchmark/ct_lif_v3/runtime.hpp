#pragma once
#include <algorithm>
#include <cmath>

template<int C,int H,bool Residual,bool Recent,int Variant>
float infer_v3(const float* raw,const float* W,const float* b,const float* w,const float* q,float bias,const float* scale) {
    float state[H>0?H:1]={}, count[H>0?H:1]={}, feature[C]={};
    float evidence[Residual||H==0?C:1]={};
    for(int t=2;t<34;++t) {
        for(int axis=0;axis<3;++axis) {
            float m=0, a=0;
            for(int pin=0;pin<18;++pin) {
                float difference=raw[t*54+pin*3+axis]-raw[(t-1)*54+pin*3+axis];
                m+=difference;
                if constexpr(Variant!=0) a+=std::fabs(difference);
            }
            m/=18.f; a/=18.f;
            float common=m/(scale[axis]+std::fabs(m));
            float opposed=std::max(0.f,a-std::fabs(m));
            if constexpr(Variant==2) {
                feature[axis]=std::fabs(common);
                feature[axis+3]=opposed/(scale[axis]+opposed);
            } else {
                feature[axis]=std::max(0.f,common);
                feature[axis+3]=std::max(0.f,-common);
                if constexpr(Variant==1) feature[axis+6]=opposed/(scale[axis]+opposed);
            }
        }
        float pooling=Recent?(.015625f+(t>=26?.0625f:0.f)):.03125f;
        if constexpr(Residual||H==0) {
            for(int c=0;c<C;++c) evidence[c]+=pooling*feature[c];
        }
        if constexpr(H>0) {
            float current[H];
            for(int h=0;h<H;++h) current[h]=b[h];
            for(int c=0;c<C;++c)
                for(int h=0;h<H;++h) current[h]+=feature[c]*W[c*H+h];
            for(int h=0;h<H;++h) {
                float voltage=.875f*state[h]+current[h];
                float spike=voltage>=1.f?1.f:0.f;
                state[h]=voltage-spike;
                count[h]+=pooling*spike;
            }
        }
    }
    float logit=bias;
    if constexpr(H>0) for(int h=0;h<H;++h) logit+=w[h]*count[h];
    if constexpr(Residual||H==0) for(int c=0;c<C;++c) logit+=q[c]*evidence[c];
    return logit;
}
