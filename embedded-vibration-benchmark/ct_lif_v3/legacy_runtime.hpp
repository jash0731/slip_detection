#pragma once
#include <algorithm>
#include <cmath>

template<bool NeedAbsolute>
inline void pin_stats(const float* now, const float* prev, float* m, float* a) {
    for (int axis=0; axis<3; ++axis) {
        m[axis]=0; a[axis]=0;
        for (int p=0; p<18; ++p) {
            float d=now[p*3+axis]-prev[p*3+axis];
            m[axis]+=d;
            if constexpr(NeedAbsolute) a[axis]+=std::fabs(d);
        }
        m[axis]/=18.f; a[axis]/=18.f;
    }
}

template<int C,int H,int T,int Kind,int Variant>
float infer_window(const float* raw,const float* W,const float* b,const float* w,float bias,const float* scale) {
    float state[H]={}, counts[H]={}, feature[C]={};
    float previous_p[Variant==0?54:1]={};
    float previous_gate[3]={};
    bool previous_good[3]={};
    constexpr int start=34-T;
    if constexpr(Variant==0) {
        float m[3], a[3];
        pin_stats<true>(raw+(start-1)*54,raw+(start-2)*54,m,a);
        for(int axis=0;axis<3;++axis) {
            previous_good[axis]=a[axis]>1e-12f*scale[axis];
            previous_gate[axis]=std::min(1.f,a[axis]/(.02f*scale[axis]));
            for(int p=0;p<18;++p)
                previous_p[p*3+axis]=previous_good[axis]?std::fabs(raw[(start-1)*54+p*3+axis]-raw[(start-2)*54+p*3+axis])/(18.f*a[axis]):0;
        }
    }
    for(int t=start;t<34;++t) {
        float m[3],a[3];
        pin_stats<Variant!=2>(raw+t*54,raw+(t-1)*54,m,a);
        for(int axis=0;axis<3;++axis) {
            if constexpr(Variant<=1) {
                bool good=a[axis]>1e-12f*scale[axis];
                float q=good?m[axis]/a[axis]:0;
                float cancellation=good?std::max(0.f,1.f-std::fabs(q)):0;
                float gate=std::min(1.f,a[axis]/(.02f*scale[axis]));
                feature[axis]=std::max(0.f,gate*q);
                feature[3+axis]=std::max(0.f,-gate*q);
                feature[6+axis]=gate*cancellation;
                feature[9+axis]=0;
                if constexpr(Variant==0) {
                    float turnover=0;
                    for(int p=0;p<18;++p) {
                        float pi=good?std::fabs(raw[t*54+p*3+axis]-raw[(t-1)*54+p*3+axis])/(18.f*a[axis]):0;
                        turnover+=std::fabs(pi-previous_p[p*3+axis]);
                        previous_p[p*3+axis]=pi;
                    }
                    feature[9+axis]=(good&&previous_good[axis])?std::min(gate,previous_gate[axis])*cancellation*.5f*turnover:0;
                    previous_good[axis]=good;
                    previous_gate[axis]=gate;
                }
            } else {
                float opposed=std::max(0.f,a[axis]-std::fabs(m[axis]));
                if constexpr(Variant==4) m[axis]=std::copysign(std::max(0.f,std::fabs(m[axis])-.5f*opposed),m[axis]);
                float z=m[axis]/scale[axis];
                feature[axis]=std::min(1.f,std::max(0.f,z));
                feature[3+axis]=std::min(1.f,std::max(0.f,-z));
                if constexpr(Variant==3) feature[6+axis]=std::min(1.f,opposed/scale[axis]);
            }
        }
        float current[H];
        for(int h=0;h<H;++h) current[h]=b[h];
        for(int c=0;c<C;++c)
            for(int h=0;h<H;++h) current[h]+=feature[c]*W[c*H+h];
        for(int h=0;h<H;++h) {
            float v=.875f*state[h]+current[h];
            float activity;
            if constexpr(Kind==0) {
                activity=v>=1.f?1.f:0.f;
                state[h]=v-activity;
            } else {
                activity=std::max(0.f,v);
                state[h]=activity;
            }
            counts[h]+=activity;
        }
    }
    float logit=bias;
    for(int h=0;h<H;++h) logit+=w[h]*(counts[h]/float(T));
    return logit;
}
