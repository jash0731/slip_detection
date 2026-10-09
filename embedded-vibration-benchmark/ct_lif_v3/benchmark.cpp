#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include "models_generated.hpp"

int main(int argc,char** argv) {
    if(argc!=4) return 2;
    std::ifstream in(argv[1],std::ios::binary);
    int32_t n=0;
    in.read(reinterpret_cast<char*>(&n),sizeof(n));
    if(n<=0||n>100000) return 3;
    std::vector<int32_t> object(n);
    std::vector<float> raw(size_t(n)*34*54);
    in.read(reinterpret_cast<char*>(object.data()),n*sizeof(int32_t));
    in.read(reinterpret_cast<char*>(raw.data()),raw.size()*sizeof(float));
    if(!in) return 4;
    constexpr size_t model_count=sizeof(models)/sizeof(models[0]);
    std::vector<std::vector<int>> ids(model_count);
    std::ofstream predictions(argv[2]);
    predictions<<"checkpoint,index,logit\n"<<std::setprecision(10);
    for(size_t m=0;m<model_count;++m) {
        for(int i=0;i<n;++i) if(object[i]==models[m].held) ids[m].push_back(i);
        for(int i:ids[m]) predictions<<models[m].name<<','<<i<<','<<models[m].predict(raw.data()+size_t(i)*34*54)<<'\n';
    }
    std::vector<int> order(model_count);
    std::iota(order.begin(),order.end(),0);
    std::mt19937 rng(20261009);
    std::vector<std::vector<double>> timings(model_count);
    volatile float sink=0;
    using clock=std::chrono::steady_clock;
    for(int round=0;round<5;++round) {
        std::shuffle(order.begin(),order.end(),rng);
        for(int m:order) {
            for(int j=0;j<32;++j) sink=sink+models[m].predict(raw.data()+size_t(ids[m][j%ids[m].size()])*34*54);
            for(int j=0;j<1024;++j) {
                int id=ids[m][(j*17+round*31)%ids[m].size()];
                auto start=clock::now();
                float result=models[m].predict(raw.data()+size_t(id)*34*54);
                auto end=clock::now();
                sink=sink+result;
                timings[m].push_back(std::chrono::duration<double,std::micro>(end-start).count());
            }
        }
    }
    std::ofstream costs(argv[3]);
    costs<<"checkpoint,median_us,p95_us,samples\n"<<std::setprecision(10);
    for(size_t m=0;m<model_count;++m) {
        auto& t=timings[m];
        std::sort(t.begin(),t.end());
        costs<<models[m].name<<','<<t[t.size()/2]<<','<<t[size_t(.95*(t.size()-1))]<<','<<t.size()<<'\n';
    }
    std::cout<<"Benchmarked "<<model_count<<" selected checkpoints on host; sink="<<sink<<'\n';
}
