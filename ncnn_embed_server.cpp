// Persistent binary feature service for the upstream OmniMatch application.
#include <net.h>
#include <gpu.h>
#include <layer.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main(int argc,char** argv) {
    if(argc!=3)return 64;
    bool gpu=std::string(argv[2])=="gpu16s";
    if(!gpu && std::string(argv[2])!="cpu")return 65;
    ncnn::Net net;
    int threads=4;
    if(const char* value=std::getenv("DINO_NUM_THREADS")) {
        char* end=nullptr;long n=std::strtol(value,&end,10);
        if(!end||*end||n<1||n>64)return 66;
        threads=static_cast<int>(n);
    }
    net.opt.num_threads=threads;
    net.opt.use_vulkan_compute=gpu;
    net.opt.use_fp16_packed=gpu;
    net.opt.use_fp16_storage=gpu;
    net.opt.use_fp16_arithmetic=false;
    net.opt.use_bf16_storage=false;
    if(gpu) {
        setenv("SCV41_NCNN_STRICT_GPU","1",1);
        ncnn::create_gpu_instance();
        if(ncnn::get_gpu_count()<1)return 10;
        // Do not label a software Vulkan driver as GPU acceleration.
        if(ncnn::get_gpu_info(0).type()==3)return 14;
        std::fprintf(stderr,"DINO_DEVICE backend=vulkan name=%s\n",ncnn::get_gpu_info(0).device_name());
        net.set_vulkan_device(0);
    }
    std::string prefix=argv[1];
    if(net.load_param((prefix+".param").c_str()) || net.load_model((prefix+".bin").c_str()))return 11;
    if(gpu)for(auto l:net.layers())if(!l->support_vulkan)return 12;
    if(gpu&&!net.opt.use_vulkan_compute)return 13;
    std::cout.write("RDY1",4);std::cout.flush();
    std::vector<unsigned char> rgb(224*224*3);
    const float mean[]={123.675f,116.28f,103.53f};
    const float scale[]={1.f/(255.f*.229f),1.f/(255.f*.224f),1.f/(255.f*.225f)};
    while(std::cin.read(reinterpret_cast<char*>(rgb.data()),rgb.size())) {
        auto start=std::chrono::steady_clock::now();
        ncnn::Mat x=ncnn::Mat::from_pixels(rgb.data(),ncnn::Mat::PIXEL_RGB,224,224);
        x.substract_mean_normalize(mean,scale);
        auto ex=net.create_extractor();ncnn::Mat g;
        if(ex.input("pixel_values",x)||ex.extract("global_embedding",g))return 17;
        if(g.total()!=768||g.elemsize!=4)return 18;
        const float* p=g;
        for(int i=0;i<768;i++)if(!std::isfinite(p[i]))return 19;
        auto end=std::chrono::steady_clock::now();
        double ms=std::chrono::duration<double,std::milli>(end-start).count();
        std::cout.write(reinterpret_cast<const char*>(p),768*4);
        std::cout.write(reinterpret_cast<const char*>(&ms),sizeof(ms));std::cout.flush();
    }
    return std::cin.eof()?0:20;
}
