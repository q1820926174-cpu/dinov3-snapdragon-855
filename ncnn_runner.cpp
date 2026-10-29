#include <net.h>
#include <gpu.h>
#include <layer.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int main(int argc,char** argv) {
    if(argc!=5) { std::fprintf(stderr,"usage: runner model_prefix cpu|gpu input_list output_prefix\n");return 64; }
    bool gpu=std::string(argv[2])!="cpu";
    bool half=std::string(argv[2])=="gpu" || std::string(argv[2])=="gpu16s";
    int iw=224,ih=224,ic=3;
    if(const char* dims=std::getenv("DINO_INPUT_SHAPE")) {
        if(std::sscanf(dims,"%d,%d,%d",&iw,&ih,&ic)!=3||iw<1||ih<1||ic<1)return 65;
    }
    ncnn::Net net;
    net.opt.num_threads=4;
    net.opt.use_vulkan_compute=gpu;
    net.opt.use_fp16_packed=half;
    net.opt.use_fp16_storage=half;
    net.opt.use_fp16_arithmetic=half && std::string(argv[2])!="gpu16s";
    net.opt.use_bf16_storage=false;
    if(gpu) {
        setenv("SCV41_NCNN_STRICT_GPU","1",1);
        ncnn::create_gpu_instance();
        if(ncnn::get_gpu_count()<1) return 10;
        net.set_vulkan_device(0);
    }
    std::string prefix=argv[1];
    if(net.load_param((prefix+".param").c_str()) || net.load_model((prefix+".bin").c_str())) return 11;
    int unsupported=0;
    if(gpu) for(auto layer:net.layers()) if(!layer->support_vulkan) {
        std::fprintf(stderr,"UNSUPPORTED_GPU_LAYER %s %s\n",layer->name.c_str(),layer->type.c_str());unsupported++;
    }
    if(unsupported) return 12;
    if(gpu && !net.opt.use_vulkan_compute) return 13;
    std::ifstream list(argv[3]);std::string path;
    std::ifstream rgb_list;
    if(const char* file=std::getenv("DINO_RGB_LIST"))rgb_list.open(file);
    bool correction=std::getenv("DINO_RGB_LIST")!=nullptr;
    std::string out=argv[4];
    std::FILE* global=std::fopen((out+"_global.raw").c_str(),"wb");
    std::FILE* dense=std::fopen((out+"_dense.raw").c_str(),"wb");
    std::FILE* timings=std::fopen((out+"_timings.csv").c_str(),"w");
    if(!list||!global||!dense||!timings) return 14;
    std::fprintf(timings,"index,milliseconds\n");
    int index=0;
    while(std::getline(list,path)) {
        if(path.empty()) continue;
        std::vector<float> raw(size_t(iw)*ih*ic);
        std::ifstream file(path,std::ios::binary);
        if(!file.read(reinterpret_cast<char*>(raw.data()),raw.size()*sizeof(float)))return 15;
        ncnn::Mat x(iw,ih,ic);
        for(int c=0;c<ic;c++) {
            float* p=x.channel(c);
            for(int i=0;i<iw*ih;i++)p[i]=raw[i*ic+c];
        }
        ncnn::Mat original;
        if(correction) {
            std::string rgb_path;if(!std::getline(rgb_list,rgb_path))return 66;
            std::vector<float> rgb(224*224*3);std::ifstream f(rgb_path,std::ios::binary);
            if(!f.read(reinterpret_cast<char*>(rgb.data()),rgb.size()*sizeof(float)))return 67;
            original.create(224,224,3);
            for(int c=0;c<3;c++){float* dst=original.channel(c);for(int i=0;i<224*224;i++)dst[i]=rgb[i*3+c];}
        }
        if(std::getenv("DINO_DIAG")) {
            auto probe=net.create_extractor();probe.set_light_mode(false);
            if(probe.input("pixel_values",x))return 23;
            ncnn::Net reference;
            reference.opt.num_threads=4;
            reference.opt.use_fp16_packed=false;reference.opt.use_fp16_storage=false;
            reference.opt.use_fp16_arithmetic=false;reference.opt.use_bf16_storage=false;
            if(reference.load_param((prefix+".param").c_str())||reference.load_model((prefix+".bin").c_str()))return 26;
            auto ref=reference.create_extractor();ref.set_light_mode(false);ref.input("pixel_values",x);
            for(auto layer:net.layers()) for(int top:layer->tops) {
                ncnn::Mat value;
                const auto& name=net.blobs()[top].name;
                if(probe.extract(name.c_str(),value))return 24;
                if(value.elemsize!=4)continue;
                float lo=INFINITY,hi=-INFINITY;
                for(int c=0;c<value.c;c++) {
                    const float* data=value.channel(c);
                    for(int j=0;j<value.w*value.h*value.d;j++) {
                        if(!std::isfinite(data[j])) {
                            std::fprintf(stderr,"DIAG_NONFINITE layer=%s type=%s output=%s index=%d/%d\n",layer->name.c_str(),layer->type.c_str(),name.c_str(),c,j);return 25;
                        }
                        lo=std::min(lo,data[j]);hi=std::max(hi,data[j]);
                    }
                }
                ncnn::Mat expected;
                if(ref.extract(name.c_str(),expected))return 27;
                if(value.w!=expected.w||value.h!=expected.h||value.c!=expected.c||value.d!=expected.d)return 28;
                double ab=0,aa=0,bb=0,ee=0;
                for(int c=0;c<value.c;c++) {
                    const float* a=value.channel(c);const float* b=expected.channel(c);
                    for(int j=0;j<value.w*value.h*value.d;j++) {
                        ab+=double(a[j])*b[j];aa+=double(a[j])*a[j];bb+=double(b[j])*b[j];
                        ee+=double(a[j]-b[j])*(a[j]-b[j]);
                    }
                }
                double cs=ab/std::sqrt(aa*bb+1e-100),relative=std::sqrt(ee/(bb+1e-100));
                std::printf("DIAG_FINITE %s %.6g %.6g cosine=%.8f relrms=%.8f\n",layer->name.c_str(),lo,hi,cs,relative);std::fflush(stdout);
                if(cs<.995 && bb>1e-12 && aa>1e-12)return 29;
            }
            return 0;
        }
        // First input also warms the exact model; this is excluded from measured rows.
        if(index==0) for(int k=0;k<3;k++) {
            auto warm=net.create_extractor();ncnn::Mat y;
            if(warm.input("pixel_values",x)||(correction&&warm.input("original_pixels",original))||warm.extract("global_embedding",y))return 16;
        }
        auto start=std::chrono::steady_clock::now();
        auto ex=net.create_extractor();ncnn::Mat g,d;
        if(ex.input("pixel_values",x)||(correction&&ex.input("original_pixels",original))||ex.extract("global_embedding",g)||ex.extract("dense_features",d))return 17;
        auto end=std::chrono::steady_clock::now();
        if(g.w*g.h*g.c!=768 || d.w*d.h*d.c!=768*7*7 || g.elemsize!=4 || d.elemsize!=4) {
            std::fprintf(stderr,"unexpected outputs global=%zu dense=%zu sizes=%zu/%zu\n",g.total(),d.total(),g.elemsize,d.elemsize);return 18;
        }
        const float* gp=g;
        for(size_t i=0;i<g.total();i++) if(!std::isfinite(gp[i]))return 19;
        std::fwrite(g.data,4,768,global);
        // Mat channels can be padded to cstep: write each actual 7x7 plane.
        if(d.dims!=3||d.c!=768||d.w!=7||d.h!=7)return 20;
        for(int c=0;c<d.c;c++) {
            const float* dp=d.channel(c);
            for(int i=0;i<49;i++)if(!std::isfinite(dp[i]))return 21;
            std::fwrite(dp,4,49,dense);
        }
        double ms=std::chrono::duration<double,std::milli>(end-start).count();
        std::fprintf(timings,"%d,%.6f\n",index,ms);std::fflush(timings);
        std::printf("sample %d milliseconds %.3f backend %s\n",index,ms,gpu?(half?(net.opt.use_fp16_arithmetic?"Vulkan_FP16":"Vulkan_FP16_storage_FP32_math"):"Vulkan_FP32"):"CPU_FP32");std::fflush(stdout);
        index++;
    }
    std::fclose(global);std::fclose(dense);std::fclose(timings);
    return index?0:22;
}
