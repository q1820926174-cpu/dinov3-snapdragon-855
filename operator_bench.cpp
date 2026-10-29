#include <net.h>
#include <gpu.h>
#include <command.h>
#include <layer.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point s){return std::chrono::duration<double,std::milli>(Clock::now()-s).count();}
int main(int argc,char**argv){
 if(argc!=3)return 64;
 std::string base=argv[1],backend=argv[2];bool gpu=backend!="cpu",half=backend=="gpu16s";
 setenv("SCV41_NCNN_STRICT_GPU","1",1);
 if(gpu){ncnn::create_gpu_instance();if(ncnn::get_gpu_count()<1)return 2;}
 std::ifstream cases(base+"/cases.txt");std::ofstream result(base+"/"+backend+".csv");
 result<<"case,backend,iteration,milliseconds,mac\n";
 std::string name;int ci,hi,wi,co,ho,wo;long long mac;
 while(cases>>name>>ci>>hi>>wi>>co>>ho>>wo>>mac){
  ncnn::Net net;net.opt.num_threads=4;net.opt.use_vulkan_compute=gpu;
  net.opt.use_fp16_packed=half;net.opt.use_fp16_storage=half;net.opt.use_fp16_arithmetic=false;net.opt.use_bf16_storage=false;
  ncnn::VkAllocator* blob=nullptr; ncnn::VkAllocator* staging=nullptr;
  if(gpu){net.set_vulkan_device(0);blob=net.vulkan_device()->acquire_blob_allocator();staging=net.vulkan_device()->acquire_staging_allocator();
   net.opt.blob_vkallocator=blob;net.opt.workspace_vkallocator=blob;net.opt.staging_vkallocator=staging;}
  auto prefix=base+"/"+name+"/model";
  if(net.load_param((prefix+".param").c_str())||net.load_model((prefix+".bin").c_str()))return 3;
  if(gpu){if(!net.opt.use_vulkan_compute)return 4;for(auto l:net.layers())if(!l->support_vulkan)return 5;}
  ncnn::Mat input(wi,hi,ci);std::ifstream f(base+"/"+name+"/input.raw",std::ios::binary);
  for(int c=0;c<ci;c++){float* p=input.channel(c);f.read(reinterpret_cast<char*>(p),wi*hi*4);if(!f)return 6;}
  ncnn::VkMat resident;
  if(gpu){ncnn::VkCompute upload(net.vulkan_device());upload.record_upload(input,resident,net.opt);if(upload.submit_and_wait())return 7;}
  ncnn::Mat actual;
  for(int i=-5;i<20;i++){
   auto ex=net.create_extractor();auto start=Clock::now();int status=0;
   if(gpu){
    ncnn::VkMat output;ncnn::VkCompute cmd(net.vulkan_device());
    status=ex.input("input",resident);if(!status)status=ex.extract("output",output,cmd);if(!status)status=cmd.submit_and_wait();
    double elapsed=ms(start);if(status)return 8;
    if(i>=0)result<<name<<','<<backend<<','<<i<<','<<elapsed<<','<<mac<<'\n';
    if(i==19&&ex.extract("output",actual))return 9;
   }else{
    status=ex.input("input",input);if(!status)status=ex.extract("output",actual);
    double elapsed=ms(start);if(status)return 10;
    if(i>=0)result<<name<<','<<backend<<','<<i<<','<<elapsed<<','<<mac<<'\n';
   }
  }
  if(actual.w!=wo||actual.h!=ho||actual.c!=co||actual.elemsize!=4)return 11;
  std::ofstream raw(base+"/"+name+"/"+backend+"_output.raw",std::ios::binary);
  for(int c=0;c<co;c++){
   const float* data=actual.channel(c);for(int i=0;i<wo*ho;i++)if(!std::isfinite(data[i]))return 12;
   raw.write(reinterpret_cast<const char*>(data),wo*ho*4);
  }
  result.flush();std::printf("OPERATOR_COMPLETE %s %s\n",name.c_str(),backend.c_str());std::fflush(stdout);
  if(gpu){auto dev=net.vulkan_device();actual.release();resident.release();net.clear();dev->reclaim_blob_allocator(blob);dev->reclaim_staging_allocator(staging);}
 }
 return 0;
}
