/*
 * TaskHelper.h
 *
 *  Created on: Jul 9, 2015
 *      Author: payne
 */

#ifndef TASKHELPER_H_
#define TASKHELPER_H_

#include <legion.h>
#include <typeinfo>
#ifdef USE_CUDA_FKOP
#include <cuda.h>
#include <cuda_runtime.h>
#else
#define __host__
#define __device__
#endif

//#include <math.h>
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

namespace TaskHelper {
  template<typename T>
  void dispatch_task(T &launcher, Context ctx, HighLevelRuntime *runtime,
                     bool perform_checks, bool &simulation_success, bool wait = false)
  {
		printf("Dsipatching task %s\n",T::TASK_NAME());
	if(T::INDEX)
	{
		FutureMap fm = runtime->execute_index_space(ctx, (IndexLauncher&)launcher);
		if (wait)
		  fm.wait_all_results();
	}
	else
	{
		Future fm = runtime->execute_task(ctx, (TaskLauncher&)launcher);
		if (wait)
		  fm.get_void_result();
	}
    if (simulation_success && perform_checks)
    {
      if (!simulation_success)
        printf("WARNING: First NaN values found in %s\n", T::TASK_NAME());
    }
  }
//  template <class T>
//  class RetExctractor2
//  {
//  public:
//  	typedef void rT;
//
//  };
//  template<class rVal,class... args>
//  class RetExctractor2<rVal(*)(int,args...)>
//  {
//  public:
//  	typedef rVal rT;
//  };
template<class rT,class T>
rT   base_cpu_wrapper(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, HighLevelRuntime *runtime)
{
//	T* base_class = (T*)task->args;
	return T::cpu_base_impl(task,ctx, regions,runtime);
}

template<class T>
void v_base_cpu_wrapper(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, HighLevelRuntime *runtime)
{
//	T* base_class = (T*)task->args;
	T::cpu_base_impl(task,ctx, regions,runtime);
}


template<class T>
void  base_gpu_wrapper(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, HighLevelRuntime *runtime)
{
//	T* base_class = (T*)task->args;
	T::gpu_base_impl(task,ctx, regions,runtime);
}

template<class rT,class T>
class register_cpu_variants_rt
{
public:

  static void register_cpu(){

	  printf("registering task id %u\n",T::TASK_ID());
  HighLevelRuntime::register_legion_task<rT,base_cpu_wrapper<rT,T> >(T::TASK_ID(), Processor::LOC_PROC,
                                                           T::SINGLE/*single*/, T::INDEX/*index*/,
                                                           0,
                                                           TaskConfigOptions(T::CPU_BASE_LEAF),
                                                           T::TASK_NAME());
  }
};

template<class T>
class register_cpu_variants_rt<void,T>
{
public:

  static void register_cpu(){
	  printf("registering void task id %u\n",T::TASK_ID());

  HighLevelRuntime::register_legion_task<v_base_cpu_wrapper<T> >(T::TASK_ID(), Processor::LOC_PROC,
                                                           T::SINGLE/*single*/, T::INDEX/*index*/,
                                                           0,
                                                           TaskConfigOptions(T::CPU_BASE_LEAF),
                                                           T::TASK_NAME());
  }
};

template<class T>
void register_cpu_variants(void)
{

  typedef typename T::rT rT;

  register_cpu_variants_rt<rT,T>::register_cpu();
//  HighLevelRuntime::register_legion_task<rT,base_cpu_wrapper<rT,T> >(T::TASK_ID(), Processor::LOC_PROC,
//                                                           T::SINGLE/*single*/, T::INDEX/*index*/,
//                                                           0,
//                                                           TaskConfigOptions(T::CPU_BASE_LEAF),
//                                                           T::TASK_NAME());


}


template<class T>
void register_hybrid_variants(void)
{

	  typedef typename T::rT rT;

	  register_cpu_variants_rt<rT,T>::register_cpu();
//	  typedef typename RetExctractor2<decltype(&(T::cpu_base_impl))>::rT rT;
//
//		  HighLevelRuntime::register_legion_task<v_base_cpu_wrapper<T> >(T::TASK_ID(), Processor::LOC_PROC,
//                                                               T::SINGLE/*single*/, T::INDEX/*index*/,
//                                                               0,
//                                                               TaskConfigOptions(T::CPU_BASE_LEAF),
//                                                               T::TASK_NAME());


  printf("Setting call for task %lu\n",T::TASK_ID());
//#ifdef USE_CUDA
//  HighLevelRuntime::register_legion_task<base_gpu_wrapper<T> >(T::TASK_ID(), Processor::TOC_PROC,
//                                                               T::SINGLE/*single*/, T::INDEX/*index*/,
//                                                               CIRCUIT_GPU_LEAF_VARIANT,
//                                                               TaskConfigOptions(T::GPU_BASE_LEAF),
//                                                               T::TASK_NAME());
//#endif
}



}


#endif /* TASKHELPER_H_ */
