/*
 * LegionHelper.h
 *
 *  Created on: Jul 9, 2015
 *      Author: payne
 */

#ifndef LEGIONHELPER_H_
#define LEGIONHELPER_H_

#include <legion.h>
#include <typeinfo>
#ifdef USE_CUDA_FKOP
#include <cuda.h>
#include <cuda_runtime.h>
#else
#define __host__
#define __device__
#endif

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

class LegionHelper
{
public:

	Context &ctx;
	HighLevelRuntime *const runtime;

	LegionHelper(Context& _ctx, HighLevelRuntime *const _runtime) : ctx(_ctx),runtime(_runtime) {}



	template<class T>
	class Setter : public TaskLauncher
	{
	public:
	  static const char * const TASK_NAME(){return typeid(ReturnStorage<T>).name();};
	  static const int TASK_ID(){return typeid(ReturnStorage<T>).hash_code();}
	  static const bool CPU_BASE_LEAF = false;
	  static const bool GPU_BASE_LEAF = false;
	  static const int MAPPER_ID = 0;
	  static const bool SINGLE = true;
	  static const bool INDEX = false;
	public:
	  typedef void rT;

		Setter(LogicalRegion& lr,FieldID fid,T* array, int n) :
			TaskLauncher(TASK_ID(),TaskArgument(array,n*sizeof(T)))
		{
			RegionRequirement rr(lr,WRITE_ONLY,EXCLUSIVE,lr);
			rr.add_field(fid);

			add_region_requirement(rr);
		}

		 static void cpu_base_impl(const Task *task,Context ctx,
		                            const std::vector<PhysicalRegion> &regions, HighLevelRuntime *runtime)

		  {
			 T* array = (T*)task->args;
			 int n = task->arglen/sizeof(T);
			 FieldID fid = task->regions[0].instance_fields[0];

			RegionAccessor<AccessorType::Generic,T> accessor =
					regions[0].get_field_accessor(fid).typeify<T>();

//			IndexIterator iter(task->regions[0].region);

			for(unsigned i=0;i<n;i++)
					accessor.write(ptr_t(i),array[i]);


		  }
	};

	template<class T>
	class ReturnStorage
	{
	public:
		T vals[MAX_RETURN_SIZE/sizeof(T)];
	};

	template<class T>
	class Getter : public IndexLauncher
	{
	public:
	  static const char * const TASK_NAME(){return typeid(Getter<T>).name();};
	  static const int TASK_ID(){return typeid(Getter<T>).hash_code();}
	  static const bool CPU_BASE_LEAF = false;
	  static const bool GPU_BASE_LEAF = false;
	  static const int MAPPER_ID = 0;
	  static const bool SINGLE = false;
	  static const bool INDEX = true;
	public:
	  typedef ReturnStorage<T> rT;

	  Getter(LogicalPartition& lp,LogicalRegion parent,FieldID fid,Domain dom,
	         ArgumentMap& argmap,const unsigned& n) :
		  IndexLauncher(TASK_ID(),dom,TaskArgument(&n,sizeof(unsigned)),argmap)
		{
			RegionRequirement rr(lp,0,READ_WRITE,EXCLUSIVE,parent);
			rr.add_field(fid);

			add_region_requirement(rr);
		}

		 static ReturnStorage<T> cpu_base_impl(const Task *task,Context ctx,
		                            const std::vector<PhysicalRegion> &regions, HighLevelRuntime *runtime)

		  {
			 unsigned ndo = ((unsigned*)task->local_args)[0];
			 FieldID fid = task->regions[0].instance_fields[0];
			 ReturnStorage<T> tmp;

			RegionAccessor<AccessorType::Generic,T> accessor =
					regions[0].get_field_accessor(fid).typeify<T>();

			IndexIterator iter(runtime,ctx,task->regions[0].region);

			for(int i=0;i<ndo;i++)
				if(iter.has_next())
					tmp.vals[i] = accessor.read(iter.next());
				else
				{printf("Warning array dimension mismatch, filled %i out of %i\n",i,ndo);break;}
			return tmp;
		  }
	};

	template<class T>
	void set(LogicalRegion& lr,FieldID fid,T* array, int n)
	{

		Setter<T> setter(lr,fid,array,n);

		runtime->execute_task(ctx,setter);

//		PhysicalRegion pr = runtime->map_region(ctx,rr);
//		{
//		RegionAccessor<AccessorType::Generic,T> accessor =
//				pr.get_field_accessor(fid).typeify<T>();
//
//		IndexIterator iter(lr);
//
//		for(int i=0;i<n;i++)
//			if(iter.has_next())
//				accessor.write(iter.next(),array[i]);
//			else
//			{printf("Warning array dimension mismatch, filled %i out of %i\n",i,n);break;}
//
//		}
//		runtime->unmap_region(ctx,pr);
	}

	template<class T>
	void set(T* array,LogicalRegion& lr,FieldID fid,const unsigned n)
	{
//		Coloring cl;
//		ArgumentMap argmap;
//		Domain dom;
//		IndexPartition ip;
//		LogicalPartition lp;
//		unsigned i_cl = 0;
//		unsigned it=0;
//		unsigned nmax = std::min(n,(unsigned)(MAX_RETURN_SIZE/sizeof(T)));
//		unsigned n_dos[(nmax+n-1)/n];
//		while(it < n)
//		{
//			unsigned nstop = std::min(n,it+nmax);
//			n_dos[i_cl] = nstop;
//			DomainPoint pt = DomainPoint::from_point<1>(Point<1>(i_cl));
//			argmap.set_point(pt,TaskArgument(n_dos+i_cl,sizeof(unsigned)));
//			cl[i_cl].ranges.insert(std::pair<ptr_t,ptr_t>(it,nstop-1));
//
//			i_cl++;
//			it+=nmax;
//		}
//		ip = runtime->create_index_partition(ctx,lr.get_index_space(),cl,true);
//		dom = runtime->get_index_partition_color_space(ctx,ip);
//		lp = runtime->get_logical_partition(ctx,lr,ip);
//
//		Getter<T> getter(lp,lr,fid,dom,argmap,n);
//
//		FutureMap mp = runtime->execute_index_space(ctx,getter);
//
//		unsigned n_cl = cl.size();
//		i_cl = 0;
//		for(Domain::DomainPointIterator p(dom);p;p++)
//		{
////			DomainPoint pt = DomainPoint::from_point<1>(Point<1>(i_cl));
//			ReturnStorage<T> tmp = mp.get_result<ReturnStorage<T>>(p.p);
//
//			memcpy(array+n_cl*nmax,tmp.vals,n_dos[i_cl]*sizeof(T));
//			i_cl++;
//		}
//
//		for(int i=0;i<n;i++)
//			printf("Array[%i] = %i\n",i,array[i]);

//		runtime->disable_profiling();

		RegionRequirement rr(lr,READ_ONLY,EXCLUSIVE,lr);
		rr.add_field(fid);
		PhysicalRegion pr = runtime->map_region(ctx,rr);
		{
		RegionAccessor<AccessorType::Generic,T> accessor =
				pr.get_field_accessor(fid).typeify<T>();

		IndexIterator iter(runtime,ctx,lr);

		for(int i=0;i<n;i++)
			if(iter.has_next())
			{
				array[i] = accessor.read(iter.next());
				printf("Array[%i] = %i\n",i,array[i]);
			}
			else
			{printf("Warning array dimension mismatch, filled %i out of %i\n",i,n);break;}
		}
		runtime->unmap_region(ctx,pr);

//		runtime->enable_profiling();

	}

	template<class T>
	void copy(LogicalRegion& dst,FieldID dst_fid,LogicalRegion& src,FieldID src_fid,int n)
	{
		RegionRequirement rr_dst(dst,WRITE_ONLY,EXCLUSIVE,dst);
		rr_dst.add_field(dst_fid);
		PhysicalRegion pr_dst = runtime->map_region(ctx,rr_dst);

		RegionRequirement rr_src(src,READ_ONLY,EXCLUSIVE,src);
		rr_src.add_field(src_fid);
		PhysicalRegion pr_src = runtime->map_region(ctx,rr_src);
		{
		RegionAccessor<AccessorType::Generic,T> acc_dst =
				pr_dst.get_field_accessor(dst_fid).typeify<T>();

		RegionAccessor<AccessorType::Generic,T> acc_src =
				pr_src.get_field_accessor(src_fid).typeify<T>();

		IndexIterator iter_dst(runtime,ctx,dst);
		IndexIterator iter_src(runtime,ctx,src);


		for(int i=0;i<n;i++)
			if(iter_dst.has_next() && iter_src.has_next())
				acc_dst.write(iter_dst.next(),acc_src.read(iter_src.next()));
			else
			{printf("Warning array dimension mismatch, filled %i out of %i\n",i,n);break;}
		}
		runtime->unmap_region(ctx,pr_dst);
		runtime->unmap_region(ctx,pr_src);


//		CopyLauncher copy_launcher;
//		RegionRequirement rr_src(src, READ_ONLY,
//									EXCLUSIVE, src);
//		rr_src.add_field(src_fid);
//
//		 RegionRequirement rr_dst(dst, READ_WRITE,
//									EXCLUSIVE, dst);
//		 rr_dst.add_field(dst_fid);
//		HighLevelRuntime* rt = HighLevelRuntime::get_runtime(Processor::get_executing_processor());
//		PhaseBarrier tmp = rt->create_phase_barrier(_ctx,1);
//		copy_launcher.add_copy_requirements(rr_src,rr_dst);
//		copy_launcher.add_arrival_barrier(tmp);
//
//		rt->issue_copy_operation(_ctx, copy_launcher);
////		rt->advance_phase_barrier(_ctx,tmp);
//		tmp.wait();
//
//		rt->destroy_phase_barrier(_ctx,tmp);

	}
};

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
  template <class T>
  class RetExctractor2
  {
  public:
  	typedef void rT;

  };
  template<class rVal,class... args>
  class RetExctractor2<rVal(*)(int,args...)>
  {
  public:
  	typedef rVal rT;
  };
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

template<class T>
void register_cpu_variants(void)
{

  typedef typename T::rT rT;
  HighLevelRuntime::register_legion_task<rT,base_cpu_wrapper<rT,T> >(T::TASK_ID(), Processor::LOC_PROC,
                                                           T::SINGLE/*single*/, T::INDEX/*index*/,
                                                           0,
                                                           TaskConfigOptions(T::CPU_BASE_LEAF),
                                                           T::TASK_NAME());
  printf("Setting call for task %lu\n",T::TASK_ID());


}


template<class T>
void register_hybrid_variants(void)
{
//	  typedef typename RetExctractor2<decltype(&(T::cpu_base_impl))>::rT rT;

		  HighLevelRuntime::register_legion_task<v_base_cpu_wrapper<T> >(T::TASK_ID(), Processor::LOC_PROC,
                                                               T::SINGLE/*single*/, T::INDEX/*index*/,
                                                               0,
                                                               TaskConfigOptions(T::CPU_BASE_LEAF),
                                                               T::TASK_NAME());


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



#endif /* LEGIONHELPER_H_ */
