/*
 * IndexKernelLauncher.h
 *
 *  Created on: Jul 13, 2015
 *      Author: payne
 */

#ifndef INDEXKERNELLAUNCHER_H_
#define INDEXKERNELLAUNCHER_H_
#include "TypeDeduction.h"
#include "LPWrapper.h"
#include "LegionMatrix.h"
#include <tuple>

namespace Dragon
{


#ifdef USE_CUDA_FKOP
template<class O,class ... Acc> __global__
void IndexKernelGPUOp(int iStart,int iEnd,O op,Acc... accessors)
{
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
//	printf("Field Kernel GPU RW op with idx %i starting at %i to %i\n",idx,iStart,iEnd);


	for(int i=iStart+idx;i<=iEnd;i+=blockDim.x*gridDim.x)
	{
//		printf("Processing Element %i on GPU\n",i);

			op.evaluate(i,accessors...);
	}
}

template<class O,class rA,class ... Acc> __global__
void IndexKernelGPUOp(int iStart,int iEnd,O op,lrAccessor<rA> result,Acc... accessors)
{
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
//	printf("Field Kernel GPU write op with idx %i starting at %i to %i\n",idx,iStart,iEnd);


	for(int i=iStart+idx;i<=iEnd;i+=blockDim.x*gridDim.x)
	{
//		printf("Processing Element %i on GPU\n",i);

			result(i) = op.evaluate(i,accessors...);

	}
}
#endif

	template<class O>
	class IndexTaskOp
	{
	public:

			DragonOpInfo<O> op_info;
			Domain loop_domain;

			typedef ArgExctractor<decltype(&(O::evaluate_s))> ArgList;
			typedef typename ArgList::tList tList;
			typedef typename ArgList::cuda_capable cuda_capable;
			typedef typename RetExctractor<decltype(&(O::evaluate_s))>::rT rT;

			typedef typename std::integral_constant<bool,std::is_void<rT>::value> rw_impl;

			static const tList tList_a;

			IndexTaskOp(const O _op) : op_info(_op) {}

			template<class gpu_impl>
			void execute(gpu_impl imp,const Task *task,Context ctx,
				       const std::vector<PhysicalRegion> &regions, HighLevelRuntime *rt)
			{

				std::vector<Domain> bounds;

//				rt->get_index_space_domains(ctx,task->regions[0].region.get_index_space(),bounds);
//				if(loop_domain.rect_data[0] == 0 && loop_domain.rect_data[3] == 0)
//				{


//				}
//				else
//					for(int i=0;i<3;i++)
//					{
//						unsigned n = loop_domain.rect_data[i+3] - loop_domain.rect_data[i] +1;
//						bounds.rect_data[i] = loop_domain.rect_data[i] + n*task->index_point.point_data[i];
//						bounds.rect_data[i+3] = loop_domain.rect_data[i+3] + n*task->index_point.point_data[i];
//					}

//				for(auto dom : bounds)
//					printf("Index Op bounds go from %i to %i\n",dom.get_index_space().get_valid_mask().first_enabled_elmt,dom.get_index_space().get_valid_mask().last_enabled_elmt);
				// Ensure that the function signatures are the same for the template
				// and the actual call
				if(!find_method<O,decltype(&O::evaluate_s)>::value)
				{
					fprintf(stderr,"Error, %s::evaluate and %s::evaluate_s have different signatures\n",typeid(O).name(),typeid(O).name());
					assert(false);
				}

				FieldID fids[op_info.narg];
				for(int i=0;i<op_info.narg;i++)
					fids[i] = op_info.arg_req_map[i].fid;

				int iLevel;
				if(std::is_void<rT>::value)
					iLevel = 0;
				else
					iLevel = 1;

//				evaluate(imp,bounds,task,ctx,rt,iLevel,fids,regions,IndexTaskOp<O>::tList_a.aList);
			}

			template<class gpu_impl,class rA,template<class>class rAcc,typename... rBcc_cl>
			void evaluate(gpu_impl imp,const std::vector<Domain>& bounds,const Task *task,Context ctx, HighLevelRuntime* rt,int iLevel,FieldID* fids,
										   const std::vector<PhysicalRegion>& regions,
										   ClassList<rAcc<rA>,rBcc_cl...> accTypes)
			{
				evaluate(imp,bounds,task,ctx,rt,iLevel+1,fids,regions,ClassList<rBcc_cl...>(),rAcc<rA>(op_info.wrappers[op_info.arg_req_map[iLevel].wrap],regions[op_info.arg_req_map[iLevel].req],fids[iLevel]));
			}

			template<class gpu_impl,class rA,template<class>class rAcc,typename... rBcc_cl,typename... rAcc2>
			void evaluate(gpu_impl imp,const std::vector<Domain>& bounds,const Task *task,Context ctx, HighLevelRuntime* rt,int iLevel,FieldID* fids,
										   const std::vector<PhysicalRegion>& regions,
										   ClassList<rAcc<rA>,rBcc_cl...> accTypes,rAcc2... accessors)
			{
				evaluate(imp,bounds,task,ctx,rt,iLevel+1,fids,regions,ClassList<rBcc_cl...>(),accessors...,rAcc<rA>(op_info.wrappers[op_info.arg_req_map[iLevel].wrap],regions[op_info.arg_req_map[iLevel].req],fids[iLevel]));
			}

			template<class gpu_impl,class rA,template<class>class rAcc> __host__
			void evaluate(gpu_impl imp,const std::vector<Domain>& bounds,const Task *task,Context ctx, HighLevelRuntime* rt,int iLevel,FieldID* fids,
										   const std::vector<PhysicalRegion>& regions,
										   ClassList<rAcc<rA>> a)
			{
				rw_picker(imp,bounds,task,ctx,rt,iLevel,fids,regions,
						rAcc<rA>(op_info.wrappers[op_info.arg_req_map[iLevel].wrap],regions[op_info.arg_req_map[iLevel].req],fids[iLevel]));
			}

			template<class gpu_impl,template<class>class rAcc,class rA,typename... rAcc2> __host__
			void evaluate(gpu_impl imp,const std::vector<Domain>& bounds,const Task *task,Context ctx, HighLevelRuntime* rt,int iLevel,FieldID* fids,
										   const std::vector<PhysicalRegion>& regions,
										   ClassList<rAcc<rA>> a,rAcc2... accessors)
			{

				rw_picker(imp,bounds,task,ctx,rt,iLevel,fids,regions,accessors...,rAcc<rA>(op_info.wrappers[op_info.arg_req_map[iLevel].wrap],regions[op_info.arg_req_map[iLevel].req],fids[iLevel]));
			}

			template<class gpu_impl,class... rAcc2,class U = rw_impl> __host__
			typename std::enable_if<U::value>::type
			rw_picker(gpu_impl imp,const std::vector<Domain>& bounds,
			                               const Task *task,Context ctx, HighLevelRuntime* rt,int iLevel,FieldID* fids,
										   const std::vector<PhysicalRegion>& regions,
										   rAcc2... accessors)
			{
				// No external write value
				device_picker(imp,bounds,task,ctx,rt,iLevel,fids,regions,accessors...);
			}


			template<class gpu_impl,class... rAcc2,class U = rw_impl> __host__
			typename std::enable_if<!U::value>::type
			rw_picker(gpu_impl imp,const std::vector<Domain>& bounds,
			                               const Task *task,Context ctx, HighLevelRuntime* rt,int iLevel,FieldID* fids,
										   const std::vector<PhysicalRegion>& regions,
										   rAcc2... accessors)
			{
				// There is an external write value
				device_picker(imp,bounds,task,ctx,rt,iLevel,fids,regions,lrAccessor<rT>(regions[op_info.arg_req_map[0].req],fids[0]),accessors...);

			}

#ifdef USE_CUDA_FKOP

		    // GPU version
		    template<typename... rAcc2,class U = cuda_capable> __host__
		    	typename std::enable_if<U::value>::type device_picker(std::true_type gpu_impl,const std::vector<Domain>& bounds,
		    	                           const Task *task,Context ctx, HighLevelRuntime* rt,int iLevel,FieldID* fids,
		                                   const std::vector<PhysicalRegion>& regions,
		                                   rAcc2... accessors)
		    {
		    	int n = loop_domain.rect_data[1] - loop_domain.rect_data[0] +1;

		    	for(auto dom : bounds)
		    	{
				int iStart = loop_domain.rect_data[0] + n*dom.get_index_space().get_valid_mask().first_enabled_elmt;
				int iEnd = loop_domain.rect_data[1] + n*dom.get_index_space().get_valid_mask().last_enabled_elmt;
		  		int cudaBlockSize = 256;
		  		int cudaGridSize = (iEnd-iStart + cudaBlockSize )/cudaBlockSize;

//		  		printf("Executing Index Kernel on GPU going from %i to %i\n",iStart,iEnd);
//		  		cudaStreamSynchronize(0);
		  		((IndexKernelGPUOp<<<cudaBlockSize,cudaGridSize>>>(iStart,iEnd,op_info.op,accessors...)));
//		  		cudaStreamSynchronize(0);
		    	}
		    }
#endif


		    // CPU version with a return value
		    template<class... rAcc2,class U = rw_impl> __host__
		    typename std::enable_if<!(U::value)>::type device_picker(std::false_type gpu_impl,const std::vector<Domain>& bounds,
		                                   const Task *task,Context ctx, HighLevelRuntime* rt,int iLevel,FieldID* fids,
		                                   const std::vector<PhysicalRegion>& regions,
		                                   lrAccessor<rT> rval,rAcc2... accessors)
		    {
		    	int n = loop_domain.rect_data[1] - loop_domain.rect_data[0] +1;

//		    	printf("Inside Index Kernel with a return value: %s\n",typeid(IndexTaskOp<O>).name());
				// We aren't sure what method will be available, so try them at compile time
				if(find_method<O,decltype(&DummyFoo<rT,rAcc2...>)>::value)
				{
					for(auto dom : bounds)
					for(Domain::DomainPointIterator points(dom);points;points++)
					{
						for(int i=loop_domain.rect_data[0];i<=loop_domain.rect_data[1];i++)
							rval(points.p.point_data[0]) = find_method<O,decltype(&DummyFoo<rT,rAcc2...>)>::eval(op_info.op,accessors...);
					}
				}
				else if(find_method<O,decltype(&DummyFoo<rT,int,rAcc2...>)>::value)
				{
					for(auto dom : bounds)
					for(Domain::DomainPointIterator points(dom);points;points++)
					{
						for(int i=loop_domain.rect_data[0];i<=loop_domain.rect_data[1];i++)
							rval(points.p.point_data[0]) = find_method<O,decltype(&DummyFoo<rT,int,rAcc2...>)>::eval(op_info.op,i+n*points.p.point_data[0],
																		 accessors...);
					}
				}
				else if(find_method<O,decltype(&DummyFoo<rT,const Task*,Context,HighLevelRuntime*,rAcc2...>)>::value)
				{

					  rval(task->index_point.point_data[0]) = find_method<O,decltype(&DummyFoo<rT,const Task*,Context,HighLevelRuntime*,rAcc2...>)>::eval(op_info.op,task,ctx,rt,
																			 accessors...);

				}
				else
				  assert(false);

		    }

		   // CPU version without a return value
		   template<class... rAcc2,class U = rw_impl> __host__
		   typename std::enable_if<U::value>::type device_picker(std::false_type gpu_impl,const std::vector<Domain>& bounds,
		                                   const Task *task,Context ctx, HighLevelRuntime* rt,int iLevel,FieldID* fids,
		                                   const std::vector<PhysicalRegion>& regions,
		                                   rAcc2... accessors)
		    {
		    	int n = loop_domain.rect_data[1] - loop_domain.rect_data[0] +1;

//		    	printf("Inside Index Kernel without a return value: %s\n",typeid(IndexTaskOp<O>).name());

				// We aren't sure what method will be available, so try them at compile time
				if(find_method<O,decltype(&DummyFoo<rT,rAcc2...>)>::value)
				{
					for(auto dom : bounds)
					for(Domain::DomainPointIterator points(dom);points;points++)
					{
						for(int i=loop_domain.rect_data[0];i<=loop_domain.rect_data[1];i++)
							find_method<O,decltype(&DummyFoo<rT,rAcc2...>)>::eval(op_info.op,accessors...);
					}
				}
				else if(find_method<O,decltype(&DummyFoo<rT,int,rAcc2...>)>::value)
				{
					for(auto dom : bounds)
					for(Domain::DomainPointIterator points(dom);points;points++)
					{
						for(int i=loop_domain.rect_data[0];i<=loop_domain.rect_data[1];i++)
							find_method<O,decltype(&DummyFoo<rT,int,rAcc2...>)>::eval(op_info.op,i+n*points.p.point_data[0],
																		 accessors...);
					}
				}
				else if(find_method<O,decltype(&DummyFoo<rT,const Task*,Context,HighLevelRuntime*,rAcc2...>)>::value)
				{

					  find_method<O,decltype(&DummyFoo<rT,const Task*,Context,HighLevelRuntime*,rAcc2...>)>::eval(op_info.op,task,ctx,rt,
																			 accessors...);

				}
				else
				  assert(false);
		    }

	};




	class IndexKernelArgs
	{
	public:
		IndexKernelArgs()
		{
			loop_domain.rect_data[0] = 0;
			loop_domain.rect_data[1] = 0;
		}

		void add_arg(LPWrapper wrapper,FieldID fid,
		             PrivilegeMode priv=READ_WRITE,
		             CoherenceProperty co=EXCLUSIVE,
		             RegionFlags flag=NO_FLAG);

		void add_arg(LPWrapper wrapper,FieldID fid,
		             PrivilegeMode priv=READ_WRITE,
		             RegionFlags flag=NO_FLAG);

		void add_arg(LRWrapper wrapper,FieldID fid,
		             PrivilegeMode priv=READ_WRITE,
		             CoherenceProperty co=EXCLUSIVE,
		             RegionFlags flag=NO_FLAG);

		void add_arg(LRWrapper wrapper,FieldID fid,
		             PrivilegeMode priv=READ_WRITE,
		             RegionFlags flag=NO_FLAG);

		void set_result(LPWrapper wrapper,FieldID fid,
	 	             	             CoherenceProperty co);

		void add_nested(std::vector<RegionRequirement> v_nested);


		void add_nested(RegionRequirement v_nested);

		void add_loop_domain(Domain _dom);

		void add_loop_domain(unsigned iStart, unsigned iEnd);


		std::vector<LPArg > args;
		LPArg res;
		Domain loop_domain;

		std::vector<RegionRequirement> nested_reqs;
	};



	template<class O>
	class IndexKernelLauncher : public IndexLauncher{
	public:
		std::vector<RegionAccessor<AccessorType::Generic>> r_acc;

		O op;

//		DragonOpInfo<O> op_info;

		IndexKernelArgs oargs;
		typedef ArgExctractor<decltype(&(O::evaluate_s))> ArgList;
		typedef typename ArgList::tList tList;
		typedef typename ArgList::cuda_capable cuda_capable;
		typedef typename RetExctractor<decltype(&(O::evaluate_s))>::rT rT;
	//	FieldID w_fid;

//		typedef typename std::false_type cuda_capable;


		static const int NARGS = DragonOpInfo<O>::NARGS;

		IndexTaskOp<O> task_out;

		static const tList tList_a;



		char* arg_ptr;


		struct LPMode
		{
			LPWrapper lp;
			PrivilegeMode priv;
			CoherenceProperty co;
			RegionFlags flags;

			LPMode(LPWrapper _lp,PrivilegeMode _priv,CoherenceProperty _co,RegionFlags _flags) : lp(_lp),priv(_priv),co(_co),flags(_flags) {}

			bool operator<(const LPMode& rhs)
			const
			{
				if(lp < rhs.lp)
					return true;
				else if(priv < rhs.priv)
					return true;
				else if(co < rhs.co)
					return true;
				else if(flags < rhs.flags)
					return true;

				return false;
			}

			bool operator==(const LPMode& rhs)
			const
			{
				return (lp == rhs.lp) && (priv == rhs.priv) && (co == rhs.co) && (flags == rhs.flags);
			}
		};

		template<class... OpArgList>
		IndexKernelLauncher(Context ctx, HighLevelRuntime* rt,const O _op,OpArgList... op_args): op(_op), oargs(op.genArgs(op_args...)),task_out(op),
		IndexLauncher(IndexKernelLauncher<O>::TASK_ID(),
		              Domain(),TaskArgument(&task_out,sizeof(IndexTaskOp<O>)),ArgumentMap(),
		              Predicate::TRUE_PRED, false,O::MAPPER_ID)
		{

			task_out.loop_domain = oargs.loop_domain;


			// Consolidate all of the fields and regions
//			typedef typename std::tuple<LPWrapper,PrivilegeMode,CoherenceProperty,RegionFlags> LPMode;
			std::map<LPMode,std::pair<int,std::set<FieldID>>> req_map;
			std::vector<std::pair<int,LPMode>> arg_req_map;
			std::vector<FieldID> arg_fid_map;
			std::vector<LPMode> lpmode_map;
			std::vector<int> arg_wrap_map;

			std::map<LPWrapper,int> lWrap_map;


			std::vector<LPArg> t_args;

			std::vector<LPWrapper> req_lpWrap_map;


			if(!std::is_void<rT>::value)
				t_args.push_back(oargs.res);

			t_args.insert(t_args.end(),oargs.args.begin(),oargs.args.end());


			launch_domain = t_args[0].lp.lDom;

			int ireq = 0;
			int iArg = 0;
			int iWrap = 0;
			for(auto& it : t_args)
			{
				LPWrapper wrap = it.lp;
				FieldID fid = it.fid;
				PrivilegeMode priv = it.priv;


				arg_fid_map.push_back(fid);
				printf("Arg %i req %i, field %i\n",iArg,ireq,fid);

				// Try to find the partition and priviledge combo in our requirement map
				LPMode key(wrap,priv,it.co,it.flags);
	//			auto req_stuff = req_map.find(key);
				if(lWrap_map.find(wrap) != lWrap_map.end())
				{
					lWrap_map[wrap] = iWrap;
					iWrap++;
				}

				arg_wrap_map.push_back( lWrap_map[wrap]);

				// if we find it, append the field id to the end of the list of fids for that req
				if(std::get<1>(req_map[key]).size() != 0)
				{
					arg_req_map.push_back(std::pair<int,LPMode>(std::get<0>(req_map[key]),lpmode_map[std::get<0>(req_map[key])]));
					std::get<1>(req_map[key]).insert(fid);


				}
				else // Add the region and field id to the map
				{
					lpmode_map.push_back(key);
					arg_req_map.push_back(std::pair<int,LPMode>(ireq,lpmode_map[ireq]));
					printf("Adding req %i, field %i in lpmode\n",ireq,fid);
					req_map[key] = std::pair<int,std::set<FieldID>>(ireq,std::set<FieldID>({fid}));
					req_lpWrap_map.push_back(wrap);
					ireq++;
				}

				iArg++;

			}

			// For debugging
			int i2 = 0;
			for(auto it : arg_req_map)
			{
				printf("Arg %i with fid %i using req %i\n",i2,oargs.args[i2].fid,std::get<0>(it));

				assert(lpmode_map[std::get<0>(it)] == it.second);
				i2++;
			}

			task_out.op_info.narg = arg_req_map.size();
			task_out.op_info.nwrap = lWrap_map.size();

			printf("%i args versus %i compiler args\n",task_out.op_info.narg,NARGS);

			assert(arg_req_map.size() == arg_fid_map.size());
			assert(arg_req_map.size() == arg_wrap_map.size());


//			ArgMapInfo arg_req_map_out[task_out.op_info.narg];
//			for(int i=0;i<task_out.op_info.narg;i++)
//			{
//				arg_req_map_out[i].req = std::get<0>(arg_req_map[i]);
//				arg_req_map_out[i].fid = arg_fid_map[i];
//				arg_req_map_out[i].wrap = arg_wrap_map[i];
//			}
//
//			LRWrapper wrappers_out[task_out.op_info.nwrap];
//			for(auto wrap : lWrap_map)
//				wrappers_out[wrap.second] = wrap.first;

			for(int i=0;i<task_out.op_info.narg;i++)
			{
				task_out.op_info.arg_req_map[i].req = std::get<0>(arg_req_map[i]);
				task_out.op_info.arg_req_map[i].fid = arg_fid_map[i];
				task_out.op_info.arg_req_map[i].wrap = arg_wrap_map[i];
//				arg_req_map_out[i].req = std::get<0>(arg_req_map[i]);
//				arg_req_map_out[i].fid = arg_fid_map[i];
//				arg_req_map_out[i].wrap = arg_wrap_map[i];
			}

			for(auto wrap : lWrap_map)
				task_out.op_info.wrappers[wrap.second] = wrap.first;

			size_t arg_size;
			arg_size = sizeof(IndexTaskOp<O>);
			printf("Argsize is %lu\n",arg_size);
//			arg_size += task_out.op_info.narg*sizeof(ArgMapInfo);
//			arg_size += task_out.op_info.nwrap*sizeof(LRWrapper);

			arg_ptr = ((char*)malloc(arg_size));


			for(int i=0;i<lpmode_map.size();i++)
			{

				LPMode part_mode = lpmode_map[i];
				LPWrapper wrap = part_mode.lp;
				PrivilegeMode priv = part_mode.priv;
				CoherenceProperty co = part_mode.co;
				RegionFlags flag = part_mode.flags;

				printf("adding req %i\n",i);

				if(!(wrap.lDom == launch_domain))
				{
					const char* lr_name;
					rt->retrieve_name(wrap.lp,lr_name);
					printf("Error logical parition color space domain is not equal to the launch domain\n"
							"Requirement %i, Logical Partition %s\n",i,lr_name);
					assert(false);
				}

				RegionRequirement rr_out;
//				if(wrap.single_part)
//					rr_out = RegionRequirement(wrap.lr,priv,co,wrap.lr);
//				else
					rr_out = RegionRequirement(wrap.lp,0,priv,co,wrap.lr);

				rr_out.add_fields(std::vector<FieldID>(std::get<1>(req_map[part_mode]).begin(),std::get<1>(req_map[part_mode]).end()));
//				rr_out.add_flags(flag);
				add_region_requirement(rr_out);
			}

			if(oargs.nested_reqs.size() > 0)
			{
				for(auto& req : oargs.nested_reqs)
				{
					add_region_requirement(req);
				}
			}

			ireq = 0;
			for(auto req : region_requirements)
			{
				const char* lr_name;
				rt->retrieve_name(req.parent,lr_name);
				printf("Index Region requirement %i with lr %s\n",ireq,lr_name);
				ireq++;
			}


			printf("Finished setting up an index kernel op\n");
			memcpy(arg_ptr,&task_out,sizeof(IndexTaskOp<O>));
//			memcpy(arg_ptr+sizeof(IndexTaskOp<O>),arg_req_map_out,task_out.op_info.narg*sizeof(ArgMapInfo));
//			memcpy(arg_ptr+sizeof(IndexTaskOp<O>)+task_out.op_info.narg*sizeof(ArgMapInfo),wrappers_out,task_out.op_info.nwrap*sizeof(LRWrapper));
			global_arg = TaskArgument(arg_ptr,sizeof(IndexTaskOp<O>));

		}

	public:
	  bool launch_check_fields(Context ctx, HighLevelRuntime *runtime);
	public:
	  static inline const char* TASK_NAME(){return typeid(IndexKernelLauncher<O>).name();};
	  static const TaskID TASK_ID(){return typeid(IndexKernelLauncher<O>).hash_code();}
	  static const bool CPU_BASE_LEAF = false;
	  static const bool GPU_BASE_LEAF = true;
	  static const int MAPPER_ID = O::MAPPER_ID;
	  static const int SINGLE = O::SINGLE;
	  static const int INDEX = O::INDEX;
	public:
	  static void cpu_base_impl(const Task *task,
	                            const std::vector<PhysicalRegion> &regions,
	                            Context ctx, HighLevelRuntime *runtime)

	  {
		  IndexTaskOp<O>* temp = ((IndexTaskOp<O>*)(task->args));
//		  temp->op_info.arg_req_map = (ArgMapInfo*)((char*)(task->args)+sizeof(IndexTaskOp<O>));
//		  temp->op_info.wrappers = (LRWrapper*)(temp->op_info.arg_req_map + temp->op_info.narg);

		  temp->execute(std::false_type(),task,ctx,regions,runtime);

	  }

	  template<class U = cuda_capable>
	  static typename std::enable_if<U::value,void>::type gpu_base_impl(const Task *task,
	                                                                               const std::vector<PhysicalRegion> &regions,
	                                                                               Context ctx, HighLevelRuntime *runtime)

	  {
		  IndexTaskOp<O>* temp = ((IndexTaskOp<O>*)(task->args));
//		  temp->op_info.arg_req_map = (ArgMapInfo*)((char*)(task->args)+sizeof(IndexTaskOp<O>));
//		  temp->op_info.wrappers = (LRWrapper*)(temp->op_info.arg_req_map + temp->op_info.narg);

		  temp->execute(cuda_capable(),task,ctx,regions,runtime);

	  }

	  static void register_cpu(){

	  HighLevelRuntime::register_legion_task<IndexKernelLauncher<O>::cpu_base_impl>(TASK_ID(), Processor::LOC_PROC,
	                                                           SINGLE/*single*/, INDEX/*index*/,
	                                                           0,
	                                                           TaskConfigOptions(CPU_BASE_LEAF),
	                                                           TASK_NAME());
	  }

	  template<class U = cuda_capable>
	  static typename std::enable_if<U::value,void>::type register_gpu(){

	  HighLevelRuntime::register_legion_task<IndexKernelLauncher<O>::gpu_base_impl>(TASK_ID(), Processor::TOC_PROC,
	                                                           SINGLE/*single*/, INDEX/*index*/,
	                                                           1,
	                                                           TaskConfigOptions(GPU_BASE_LEAF),
	                                                           TASK_NAME());
	  }

	  template<class U = cuda_capable>
	  static typename std::enable_if<!U::value,void>::type register_gpu(){

	  HighLevelRuntime::register_legion_task<IndexKernelLauncher<O>::cpu_base_impl>(TASK_ID(), Processor::LOC_PROC,
	                                                           SINGLE/*single*/, INDEX/*index*/,
	                                                           0,
	                                                           TaskConfigOptions(CPU_BASE_LEAF),
	                                                           TASK_NAME());
	  }

	};

	template<class O,class... OpArgList>
	IndexKernelLauncher<O>& genIndexKernel(O op,Context ctx, HighLevelRuntime *rt,OpArgList... oargs)
	{
		return *(new IndexKernelLauncher<O>(ctx,rt,op,ctx,rt,oargs...));
	}

	template<class O,class... OpArgList>
	IndexKernelLauncher<O>& genIndexKernel(Context ctx, HighLevelRuntime *rt,O op,OpArgList... oargs)
	{
		return *(new IndexKernelLauncher<O>(ctx,rt,op,oargs...));
	}

} /* namespace Dragon */

#endif /* INDEXKERNELLAUNCHER_H_ */
