/*
 * SingleKernelLauncher.h
 *
 *  Created on: Jul 8, 2015
 *      Author: payne
 *
 * Copyright (c) 2014-2015 Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the  LANL Contributions to Legion (C15091) project.
 * See the LICENSE.txt file at the top-level directory of this distribution.
 */

#ifndef SINGLE_KERNEL_LAUNCHER_H_
#define SINGLE_KERNEL_LAUNCHER_H_
#include "TypeDeduction.h"

namespace Dragon
{
	struct LRArg
	{
	public:

		LRArg(){}
		LRArg(LRWrapper _lr, FieldID _fid,PrivilegeMode _priv,CoherenceProperty _co,RegionFlags _flags):
			lr(_lr),fid(_fid),priv(_priv),co(_co),flags(_flags){}
		LRWrapper lr;
		FieldID fid;
		PrivilegeMode priv;
		CoherenceProperty co;
		RegionFlags flags;
	};

	template<class O>
	class SingleTaskOp
	{
	public:

			DragonOpInfo<O> op_info;
			typedef typename ArgExctractor<decltype(&(O::evaluate_s))>::tList tList;
			typedef typename RetExctractor<decltype(&(O::evaluate_s))>::rT rT;

			static const tList tList_a;

			SingleTaskOp(const O _op) : op_info(_op) {}

			template<class U = std::is_void<rT>>
			typename std::enable_if<!(U::value),rT>::type  execute(const Task *task,Context ctx,
				       const std::vector<PhysicalRegion> &regions, HighLevelRuntime *runtime)
			{

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

				has_on_start_method<O>::eval(op_info.op);

				rT ret_val = evaluate(task,ctx,runtime,0,fids,regions,SingleTaskOp<O>::tList_a.aList);

				has_on_finish_method<O>::eval(op_info.op);


				return ret_val;
			}

			template<class U = std::is_void<rT>>
			typename std::enable_if<(U::value),void>::type  execute(const Task *task,Context ctx,
				       const std::vector<PhysicalRegion> &regions, HighLevelRuntime *runtime)
			{

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

				has_on_start_method<O>::eval(op_info.op);

				evaluate(task,ctx,runtime,0,fids,regions,SingleTaskOp<O>::tList_a.aList);

				has_on_finish_method<O>::eval(op_info.op);


			}

		  template<class rA,template<class>class rAcc,typename... rBcc_cl>
			rT evaluate(const Task *task,Context ctx, HighLevelRuntime* rt,int iLevel,FieldID* fids,
										   const std::vector<PhysicalRegion>& regions,
										   ClassList<rAcc<rA>,rBcc_cl...> accTypes)
			{
			  return evaluate(task,ctx,rt,iLevel+1,fids,regions,ClassList<rBcc_cl...>(),rAcc<rA>(op_info.wrappers[op_info.arg_req_map[iLevel].wrap],regions[op_info.arg_req_map[iLevel].req],fids[iLevel]));
			}

		  template<class rA,template<class>class rAcc,typename... rBcc_cl, typename... rAcc2>
			rT evaluate(const Task *task,Context ctx, HighLevelRuntime* rt,int iLevel,FieldID* fids,
										   const std::vector<PhysicalRegion>& regions,
										   ClassList<rAcc<rA>,rBcc_cl...> accTypes,rAcc2... accessors)
			{
			  return evaluate(task,ctx,rt,iLevel+1,fids,regions,ClassList<rBcc_cl...>(),accessors...,rAcc<rA>(op_info.wrappers[op_info.arg_req_map[iLevel].wrap],regions[op_info.arg_req_map[iLevel].req],fids[iLevel]));
			}

		  template<class rA,template<class>class rAcc,typename... rBcc_cl> __host__
			rT evaluate(const Task *task,Context ctx, HighLevelRuntime* rt,int iLevel,FieldID* fids,
										   const std::vector<PhysicalRegion>& regions,
										   ClassList<rAcc<rA>> a)
			{
			  // We aren't sure what method will be available, so try them at compile time
			  if(find_method<O,decltype(&DummyFoo<rT,rAcc<rA>>)>::value)
				  return find_method<O,decltype(&DummyFoo<rT,rAcc<rA>>)>::eval(op_info.op,rAcc<rA>(op_info.wrappers[op_info.arg_req_map[iLevel].wrap],regions[op_info.arg_req_map[iLevel].req],fids[iLevel]));
			  else if(find_method<O,decltype(&DummyFoo<rT,int,rAcc<rA>>)>::value)
				  return find_method<O,decltype(&DummyFoo<rT,int,rAcc<rA>>)>::eval(op_info.op,task->index_point.point_data[0],
				                                                         rAcc<rA>(op_info.wrappers[op_info.arg_req_map[iLevel].wrap],regions[op_info.arg_req_map[iLevel].req],fids[iLevel]));
			  else if(find_method<O,decltype(&DummyFoo<rT,const Task*,Context,HighLevelRuntime*,rAcc<rA>>)>::value)
				  return find_method<O,decltype(&DummyFoo<rT,const Task*,Context,HighLevelRuntime*,rAcc<rA>>)>::eval(op_info.op,task,ctx,rt,
				                                                         rAcc<rA>(op_info.wrappers[op_info.arg_req_map[iLevel].wrap],regions[op_info.arg_req_map[iLevel].req],fids[iLevel]));
			  else
				  assert(false);

			}

		    template<template<class>class rAcc,class rA,typename... rAcc2> __host__
		    rT evaluate(const Task *task,Context ctx, HighLevelRuntime* rt,int iLevel,FieldID* fids,
		                                   const std::vector<PhysicalRegion>& regions,
		                                   ClassList<rAcc<rA>> a,rAcc2... accessors)
		    {
		    	assert(iLevel < DragonOpInfo<O> ::NARGS);
		    	rAcc<rA> last_acc(op_info.wrappers[op_info.arg_req_map[iLevel].wrap],regions[op_info.arg_req_map[iLevel].req],fids[iLevel]);
				  // We aren't sure what method will be available, so try them at compile time
				  if(find_method<O,decltype(&DummyFoo<rT,rAcc2...,rAcc<rA>>)>::value)
					  return find_method<O,decltype(&DummyFoo<rT,rAcc2...,rAcc<rA>>)>::eval(op_info.op,accessors...,rAcc<rA>(op_info.wrappers[op_info.arg_req_map[iLevel].wrap],regions[op_info.arg_req_map[iLevel].req],fids[iLevel]));
				  else if(find_method<O,decltype(&DummyFoo<rT,int,rAcc2...,rAcc<rA>>)>::value)
					  return find_method<O,decltype(&DummyFoo<rT,int,rAcc2...,rAcc<rA>>)>::eval(op_info.op,task->index_point.point_data[0],
					                                                         accessors...,rAcc<rA>(op_info.wrappers[op_info.arg_req_map[iLevel].wrap],regions[op_info.arg_req_map[iLevel].req],fids[iLevel]));
				  else if(find_method<O,decltype(&DummyFoo<rT,const Task*,Context,HighLevelRuntime*,rAcc2...,rAcc<rA>>)>::value)
					  return find_method<O,decltype(&DummyFoo<rT,const Task*,Context,HighLevelRuntime*,rAcc2...,rAcc<rA>>)>::eval(op_info.op,task,ctx,rt,
					                                                         accessors...,last_acc);
				  else
					  assert(false);
//		  	  return op_info.op.evaluate(task,ctx,rt,accessors...,rAcc<rA>(regions[op_info.arg_req_map[iLevel]],fids[iLevel]));
		    }
	};




	class SingleKernelArgs
	{
	public:
		SingleKernelArgs(){}

		void add_arg(LRWrapper wrapper,FieldID fid,
		             PrivilegeMode priv=READ_WRITE,
		             CoherenceProperty co=EXCLUSIVE,
		             RegionFlags flag=NO_FLAG);
//		{
//			args.push_back(LRArg(wrapper,fid,priv,co,flag));
//		}

		void add_arg(LRWrapper wrapper,FieldID fid,
		             PrivilegeMode priv=READ_WRITE,
		             RegionFlags flag=NO_FLAG);
//		{
//			args.push_back(LRArg(wrapper,fid,priv,EXCLUSIVE,flag));
//		}

		void add_nested(std::vector<RegionRequirement> v_nested);
//		{
//			nested_reqs.insert(nested_reqs.end(),v_nested.begin(),v_nested.end());
//		}

		void add_nested(RegionRequirement v_nested);
//		{
//			nested_reqs.push_back(v_nested);
//		}

		std::vector<LRArg > args;
		std::vector<RegionRequirement> nested_reqs;
	};



	template<class O>
	class SingleKernelLancher : public TaskLauncher{
	public:
//		std::vector<RegionAccessor<AccessorType::Generic>> r_acc;

		O op;

//		DragonOpInfo<O> op_info;

		SingleKernelArgs oargs;
		typedef typename ArgExctractor<decltype(&(O::evaluate_s))>::tList tList;
		typedef typename RetExctractor<decltype(&(O::evaluate_s))>::rT rT;
	//	FieldID w_fid;


		SingleTaskOp<O> task_out;

		static const tList tList_a;



		char* arg_ptr;


		template<class... OpArgList>
		SingleKernelLancher(Context ctx, HighLevelRuntime* rt,const O _op,OpArgList... op_args): op(_op), oargs(op.genArgs(op_args...)),task_out(op),
		TaskLauncher(SingleKernelLancher<O>::TASK_ID(),TaskArgument(arg_ptr,1),Predicate::TRUE_PRED, SingleKernelLancher<O>::MAPPER_ID)
		{


			// Consolidate all of the fields and regions
			typedef typename std::tuple<LRWrapper,PrivilegeMode,CoherenceProperty,RegionFlags> LPMode;
			std::map<LPMode,std::pair<int,std::vector<FieldID>>> req_map;
			std::vector<std::pair<int,LPMode>> arg_req_map;
			std::vector<FieldID> arg_fid_map;
			std::vector<LPMode> lpmode_map;
			std::vector<int> arg_wrap_map;

			std::map<LRWrapper,int> lWrap_map;


			std::vector<LRArg> t_args;

			t_args.insert(t_args.end(),oargs.args.begin(),oargs.args.end());


			int ireq = 0;
			int iArg = 0;
			int iWrap = 0;
			for(auto& it : t_args)
			{
				LRWrapper wrap = it.lr;
				FieldID fid = it.fid;
				PrivilegeMode priv = it.priv;


				arg_fid_map.push_back(fid);
	//			printf("Adding req %i, field %i\n",ireq,fid);

				// Try to find the partition and priviledge combo in our requirement map
				LPMode key(wrap,priv,it.co,it.flags);
	//			auto req_stuff = req_map.find(key);
				if(lWrap_map.find(wrap) == lWrap_map.end())
				{
					lWrap_map[wrap] = iWrap;
					iWrap++;
				}

				arg_wrap_map.push_back( lWrap_map[wrap]);

				// if we find it, append the field id to the end of the list of fids for that req
				if(std::get<1>(req_map[key]).size() != 0)
				{
					arg_req_map.push_back(std::pair<int,LPMode>(std::get<0>(req_map[key]),lpmode_map[std::get<0>(req_map[key])]));
					std::get<1>(req_map[key]).push_back(fid);


				}
				else // Add the region and field id to the map
				{
					lpmode_map.push_back(key);
					arg_req_map.push_back(std::pair<int,LPMode>(ireq,lpmode_map[ireq]));
	//				printf("Adding req %i, field %i in lpmode\n",ireq,fid);
					req_map[key] = std::pair<int,std::vector<FieldID>>(ireq,std::vector<FieldID>({fid}));
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

			assert(arg_req_map.size() == arg_fid_map.size());
			assert(arg_req_map.size() == arg_wrap_map.size());


			ArgMapInfo arg_req_map_out[task_out.op_info.narg];
			for(int i=0;i<task_out.op_info.narg;i++)
			{
				task_out.op_info.arg_req_map[i].req = std::get<0>(arg_req_map[i]);
				task_out.op_info.arg_req_map[i].fid = arg_fid_map[i];
				task_out.op_info.arg_req_map[i].wrap = arg_wrap_map[i];
//				arg_req_map_out[i].req = std::get<0>(arg_req_map[i]);
//				arg_req_map_out[i].fid = arg_fid_map[i];
//				arg_req_map_out[i].wrap = arg_wrap_map[i];
			}

			LRWrapper wrappers_out[task_out.op_info.nwrap];
			for(auto wrap : lWrap_map)
				task_out.op_info.wrappers[wrap.second] = wrap.first;


			size_t arg_size;
			arg_size = sizeof(SingleTaskOp<O>);
			printf("Argsize is %lu\n",arg_size);

//			arg_size += task_out.op_info.narg*sizeof(ArgMapInfo);
//			arg_size += task_out.op_info.nwrap*sizeof(LRWrapper);

			arg_ptr = ((char*)malloc(arg_size));
			argument = TaskArgument(arg_ptr,arg_size);


			for(int i=0;i<lpmode_map.size();i++)
			{

				LPMode part_mode = lpmode_map[i];
				LRWrapper wrap = (std::get<0>(part_mode));
				PrivilegeMode priv = std::get<1>(part_mode);
				CoherenceProperty co = std::get<2>(part_mode);
				RegionFlags flag = std::get<3>(part_mode);


//				printf("getting logical subregion %i %i\n",i,p.p.point_data[0]);
				RegionRequirement rr_out(wrap.lr,priv,co,wrap.lr);
				rr_out.add_fields(std::get<1>(req_map[part_mode]));
				rr_out.add_flags(flag);
				add_region_requirement(rr_out);
			}

			if(oargs.nested_reqs.size() > 0)
			{
				for(auto req : oargs.nested_reqs)
				{
					add_region_requirement(req);
				}
			}


			printf("Finished setting up a field kernel op\n");
			memcpy(arg_ptr,&task_out,sizeof(SingleTaskOp<O>));
//			memcpy(arg_ptr+sizeof(SingleTaskOp<O>),arg_req_map_out,task_out.op_info.narg*sizeof(ArgMapInfo));
//			memcpy(arg_ptr+sizeof(SingleTaskOp<O>)+task_out.op_info.narg*sizeof(ArgMapInfo),wrappers_out,task_out.op_info.nwrap*sizeof(LRWrapper));

		}

	public:
	  bool launch_check_fields(Context ctx, HighLevelRuntime *runtime);
	public:
	  static inline const char* TASK_NAME(){return typeid(SingleKernelLancher<O>).name();};
	  static const TaskID TASK_ID(){return typeid(SingleKernelLancher<O>).hash_code();}
	  static const bool CPU_BASE_LEAF = false;
	  static const bool GPU_BASE_LEAF = false;
	  static const int MAPPER_ID = O::MAPPER_ID;
	  static const int SINGLE = O::SINGLE;
	  static const int INDEX = O::INDEX;
	public:
	  static rT cpu_base_impl(const Task *task,
	                            const std::vector<PhysicalRegion> &regions,
	                            Context ctx, HighLevelRuntime *runtime)

	  {
		  SingleTaskOp<O>* temp = ((SingleTaskOp<O>*)(task->args));
//		  temp->op_info.arg_req_map = (ArgMapInfo*)((char*)(task->args)+sizeof(SingleTaskOp<O>));
//		  temp->op_info.wrappers = (LRWrapper*)(temp->op_info.arg_req_map + temp->op_info.narg);

		  return temp->execute(task,ctx,regions,runtime);

	  }

	  // Register a task that returns a void result
	  template<class U = std::is_void<rT>>
	  static typename std::enable_if<U::value>::type register_cpu(){

		  printf("Setting call for task %lu\n",TASK_ID());

	  HighLevelRuntime::register_legion_task<SingleKernelLancher<O>::cpu_base_impl>(TASK_ID(), Processor::LOC_PROC,
	                                                           SINGLE/*single*/, INDEX/*index*/,
	                                                           0,
	                                                           TaskConfigOptions(CPU_BASE_LEAF),
	                                                           TASK_NAME());
	  }

	  // Register a task that returns a non-void result
	  template<class U = std::is_void<rT>>
	  static typename std::enable_if<!U::value>::type register_cpu(){

		  printf("Setting call for task %lu\n",TASK_ID());

	  HighLevelRuntime::register_legion_task<rT,SingleKernelLancher<O>::cpu_base_impl>(TASK_ID(), Processor::LOC_PROC,
															   SINGLE/*single*/, INDEX/*index*/,
															   0,
															   TaskConfigOptions(CPU_BASE_LEAF),
															   TASK_NAME());
	  }


	};

	template<class O,class... OpArgList>
	SingleKernelLancher<O> genSingleKernel(O op,Context ctx, HighLevelRuntime *rt,OpArgList... oargs)
	{
		return SingleKernelLancher<O>(ctx,rt,op,ctx,rt,oargs...);
	}

	template<class O,class... OpArgList>
	SingleKernelLancher<O> genSingleKernel(Context ctx, HighLevelRuntime *rt,O op,OpArgList... oargs)
	{
		return SingleKernelLancher<O>(ctx,rt,op,oargs...);
	}

} /* namespace Dragon */

#endif /* SINGLE_KERNEL_LAUNCHER_H_ */
