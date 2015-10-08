/*
 * MustEpochKernelLauncher.h
 *
 *  Created on: Jul 23, 2015
 *      Author: payne
 *
 * Copyright (c) 2014-2015 Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the  LANL Contributions to Legion (C15091) project.
 * See the LICENSE.txt file at the top-level directory of this distribution.
 */

#ifndef MUSTEPOCHKERNELLAUNCHER_H_
#define MUSTEPOCHKERNELLAUNCHER_H_


#include "TypeDeduction.h"
#include "SingleKernelLauncher.h"
#include "LPWrapper.h"
#include "LegionMatrix.h"
#include <tuple>

namespace Dragon
{

	class EpochKernelArgs
	{
	public:


		void add_arg(LPWrapper wrapper,FieldID fid,
		             PrivilegeMode priv=READ_WRITE,
		             CoherenceProperty co=EXCLUSIVE,
		             RegionFlags flag=NO_FLAG);

		void add_arg(LPWrapper wrapper,FieldID fid,
		             PrivilegeMode priv=READ_WRITE,
		             RegionFlags flag=NO_FLAG);

//		void add_arg(LRWrapper wrapper,FieldID fid,
//		             PrivilegeMode priv=READ_WRITE,
//		             CoherenceProperty co=EXCLUSIVE,
//		             RegionFlags flag=NO_FLAG);
//
//		void add_arg(LRWrapper wrapper,FieldID fid,
//		             PrivilegeMode priv=READ_WRITE,
//		             RegionFlags flag=NO_FLAG);
//
//		void set_result(LPWrapper wrapper,FieldID fid,
//	 	             	             CoherenceProperty co);

		void add_nested(Color cl,std::vector<RegionRequirement> v_nested);


		void add_nested(Color cl,RegionRequirement v_nested);

		void add_loop_domain(Domain _dom);

		void add_loop_domain(unsigned iStart, unsigned iEnd);



		std::vector<LPArg > args;

		std::map<Color,std::vector<RegionRequirement>> nested_reqs;

	};

	class MEKLInterface: public MustEpochLauncher
	{
	public:
		std::map<Color,TaskLauncher> task_map;

		void execute_tasks(Context ctx, HighLevelRuntime* rt,LRWrapper& predicate)
		{
//			RegionRequirement rr(predicate.lr,READ_ONLY,EXCLUSIVE,predicate.lr);
//			rr.add_field(0);
//			PhysicalRegion pr = rt->map_region(ctx,rr);
//
//			LegionMatrix<bool> pred(predicate,pr,0);
			MustEpochLauncher launcher;
			  for(auto& task : single_tasks)
			  {
//				  single_tasks.push_back(task.second);
//				  if(pred(task.point.point_data[0]))
					  rt->execute_task(ctx,task);
			  }

//			  rt->execute_must_epoch(ctx,launcher);

//			  rt->unmap_region(ctx,pr);
		}
	};


	template<class O>
	class MustEpochKernelLauncher : public MEKLInterface{
	public:

		O op;

//		DragonOpInfo<O> op_info;

		EpochKernelArgs oargs;
		typedef ArgExctractor<decltype(&(O::evaluate_s))> ArgList;
		typedef typename ArgList::tList tList;
		typedef typename ArgList::cuda_capable cuda_capable;
		typedef typename RetExctractor<decltype(&(O::evaluate_s))>::rT rT;
	//	FieldID w_fid;

//		typedef typename std::false_type cuda_capable;


		static const int NARGS = DragonOpInfo<O>::NARGS;

		SingleTaskOp<O> task_out;

		static const tList tList_a;



		Domain launch_domain;

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
				if(!(lp == rhs.lp))
					if(lp < rhs.lp)
						return true;
					else
						return false;
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
		MustEpochKernelLauncher(Context ctx, HighLevelRuntime* rt,const O _op,OpArgList... op_args): op(_op), oargs(op.genArgs(op_args...)),task_out(op)
		{

			// Consolidate all of the fields and regions
			std::map<LPMode,int> req_map;
			std::vector<std::pair<int,LPMode>> arg_req_map;
			std::vector<FieldID> arg_fid_map;
			std::vector<LPMode> lpmode_map;
			std::vector<int> arg_wrap_map(oargs.args.size());

			std::vector<std::set<FieldID>> req_fid_map;

			std::map<LPWrapper,int> lWrap_map;


			std::vector<LPArg> t_args = oargs.args;

			std::vector<LPWrapper> req_lpWrap_map;



			int ireq = 0;
			int iArg = 0;
			int iWrap = 0;
			for(auto it : t_args)
			{
				LPWrapper wrap = it.lp;
				FieldID fid = it.fid;
				PrivilegeMode priv = it.priv;


				arg_fid_map.push_back(fid);
	//			printf("Adding req %i, field %i\n",ireq,fid);

				// Try to find the partition and priviledge combo in our requirement map
				LPMode key(it.lp, it.priv,it.co,it.flags);
	//			auto req_stuff = req_map.find(key);
				if((lWrap_map.count(wrap) == 0))
				{
//					printf("adding wrap %lu as wrap %i\n",wrap.chksm,iWrap);
					lWrap_map[wrap] = iWrap;
					arg_wrap_map[iArg] = iWrap;

					iWrap++;
				}
				else
					arg_wrap_map[iArg] = lWrap_map[wrap];

				// if we find it, append the field id to the end of the list of fids for that req
//				if(std::get<1>(req_map[key]).size() != 0)
				if(req_map.count(key) != 0)
				{
					arg_req_map.push_back(std::pair<int,LPMode>(req_map[key],lpmode_map[req_map[key]]));
//					std::get<1>(req_map[key]).push_back(fid);

					req_fid_map[req_map[key]].insert(fid);


				}
				else // Add the region and field id to the map
				{
					lpmode_map.push_back(key);
					arg_req_map.push_back(std::pair<int,LPMode>(ireq,lpmode_map[ireq]));
//					printf("Adding req %i, field %i, wrap %i in lpmode\n",ireq,fid,arg_wrap_map[iArg]);
					req_map[key] = ireq;
					req_lpWrap_map.push_back(it.lp);
					req_fid_map.push_back({fid});

					ireq++;
				}

				iArg++;

			}

//			printf("req_map has %i elements\n",req_map.size());
			for(auto it : req_map)
			{
//				printf("LPMode %lu = req %i\n",it.first.lp.chksm,it.second);
			}
			// For debugging
			int i2 = 0;
			for(auto it : arg_req_map)
			{
//				printf("Arg %i with fid %i using req %i with %i fids\n",i2,oargs.args[i2].fid,std::get<0>(it),req_fid_map[it.first].size());

				assert(lpmode_map[std::get<0>(it)] == it.second);
				i2++;
			}

			task_out.op_info.narg = arg_req_map.size();
			task_out.op_info.nwrap = lWrap_map.size();

//			printf("%i args versus %i compiler args with %i wrappers\n",task_out.op_info.narg,NARGS,lWrap_map.size());

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

			for(int i=0;i<NARGS;i++)
			{
				task_out.op_info.arg_req_map[i].req = std::get<0>(arg_req_map[i]);
				task_out.op_info.arg_req_map[i].fid = arg_fid_map[i];
				task_out.op_info.arg_req_map[i].wrap = arg_wrap_map[i];
//				printf("Mapping arg %i to wrapper %i\n",i,task_out.op_info.arg_req_map[i].wrap);
//				arg_req_map_out[i].req = std::get<0>(arg_req_map[i]);
//				arg_req_map_out[i].fid = arg_fid_map[i];
//				arg_req_map_out[i].wrap = arg_wrap_map[i];
			}

			for(auto wrap : lWrap_map)
				task_out.op_info.wrappers[wrap.second] = wrap.first;

			size_t arg_size;
			arg_size = sizeof(SingleTaskOp<O>);
//			printf("Argsize is %lu\n",arg_size);
//			arg_size += task_out.op_info.narg*sizeof(ArgMapInfo);
//			arg_size += task_out.op_info.nwrap*sizeof(LRWrapper);

//			arg_ptr = ((char*)malloc(arg_size));

			for(int i=0;i<lpmode_map.size();i++)
			{
				LPMode part_mode = lpmode_map[i];
				LPWrapper wrap = part_mode.lp;
				if(!wrap.single_part)
				{
					Domain cspace = rt->get_index_partition_color_space(ctx,wrap.lp.get_index_partition());
					for(Domain::DomainPointIterator p(cspace);p;p++)
					{
						if(task_map.find(p.p.point_data[0]) == task_map.end())
						{
//							printf("adding  task %lu\n",TASK_ID());
							task_map[p.p.point_data[0]] = TaskLauncher(TASK_ID(),
															TaskArgument(&task_out,sizeof(SingleTaskOp<O>)),
															Predicate::TRUE_PRED,MAPPER_ID);

						}

					}
				}
			}



			for(int i=0;i<lpmode_map.size();i++)
			{

				LPMode part_mode = lpmode_map[i];
				LPWrapper wrap = part_mode.lp;
				PrivilegeMode priv = part_mode.priv;
				CoherenceProperty co = part_mode.co;
				RegionFlags flag = part_mode.flags;
				std::vector<FieldID> fids_out(req_fid_map[req_map[part_mode]].begin(),req_fid_map[req_map[part_mode]].end());

				if(wrap.single_part)
				{
					const char* lr_name;
					rt->retrieve_name(wrap.lr,lr_name);
//					printf("Adding region requirement %i with lr %s with %i fields to all colors\n",i,lr_name,req_fid_map[req_map[part_mode]].size());


						RegionRequirement rr_out(wrap.lr,priv,co,wrap.lr);
						rr_out.add_fields(fids_out);
						rr_out.add_flags(flag);
						for(auto& task : task_map)
							task.second.add_region_requirement(rr_out);

				}
				else
				{
					Domain cspace = rt->get_index_partition_color_space(ctx,wrap.lp.get_index_partition());
					for(Domain::DomainPointIterator p(cspace);p;p++)
					{
						const char* lr_name;
						rt->retrieve_name(wrap.lr,lr_name);
//						printf("Adding region requirement %i with lr %s with %i fields to color %i\n",i,lr_name,req_fid_map[req_map[part_mode]].size(),p.p.point_data[0]);

						LogicalRegion lr_t = rt->get_logical_subregion_by_color(ctx,wrap.lp,p.p.point_data[0]);
						RegionRequirement rr_out(lr_t,priv,co,wrap.lr);
						rr_out.add_fields(fids_out);
						rr_out.add_flags(flag);
						task_map[p.p.point_data[0]].add_region_requirement(rr_out);
					}


				}



			}

			if(oargs.nested_reqs.size() > 0)
			{
				for(auto& reqv : oargs.nested_reqs)
				{

					int ireg = lpmode_map.size();

					for(auto& req : reqv.second)
					{

						const char* lr_name;
						rt->retrieve_name(req.parent,lr_name);
						printf("Region requirement %i for color %i with lr %s\n",ireg,reqv.first,lr_name);
						task_map[reqv.first].add_region_requirement(req);
						ireg++;

					}

				}
			}

			for(auto& task : task_map)
			{
				int ireg = 0;

				for(auto& req : task.second.region_requirements)
				{
					const char* lr_name;
					rt->retrieve_name(req.parent,lr_name);
//					printf("Region requirement %i for color %i with lr %s\n",ireg,task.first,lr_name);
					ireg++;

				}
				add_single_task(DomainPoint(task.first),task.second);
			}


//			printf("Finished setting up an epoch kernel op\n");
		}

	public:
	  bool launch_check_fields(Context ctx, HighLevelRuntime *runtime);
	public:
	  static inline const char* TASK_NAME(){return typeid(MustEpochKernelLauncher<O>).name();};
	  static const TaskID TASK_ID(){return typeid(MustEpochKernelLauncher<O>).hash_code();}
	  static const bool CPU_BASE_LEAF = false;
	  static const bool GPU_BASE_LEAF = true;
	  static const int MAPPER_ID = O::MAPPER_ID;
	  static const int SINGLE = O::SINGLE;
	  static const int INDEX = O::INDEX;
	public:
	  static rT cpu_base_impl(const Task *task,
	                            const std::vector<PhysicalRegion> &regions,
	                            Context ctx, HighLevelRuntime *runtime)

	  {
		  SingleTaskOp<O>* temp = ((SingleTaskOp<O>*)(task->args));
//		  temp->op_info.arg_req_map = (ArgMapInfo*)((char*)(task->args)+sizeof(IndexTaskOp<O>));
//		  temp->op_info.wrappers = (LRWrapper*)(temp->op_info.arg_req_map + temp->op_info.narg);

		  return temp->execute(task,ctx,regions,runtime);

	  }


	  // Register a task that returns a void result
	  template<class U = std::is_void<rT>>
	  static typename std::enable_if<U::value>::type register_cpu(){

		  printf("Setting call for task %lu\n",TASK_ID());


	  HighLevelRuntime::register_legion_task<MustEpochKernelLauncher<O>::cpu_base_impl>(TASK_ID(), Processor::LOC_PROC,
	                                                           SINGLE/*single*/, INDEX/*index*/,
	                                                           0,
	                                                           TaskConfigOptions(CPU_BASE_LEAF),
	                                                           TASK_NAME());
	  }

	  // Register a task that returns a non-void result
	  template<class U = std::is_void<rT>>
	  static typename std::enable_if<!U::value>::type register_cpu(){
		  printf("Setting call for task %lu\n",TASK_ID());

	  HighLevelRuntime::register_legion_task<rT,MustEpochKernelLauncher<O>::cpu_base_impl>(TASK_ID(), Processor::LOC_PROC,
															   SINGLE/*single*/, INDEX/*index*/,
															   0,
															   TaskConfigOptions(CPU_BASE_LEAF),
															   TASK_NAME());
	  }

	  void execute(Context ctx, HighLevelRuntime* rt)
	  {

		  std::vector<Future> fs;
		  for(auto task : task_map)
		  {
			  task.second.point.point_data[0] = task.first;
//			  single_tasks.push_back(task.second);
			  fs.push_back(rt->execute_task(ctx,task.second));
		  }

//		  for(auto f : fs)
//			  f.get_void_result();
//		  rt->execute_must_epoch(ctx,*this).wait_all_results();

	  }


	};

	template<class O,class... OpArgList>
	MustEpochKernelLauncher<O>& genMustEpochKernel(O op,Context ctx, HighLevelRuntime *rt,OpArgList... oargs)
	{
		return *(new MustEpochKernelLauncher<O>(ctx,rt,op,ctx,rt,oargs...));
	}

	template<class O,class... OpArgList>
	MustEpochKernelLauncher<O>& genMustEpochKernel(Context ctx, HighLevelRuntime *rt,O op,OpArgList... oargs)
	{
		return *(new MustEpochKernelLauncher<O>(ctx,rt,op,oargs...));
	}


} /* namespace Dragon */

#endif /* MUSTEPOCHKERNELLAUNCHER_H_ */
