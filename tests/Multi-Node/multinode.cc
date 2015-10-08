/*
 * multinode.cc
 *
 *  Created on: Aug 13, 2015
 *      Author: payne
 *
 * Copyright (c) 2014-2015 Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the Dragon project. See the LICENSE.txt file at the
 * top-level directory of this distribution.
 */




#include <SingleKernelLauncher.h>
#include <LegionMatrix.h>
#include <LegionHelper.h>
#include <LPWrapper.h>
#include <IndexKernelLauncher.h>
#include <BetterMapper.h>
#include "IndexAddOp.h"

using namespace Dragon;

enum Task_IDs
{
	TOP_LEVEL_TASK_ID
};




class Print3D
{
public:
public: // Required members
	static const int SINGLE = false;
	static const int INDEX = true;
	static const int MAPPER_ID = 0;
public:

	Print3D(int _nx, int _ny, int _nz) : nx(_nx),ny(_ny),nz(_nz) {}
	int nx,ny,nz;
	IndexKernelArgs genArgs(Context ctx, HighLevelRuntime* runtime,
				   LRWrapper _dummy)
	{
		IndexKernelArgs args;
		LPWrapper dummy_part;

		dummy_part.slicedPart(ctx,runtime,_dummy,":",":","%1");

		args.add_arg(dummy_part,0,READ_ONLY,EXCLUSIVE);

		return args;
	}


	__host__
	static void evaluate_s(int i,LegionMatrix<int> dummy){};

	__host__
	void evaluate(int idx,LegionMatrix<int> dummy)
	{
		int i = idx%nx;
		int j = (idx/nx)%ny;
		int k = (idx/(nx*ny));
		printf("array(%i, %i, %i) = %i \n",i,j,k,dummy(i,j,k).cast());
	}

};


void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{


	LRWrapper a;
	a.create(ctx,runtime,"a",{10,10,10},0,int());

	LRWrapper b;
	b.create(ctx,runtime,"b",{10,10,10},0,int());

	LRWrapper c;
	c.create(ctx,runtime,"c",{10,10,10},0,int());

	LegionHelper helper(ctx,runtime);
	int dummy_tmp[10];
	for(int i=0;i<10;i++)
		dummy_tmp[i] = i;

	double dummy2_tmp[10];
	for(int i=0;i<10;i++)
		dummy2_tmp[i] = i;

	int a_tmp[1000];
	for(int i=0;i<1000;i++)
		a_tmp[i] = 1;

	int b_tmp[1000];
	for(int i=0;i<1000;i++)
		b_tmp[i] = i;

	helper.set(a.lr,0,a_tmp,1000);
	helper.set(b.lr,0,b_tmp,1000);

	IndexAddOp add(10,10,10);

	Print3D check(10,10,10);

	auto add_kernel = genIndexKernel(add,ctx,runtime,a,b,c);
	runtime->execute_index_space(ctx,add_kernel);

	auto check_add = genIndexKernel(check,ctx,runtime,c);

//	runtime->execute_index_space(ctx,check_add).wait_all_results();




}



static void update_mappers(Machine machine, HighLevelRuntime *rt,
						   const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
		it != local_procs.end(); it++)
  {
//	  rt->add_mapper(1,new BetterMapper(machine, rt, *it), *it);
    rt->replace_default_mapper(new BetterMapper(machine, rt, *it), *it);
  }
}


int main(int argc, char **argv)
{

  char hostname[512];
  gethostname(hostname,512);
//
//	  dup2(fileno(err_file),STDERR_FILENO);
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/);


//	  fclose(err_file);



//	  std::stringstream buffer;
  setbuf(stdout,NULL);
  setvbuf(stderr,NULL,_IOLBF,1024);

//  system("export GASNET_BACKTRACE=1");



  IndexKernelLauncher<Print3D>::register_cpu();



  register_index_add_op();

  TaskHelper::register_hybrid_variants<LegionHelper::Setter<int> >();
  TaskHelper::register_hybrid_variants<LegionHelper::Setter<double> >();



//  HighLevelRuntime::set_registration_callback(update_mappers);




  return HighLevelRuntime::start(argc, argv);
}
