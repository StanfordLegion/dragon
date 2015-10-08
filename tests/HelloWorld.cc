/*
 * HelloWorld.cc
 *
 *  Created on: Jul 9, 2015
 *      Author: payne
 *
 * Copyright (c) 2014-2015 Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the  LANL Contributions to Legion (C15091) project.
 * See the LICENSE.txt file at the top-level directory of this distribution.
 */

#include "../src/SingleKernelLauncher.h"
#include "../src/LegionMatrix.h"
#include "../src/LegionHelper.h"
#include "../src/LPWrapper.h"
#include "../src/IndexKernelLauncher.h"
#include "IndexAddOp.h"
#include "../src/BetterMapper.h"

using namespace Dragon;

enum Task_IDs
{
	TOP_LEVEL_TASK_ID
};


class HelloWorld1
{
public:
public: // Required members
	static const int SINGLE = true;
	static const int INDEX = false;
	static const int MAPPER_ID = 0;
public:


	SingleKernelArgs genArgs(Context ctx, HighLevelRuntime* runtime,
				   LRWrapper _dummy)
	{
		SingleKernelArgs args;

		args.add_arg(_dummy,0,READ_WRITE,EXCLUSIVE,NO_ACCESS_FLAG);

		return args;
	}


	__host__
	static int evaluate_s(LegionMatrix<int> dummy){};

	__host__
	int evaluate(LegionMatrix<int> dummy)
	{
		printf("Hello World!\n");
		return dummy(5).cast();
	}

};

class HelloWorld2
{
public:
public: // Required members
	static const int SINGLE = true;
	static const int INDEX = false;
	static const int MAPPER_ID = 0;
public:


	SingleKernelArgs genArgs(Context ctx, HighLevelRuntime* runtime,
				   LRWrapper _dummy)
	{
		SingleKernelArgs args;

		args.add_arg(_dummy,0,READ_WRITE,EXCLUSIVE,NO_ACCESS_FLAG);

		return args;
	}


	__host__
	static void evaluate_s(LegionMatrix<int> dummy){};

	__host__
	void evaluate(LegionMatrix<int> dummy)
	{
		printf("Hello World! void\n");
	}

};

class HelloWorld3
{
public:
public: // Required members
	static const int SINGLE = true;
	static const int INDEX = false;
	static const int MAPPER_ID = 0;
public:


	SingleKernelArgs genArgs(Context ctx, HighLevelRuntime* runtime,
				   LRWrapper _dummy)
	{
		SingleKernelArgs args;

		args.add_arg(_dummy,0,READ_WRITE,EXCLUSIVE,NO_ACCESS_FLAG);

		return args;
	}


	__host__
	static void evaluate_s(const Task* task, Context ctx, HighLevelRuntime* rt,LegionMatrix<int> dummy){};

	__host__
	void evaluate(const Task* task, Context ctx, HighLevelRuntime* rt,LegionMatrix<int> dummy)
	{
		printf("Hello World! 3\n");
	}

};

class HelloWorld4
{
public:
public: // Required members
	static const int SINGLE = true;
	static const int INDEX = false;
	static const int MAPPER_ID = 0;
public:


	SingleKernelArgs genArgs(Context ctx, HighLevelRuntime* runtime,
				   LRWrapper _dummy,
				   LRWrapper _dummy2)
	{
		SingleKernelArgs args;

		args.add_arg(_dummy,0,READ_WRITE,EXCLUSIVE);
		args.add_arg(_dummy2,0,READ_WRITE,EXCLUSIVE);


		return args;
	}


	__host__
	static void evaluate_s(LegionMatrix<int> dummy,LegionMatrix<double> dummy2){};

	__host__
	void evaluate(LegionMatrix<int> dummy,LegionMatrix<double> dummy2)
	{
		printf("Hello World 4! %i %e\n",dummy(5).cast(),dummy2(3).cast());
	}

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



Coloring slice_single_dim(unsigned nx, unsigned ny, unsigned nz,unsigned iSlice_dim, unsigned stride)
{
	typedef std::pair<ptr_t,ptr_t> ptr2;
	Coloring cl;

	auto hf = [&](unsigned i,unsigned j, unsigned k){return (unsigned)(i+nx*(j+nz*k));};

	switch(iSlice_dim)
	{
		case 0: // nx
			for(unsigned k=0;k<nz;k++)
				for(unsigned j=0;j<ny;j++)
				{
					unsigned icl = 0;
					for(unsigned i=0;i<nx;i+=stride)
					{
						unsigned i_l = std::min(nx-1,i+stride-1);
						if(stride==1)
							cl[icl].points.insert(hf(i,j,k));
						else
							cl[icl].ranges.insert(ptr2(hf(i,j,k),hf(i_l,j,k)));

						icl++;
					}
				}

			break;
		case 1: // ny
			for(unsigned k=0;k<nz;k++)
			{
				unsigned icl = 0;
				for(unsigned j=0;j<ny;j+=stride)
				{
					unsigned j_l = std::min(ny-1,j+stride-1);

					cl[icl].ranges.insert(ptr2(hf(0,j,0),hf(nx-1,j_l,nz-1)));
					icl++;
				}
			}
			break;

		case 2: // nz
			unsigned icl = 0;
			for(unsigned k=0;k<nz;k+=stride)
			{
				unsigned k_l = std::min(nz-1,k+stride-1);

				cl[icl].ranges.insert(std::pair<ptr_t,ptr_t>(hf(0,0,k),hf(nx-1,ny-1,k_l)));
				icl++;
			}
			break;
	}



}

ColoredPoints<ptr_t> CalcColoring(unsigned i0, unsigned i1, unsigned n,ColoredPoints<ptr_t> recurse)
{
	ColoredPoints<ptr_t> result;
	typedef std::pair<ptr_t,ptr_t> ptr2;

	if(recurse.points.size() > 0)
	{
		for(auto point : recurse.points)
			if(i0 == i1)
				result.points.insert(i0+n*(unsigned)point);
			else
				result.ranges.insert(ptr2(i0+n*(unsigned)point,i1+n*(unsigned)point));
	}
	else if(recurse.ranges.size() > 0)
	{
		for(auto range : recurse.ranges)
			if(i0 == i1)
				for(unsigned i=range.first;i<=(unsigned)range.second;i++)
					result.points.insert(i0+n*i);
			else if(i0 == 0 && (i1 == n-1))
				result.ranges.insert(ptr2(n*(unsigned)range.first,n-1+n*(unsigned)range.second));
			else
				for(unsigned i=range.first;i<=(unsigned)range.second;i++)
					result.ranges.insert(ptr2(i0+n*i,i1+n*i));

	}
	else
	{
		if(i0 == i1)
			result.points.insert(i0);
		else
			result.ranges.insert(ptr2(i0,i1));
	}

	return result;
}

void recurse_slice_dims(int iDim,std::vector<unsigned> dims,std::map<unsigned,unsigned> stride_map,
                        ColoredPoints<ptr_t> recurse,
                        Coloring& cl,unsigned& iCL)
{

	auto hf = [&](std::vector<unsigned> pts)
			{
				unsigned res = 0;;
				for(unsigned l=pts.size()-1;l>=0;l--)
					res = pts[l] + dims[l]*res;
				return res;
			};

	unsigned ndims = dims.size();

	if(iDim < 0)
	{
		cl[iCL] = recurse;
	}
	else
	{
		if(stride_map.find(iDim) != stride_map.end())
		{ // Slice along this dim
			unsigned stride = stride_map[iDim];

			for(unsigned i=0;i<dims[iDim];i+=stride)
			{
				unsigned i_l = std::min(dims[iDim]-1,i+stride-1);

				ColoredPoints<ptr_t> pts_out = CalcColoring(i,i_l,dims[iDim],recurse);

				recurse_slice_dims(iDim-1,dims,stride_map,pts_out,cl,iCL);

				iCL++;
			}
		}
		else
		{
			ColoredPoints<ptr_t> pts_out = CalcColoring(0,dims[iDim]-1,dims[iDim],recurse);
			recurse_slice_dims(iDim-1,dims,stride_map,pts_out,cl,iCL);
		}
	}
}

Coloring slice_multi_dims(std::vector<unsigned> dims, std::vector<std::pair<unsigned,unsigned>> slice_strides)
{
	Coloring res;
	ColoredPoints<ptr_t> recurse;
	unsigned iCL = 0;

	std::map<unsigned,unsigned> stride_map;


	for(auto stride : slice_strides)
	{
		stride_map[stride.first] = stride.second;
	}

	recurse_slice_dims(dims.size()-1,dims,stride_map,recurse,res,iCL);

	return res;
}

void print3dimCl(Coloring _cl,unsigned nx, unsigned ny, unsigned nz)
{
	printf("lpw has %i colors\n",_cl.size());
	for(auto cl : _cl)
	{
		if(cl.second.points.size() > 0)
		{
			printf("Color %u contains points:",cl.first);
			for(auto pt : cl.second.points)
			{
				unsigned idx = (unsigned)pt;

				unsigned ix = idx%nx;
				unsigned iy = (idx/nx)%ny;
				unsigned iz = (idx/(nx*ny));

				printf(" (%u):",idx);
				printf("(%u %u %u)",ix,iy,iz);

			}
			printf("\n");
		}

		if(cl.second.ranges.size() > 0)
		{
			printf("Color %u contains ranges:",cl.first);
			for(auto rg : cl.second.ranges)
			{
				unsigned first = (unsigned)(rg.first);
				unsigned last = (unsigned)(rg.second);


				unsigned ix0 = first%nx;
				unsigned iy0 = (first/nx)%ny;
				unsigned iz0 = (first/(nx*ny));

				unsigned ix1 = last%nx;
				unsigned iy1 = (last/nx)%ny;
				unsigned iz1 = (last/(nx*ny));

				printf(" (%u-%u):",first,last);

				printf("(%u-%u %u-%u %u-%u)",ix0,ix1,iy0,iy1,iz0,iz1);

			}
			printf("\n");
		}
	}
}

void print3dimCl(LPWrapper lpw,unsigned nx, unsigned ny, unsigned nz)
{
//	print3dimCl(*lpw._cl,nx,ny,nz);
}





void test_lpwrapper_coloring(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, HighLevelRuntime *runtime,
                             int nx, int ny, int nz)
{

	printf("Testing LPWrapper Partitioning\n");
	LRWrapper dummy;
	dummy.create(ctx,runtime,{nx,ny,nz},0,int());

	LPWrapper single_lp;
	LPWrapper last_dim_1;
	LPWrapper middle_dim_1;
	LPWrapper middle_dim_1_int_last;
	LPWrapper two_dims_1;
	LPWrapper single_first_split_last;
	LPWrapper range_first_split_last;

	Coloring cl;


	LRWrapper a_lr;
	a_lr.create(ctx,runtime,{10,10,10},0,int());
	LPWrapper a_lp;
	a_lp.slicedPart(ctx,runtime,a_lr,":","%1",":");

	single_lp.slicedPart(ctx,runtime,dummy,":",":",":");
	printf("single_lp check\n");
	print3dimCl(single_lp,nx,ny,nz);
	cl = slice_multi_dims({nx,ny,nz},{});
	print3dimCl(cl,nx,ny,nz);

	last_dim_1.slicedPart(ctx,runtime,dummy,":",":","%1");
	printf("last_dim_1 check\n");
	print3dimCl(last_dim_1,nx,ny,nz);
	cl = slice_multi_dims({nx,ny,nz},{{2,1}});
	print3dimCl(cl,nx,ny,nz);

	middle_dim_1.slicedPart(ctx,runtime,dummy,":","%1",":");
	printf("middle_dim_1 check\n");
	print3dimCl(middle_dim_1,nx,ny,nz);
	cl = slice_multi_dims({nx,ny,nz},{{1,1}});
	print3dimCl(cl,nx,ny,nz);

	middle_dim_1_int_last.slicedPart(ctx,runtime,dummy,":","%1",nx/2);
	printf("middle_dim_1_int_last check\n");
	print3dimCl(middle_dim_1_int_last,nx,ny,nz);

	two_dims_1.slicedPart(ctx,runtime,dummy,":","%2","%1");
	printf("two_dims_1 check\n");
	print3dimCl(two_dims_1,nx,ny,nz);

	single_first_split_last.slicedPart(ctx,runtime,dummy,0,"%1",0);
	printf("single_first_split_last check\n");
	print3dimCl(single_first_split_last,nx,ny,nz);

	range_first_split_last.slicedPart(ctx,runtime,dummy,"0:2","%1","0:3%2");
	printf("range_first_split_last check\n");
	print3dimCl(range_first_split_last,nx,ny,nz);

	Coloring cl_f;
	Coloring cl_b;
	Coloring cl_top;
	Coloring cl_bottom;
	Coloring cl_l;
	Coloring cl_r;

	LRWrapper dummy2;
	dummy2.create(ctx,runtime,{nx,ny},0,int());

	cl_f[0] = dummy.GetColoring(":",":",0);
	cl_b[0] = dummy.GetColoring(":",":",nz-1);

	cl_top[0] = dummy.GetColoring(":",0,":");
	cl_bottom[0] = dummy.GetColoring(":",ny-1,":");

	cl_l[0] = dummy.GetColoring(0,":",":");
	cl_r[0] = dummy.GetColoring(nx-1,":",":");

	printf("y_top check\n");
	print3dimCl(cl_top,nx,ny,nz);

	Coloring cl_z = LPWrapper::mergeColorings(cl_f,cl_b);
	Coloring cl_y = LPWrapper::mergeColorings(cl_top,cl_bottom);
	printf("y_halo check\n");
	print3dimCl(cl_y,nx,ny,nz);
	Coloring cl_x = LPWrapper::mergeColorings(cl_l,cl_r);

	printf("x_halo check\n");
	print3dimCl(cl_x,nx,ny,nz);

	printf("z_halo check\n");
	print3dimCl(cl_z,nx,ny,nz);

	Coloring cl_halo = LPWrapper::mergeColorings(LPWrapper::mergeColorings(cl_x,cl_y),cl_z);
	printf("halo check\n");
	print3dimCl(cl_halo,nx,ny,nz);











}


void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
	LRWrapper dummy;
	dummy.create(ctx,runtime,{10},0,int());

	LRWrapper dummy2;
	dummy2.create(ctx,runtime,{10},0,double());

	LRWrapper a;
	a.create(ctx,runtime,{10,10,10},0,int());

	LRWrapper b;
	b.create(ctx,runtime,{10,10,10},0,int());

	LRWrapper c;
	c.create(ctx,runtime,{10,10,10},0,int());

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

	helper.set(dummy.lr,0,dummy_tmp,10);
	helper.set(dummy2.lr,0,dummy2_tmp,10);


	helper.set(a.lr,0,a_tmp,1000);
	helper.set(b.lr,0,b_tmp,1000);

	HelloWorld1 hw1;
	HelloWorld2 hw2;
	HelloWorld3 hw3;
	HelloWorld4 hw4;

	IndexAddOp add_op(10,10,10);
	Print3D check(10,10,10);





	auto hw1_kern = genSingleKernel(hw1,ctx,runtime,dummy);
	auto hw2_kern = genSingleKernel(hw2,ctx,runtime,dummy);
	auto hw3_kern = genSingleKernel(hw3,ctx,runtime,dummy);
	auto hw4_kern = genSingleKernel(hw4,ctx,runtime,dummy,dummy2);

	auto add_kern = genIndexKernel(add_op,ctx,runtime,a,b,c);

	auto check_add = genIndexKernel(check,ctx,runtime,c);



	printf("hw1 taskid = %u\n",hw1_kern.task_id);
	runtime->execute_task(ctx,hw2_kern);
	runtime->execute_task(ctx,hw3_kern);
	runtime->execute_task(ctx,hw4_kern);

	runtime->execute_index_space(ctx,add_kern);
	runtime->execute_index_space(ctx,check_add);



	Future f = runtime->execute_task(ctx,hw1_kern);

	printf("hw1 = %i\n",f.get_result<int>());


//	test_lpwrapper_coloring(task,regions,ctx,runtime,4,4,4);

//	test_lpwrapper_coloring(task,regions,ctx,runtime,10,10,10);

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

 SingleKernelLancher<HelloWorld1>::register_cpu();
 SingleKernelLancher<HelloWorld2>::register_cpu();
 SingleKernelLancher<HelloWorld3>::register_cpu();
 SingleKernelLancher<HelloWorld4>::register_cpu();




  register_index_add_op();
  IndexKernelLauncher<Print3D>::register_cpu();





  TaskHelper::register_hybrid_variants<LegionHelper::Setter<int> >();
  TaskHelper::register_hybrid_variants<LegionHelper::Setter<double> >();



  HighLevelRuntime::set_registration_callback(update_mappers);




  return HighLevelRuntime::start(argc, argv);
}
