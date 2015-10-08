/*
 * Plot2DNested.h
 *
 *  Created on: Jul 30, 2015
 *      Author: payne
 *
 * Copyright (c) 2014-2015 Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the  LANL Contributions to Legion (C15091) project.
 * See the LICENSE.txt file at the top-level directory of this distribution.
 */
#ifndef PLOT2DNESTED_H_
#define PLOT2DNESTED_H_
#include "gnuplot_i.h"
#include <stdio.h>
#include <legion.h>
#include <tuple>
#include <stdlib.h>
#include <LegionMatrix.h>
#include <LRWrapper.h>
#include <unistd.h>
#include "SingleKernelLauncher.h"


namespace Dragon
{
	using namespace LegionRuntime::HighLevel;
	using namespace LegionRuntime::Accessor;
	using namespace LegionRuntime::Arrays;

	template<typename T>
	class Plot2DNested
	{
	public: // Required members
		static const int SINGLE = true;
		static const int INDEX = false;
		static const int MAPPER_ID = 0;
	public:

		LRWrapper child_partitions;
		FieldID fid_child;
		unsigned x_first,x_last;
		unsigned y_first,y_last;
		int dim_x = -1;
		int dim_y = -1;
		unsigned nx,ny;

		unsigned x_first_c,x_last_c;
		unsigned y_first_c,y_last_c;
		int dim_x_c = -1;
		int dim_y_c = -1;
		unsigned nx_c,ny_c;

		unsigned x_first_p,x_last_p;
		unsigned y_first_p,y_last_p;
		int dim_x_p = -1;
		int dim_y_p = -1;
		unsigned nx_p,ny_p;

		LRWrapper exit_signal;
		PhaseBarrier plot_barrier;


		SingleKernelArgs args_out;

		template<class... Args>
		Plot2DNested(Context ctx,HighLevelRuntime* rt,FieldID _fid_child,FieldID fid_parent,LPWrapper parent,Args... _args)
		{

			fid_child = _fid_child;

			for(int i=0;i<parent.ndims;i++)
			{
				if(parent.slices[i].first != parent.slices[i].last)
				{
					if(dim_x_p == -1)
					{
						dim_x_p = i;
						x_first_p = parent.slices[i].first;
						x_last_p = parent.slices[i].last;
					}
					else
					{
						dim_y_p = i;
						y_first_p = parent.slices[i].first;
						y_last_p = parent.slices[i].last;
					}
				}
			}

			nx_p = x_last_p - x_first_p +1;
			ny_p = y_last_p - y_first_p +1;

			child_partitions.create(ctx,rt,{nx_p,ny_p},0,LPWrapper());
			exit_signal.create(ctx,rt,{1},0,bool());


			{
				RegionRequirement rr_parent_in(parent.lp,0,READ_ONLY,EXCLUSIVE,parent.lr);
				rr_parent_in.add_field(fid_parent);

				RegionRequirement rr_parent_out(child_partitions.lr,WRITE_ONLY,EXCLUSIVE,child_partitions.lr);
				rr_parent_out.add_field(0);

				PhysicalRegion pr_in = rt->map_region(ctx,rr_parent_in);
				PhysicalRegion pr_out = rt->map_region(ctx,rr_parent_out);

				LRWrapper parent_tmp = parent;
				LegionMatrix<LRWrapper> children_in(parent_tmp,pr_in,fid_parent);
				LegionMatrix<LPWrapper> children_out(child_partitions,pr_out,fid_parent);



				Domain cspace = rt->get_index_subspace(ctx,parent.lp.get_index_partition(),0);
				for(Domain::DomainPointIterator p(cspace);p;p++)
				{
					unsigned idx = p.p.point_data[0];
					unsigned indices[MAX_LEGION_MATRIX_DIMS];
					parent.GetMDIndex(idx,indices);
					unsigned ix = indices[dim_x_p] - x_first_p;
					unsigned iy = indices[dim_y_p] - y_first_p;
					LPWrapper child;
					child.slicedPart(ctx,rt,children_in(idx).cast(),_args...);

					RegionRequirement rr_nested(rt->get_logical_subregion_by_color(ctx,child.lp,0),READ_ONLY,SIMULTANEOUS,child.lr);
					child.lr = rt->get_logical_subregion_by_color(ctx,child.lp,0);
					children_out(ix,iy) = child;

					rr_nested.add_field(fid_child);
					rr_nested.add_flags(NO_ACCESS_FLAG);
					args_out.add_nested(rr_nested);

				}

				args_out.add_arg(child_partitions,0,READ_ONLY,SIMULTANEOUS);



				rt->unmap_region(ctx,pr_in);
				rt->unmap_region(ctx,pr_out);

			}
		}
		SingleKernelArgs genArgs(Context ctx, HighLevelRuntime* runtime)
		{
			return args_out;
		}


		__host__
		static void evaluate_s(const Task* task,Context ctx,HighLevelRuntime* rt,
		         			  LegionMatrix<LPWrapper> lw_parent){};

		__host__
		void evaluate(const Task* task,Context ctx,HighLevelRuntime* rt,
		              LegionMatrix<LPWrapper> lw_parent)
		{



		}

		static void register_task();
	};

} /* namespace Dragon */

#endif /* PLOT2DNESTED_H_ */
