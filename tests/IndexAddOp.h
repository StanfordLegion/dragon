/*
 * IndexAddOp.h
 *
 *  Created on: Jul 22, 2015
 *      Author: payne
 *
 * Copyright (c) 2014-2015 Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the  LANL Contributions to Legion (C15091) project.
 * See the LICENSE.txt file at the top-level directory of this distribution.
 */

#ifndef INDEXADDOP_H_
#define INDEXADDOP_H_
#include "../src/SingleKernelLauncher.h"
#include "../src/LegionMatrix.h"
#include "../src/LegionHelper.h"
#include "../src/LPWrapper.h"
#include "../src/IndexKernelLauncher.h"
namespace Dragon
{
	void register_index_add_op();

	class IndexAddOp
	{
	public:
	public: // Required members
		static const int SINGLE = false;
		static const int INDEX = true;
		static const int MAPPER_ID = 0;
	public:
		IndexAddOp(int _nx, int _ny, int _nz) : nx(_nx),ny(_ny),nz(_nz) {}
		int nx,ny,nz;
		IndexKernelArgs genArgs(Context ctx, HighLevelRuntime* runtime,
					   LRWrapper _a,
					   LRWrapper _b,
					   LRWrapper _c)
		{
			IndexKernelArgs args;
			LPWrapper a;
			LPWrapper b;
			LPWrapper c;


			a.slicedPart(ctx,runtime,_a,":",":",":");
			b.slicedPart(ctx,runtime,_b,":",":",":");
			c.slicedPart(ctx,runtime,_c,":",":",":");


			args.add_arg(a,0,READ_ONLY,EXCLUSIVE);
			args.add_arg(b,0,READ_ONLY,EXCLUSIVE);

			args.set_result(c,0,EXCLUSIVE);

			return args;
		}


		__host__ __device__
		static int evaluate_s(int idx,LegionMatrix<int> a,
		                      LegionMatrix<int> b){};

		__host__ __device__
		int evaluate(int idx,LegionMatrix<int> a,
	                 LegionMatrix<int> b)
		{
			int i = idx%nx;
			int j = (idx/nx)%ny;
			int k = (idx/(nx*ny));

			return a(i,j,k).cast() + b(i,j,k).cast();
		}

	};

} /* namespace Dragon */

#endif /* INDEXADDOP_H_ */
