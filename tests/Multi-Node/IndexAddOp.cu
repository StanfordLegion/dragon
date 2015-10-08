/*
 * IndexAddOp.cu
 *
 *  Created on: Jul 22, 2015
 *      Author: payne
 *
 * Copyright (c) 2014-2015 Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the Dragon project. See the LICENSE.txt file at the
 * top-level directory of this distribution.
 */

#include "IndexAddOp.h"

namespace Dragon
{

	void register_index_add_op()
	{
		IndexKernelLauncher<IndexAddOp>::register_cpu();

//		IndexKernelLauncher<IndexAddOp>::register_gpu();

	}

} /* namespace Dragon */



