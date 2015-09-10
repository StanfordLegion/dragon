/*
 * IndexAddOp.cu
 *
 *  Created on: Jul 22, 2015
 *      Author: payne
 */

#include "IndexAddOp.h"

namespace Dragon
{

	void register_index_add_op()
	{
//		IndexKernelLauncher<IndexAddOp>::register_cpu();

		IndexKernelLauncher<IndexAddOp>::register_gpu();

	}

} /* namespace Dragon */
