/*
 * LRWrapper.cc
 *
 *  Created on: Jul 7, 2015
 *      Author: payne
 *
 * Copyright (c) 2014-2015 Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the  LANL Contributions to Legion (C15091) project.
 * See the LICENSE.txt file at the top-level directory of this distribution.
 */

#include "LRWrapper.h"
#include "LPWrapper.h"
#include "TypeDeduction.h"


namespace Dragon
{

	LRWrapper& LRWrapper::operator=(const LPWrapper& _in)
	{
		lr = _in.lr;
		ndims = _in.ndims;
		nfields = _in.nfields;
		ntotal = _in.ntotal;
		memcpy(dims,_in.dims,ndims*sizeof(size_t));
		memcpy(fids,_in.fids,nfields*sizeof(uint8_t));
		memcpy(f_types,_in.f_types,nfields*sizeof(size_t));
		chksm = _in.chksm;
		return *this;
	}

	void LRWrapper::GenCheckSum()
	{
		size_t params[3] = {ndims,nfields,ntotal};


		chksm = 0;
		chksm += GenCheckSumArray(params,3);
		chksm += GenCheckSumArray(dims,MAX_LEGION_MATRIX_DIMS);
		chksm += GenCheckSumArray(fids,MAX_LEGION_MATRIX_DIMS);
		chksm += GenCheckSumArray(f_types,MAX_LEGION_MATRIX_DIMS);

//		char* tmp = (char*)this;
//		chksm = GenCheckSumArray(tmp,sizeof(LRWrapper));


	}

} /* namespace Dragon */
