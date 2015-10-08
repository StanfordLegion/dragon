/*
 * IndexKernelLauncher.cc
 *
 *  Created on: Jul 13, 2015
 *      Author: payne
 *
 * Copyright (c) 2014-2015 Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the  LANL Contributions to Legion (C15091) project.
 * See the LICENSE.txt file at the top-level directory of this distribution.
 */

#include "IndexKernelLauncher.h"

namespace Dragon
{

	void IndexKernelArgs::add_arg(LPWrapper wrapper,FieldID fid,
 	             	             PrivilegeMode priv,
 	             	             CoherenceProperty co,
 	             	             RegionFlags flag)
	{
		args.push_back(LPArg(wrapper,fid,priv,co,flag));
	}

	void IndexKernelArgs::add_arg(LPWrapper wrapper,FieldID fid,
				 PrivilegeMode priv,
				 RegionFlags flag)
	{
		args.push_back(LPArg(wrapper,fid,priv,EXCLUSIVE,flag));
	}

	void IndexKernelArgs::add_arg(LRWrapper wrapper,FieldID fid,
 	             	             PrivilegeMode priv,
 	             	             CoherenceProperty co,
 	             	             RegionFlags flag)
	{
		LPWrapper lp;
		lp = wrapper;
		args.push_back(LPArg(lp,fid,priv,co,flag));
	}

	void IndexKernelArgs::add_arg(LRWrapper wrapper,FieldID fid,
				 PrivilegeMode priv,
				 RegionFlags flag)
	{
		LPWrapper lp;
		lp = wrapper;
		args.push_back(LPArg(lp,fid,priv,EXCLUSIVE,flag));
	}

	void IndexKernelArgs::add_nested(std::vector<RegionRequirement> v_nested)
	{
		nested_reqs.insert(nested_reqs.end(),v_nested.begin(),v_nested.end());
	}

	void IndexKernelArgs::add_nested(RegionRequirement v_nested)
	{
		nested_reqs.push_back(v_nested);
	}

	void IndexKernelArgs::set_result(LPWrapper wrapper,FieldID fid,
 	             	             CoherenceProperty co)
	{
		res = (LPArg(wrapper,fid,WRITE_ONLY,co,NO_FLAG));
	}

	void IndexKernelArgs::add_loop_domain(Domain _dom)
	{
		loop_domain = _dom;
	}

	void IndexKernelArgs::add_loop_domain(unsigned iStart, unsigned iEnd)
	{
		loop_domain.rect_data[0] = iStart;
		loop_domain.rect_data[1] = iEnd;
	}


} /* namespace Dragon */
