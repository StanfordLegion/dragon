/*
 * SingleKernelLauncher.cc
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

#include "SingleKernelLauncher.h"

namespace Dragon
{

	void SingleKernelArgs::add_arg(LRWrapper wrapper,FieldID fid,
	             PrivilegeMode priv,
	             CoherenceProperty co,
	             RegionFlags flag)
	{
		args.push_back(LRArg(wrapper,fid,priv,co,flag));
	}

	void SingleKernelArgs::add_arg(LRWrapper wrapper,FieldID fid,
	             PrivilegeMode priv,
	             RegionFlags flag)
	{
		args.push_back(LRArg(wrapper,fid,priv,EXCLUSIVE,flag));
	}

	void SingleKernelArgs::add_nested(std::vector<RegionRequirement> v_nested)
	{
		nested_reqs.insert(nested_reqs.end(),v_nested.begin(),v_nested.end());
	}

	void SingleKernelArgs::add_nested(RegionRequirement v_nested)
	{
		nested_reqs.push_back(v_nested);
	}

} /* namespace Dragon */
