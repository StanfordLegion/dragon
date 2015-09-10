/*
 * MustEpochKernelLauncher.cc
 *
 *  Created on: Jul 23, 2015
 *      Author: payne
 */

#include "MustEpochKernelLauncher.h"

namespace Dragon
{
	void EpochKernelArgs::add_arg(LPWrapper wrapper,FieldID fid,
 	             	             PrivilegeMode priv,
 	             	             CoherenceProperty co,
 	             	             RegionFlags flag)
	{
		args.push_back(LPArg(wrapper,fid,priv,co,flag));
	}

	void EpochKernelArgs::add_arg(LPWrapper wrapper,FieldID fid,
				 PrivilegeMode priv,
				 RegionFlags flag)
	{
		args.push_back(LPArg(wrapper,fid,priv,EXCLUSIVE,flag));
	}

//	void EpochKernelArgs::add_arg(LRWrapper wrapper,FieldID fid,
// 	             	             PrivilegeMode priv,
// 	             	             CoherenceProperty co,
// 	             	             RegionFlags flag)
//	{
//		LPWrapper lp;
//		lp = wrapper;
//		args.push_back(LPArg(lp,fid,priv,co,flag));
//	}
//
//	void EpochKernelArgs::add_arg(LRWrapper wrapper,FieldID fid,
//				 PrivilegeMode priv,
//				 RegionFlags flag)
//	{
//		LPWrapper lp;
//		lp = wrapper;
//		args.push_back(LPArg(lp,fid,priv,EXCLUSIVE,flag));
//	}

	void EpochKernelArgs::add_nested(Color cl,std::vector<RegionRequirement> v_nested)
	{
		nested_reqs[cl].insert(nested_reqs[cl].end(),v_nested.begin(),v_nested.end());
	}

	void EpochKernelArgs::add_nested(Color cl,RegionRequirement v_nested)
	{
		nested_reqs[cl].push_back(v_nested);
	}


} /* namespace Dragon */
