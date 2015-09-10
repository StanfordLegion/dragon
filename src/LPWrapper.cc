/*
 * LPWrapper.cc
 *
 *  Created on: Jul 14, 2015
 *      Author: payne
 */
#include "LRWrapper.h"
#include "LPWrapper.h"
#include "TypeDeduction.h"

namespace Dragon
{

	LPWrapper& LPWrapper::operator=(const LRWrapper& _in)
	{
		lr = _in.lr;
		ndims = _in.ndims;
		nfields = _in.nfields;
		ntotal = _in.ntotal;
		memcpy(dims,_in.dims,ndims*sizeof(size_t));
		memcpy(fids,_in.fids,nfields*sizeof(uint8_t));
		memcpy(f_types,_in.f_types,nfields*sizeof(size_t));

		single_part = true;
		GenCheckSum();
		return *this;
	}

//	LPWrapper& LPWrapper::operator=(const LPWrapper& _in)
//	{
//		lr = _in.lr;
//		lp = _in.lp;
//		lDom = _in.lDom;
//		n_slice_dims = _in.n_slice_dims;
//
//		ndims = _in.ndims;
//		nfields = _in.nfields;
//		memcpy(dims,_in.dims,ndims*sizeof(size_t));
//		memcpy(fids,_in.fids,nfields*sizeof(uint8_t));
//		memcpy(f_types,_in.f_types,nfields*sizeof(size_t));
//
//		memcpy(slices,_in.slices,ndims*sizeof(SliceParams));
//		memcpy(slice_dims,_in.slice_dims,ndims*sizeof(int8_t));
//
//
//		single_part = _in.single_part;
//		chksm = _in.chksm;
//		return *this;
//	}

	LPWrapper::operator LRWrapper()
	{
		LRWrapper _out;
		_out = *this;
		return _out;
	}

	void LPWrapper::GenCheckSum()
	{
		size_t params[5] = {ndims,n_slice_dims,nfields,ntotal,single_part};


		chksm = 0;
		chksm += GenCheckSumArray(params,5);
		chksm += GenCheckSumArray(slices,ndims);
		chksm += GenCheckSumArray(slice_dims,n_slice_dims);
		chksm += GenCheckSumArray(dims,ndims);
		chksm += GenCheckSumArray(fids,nfields);
		chksm += GenCheckSumArray(f_types,nfields);
		char* tmp = (char*)(&lp);
		chksm += GenCheckSumArray(tmp,sizeof(LogicalPartition));
//		char* tmp = (char*)this;
//		chksm = GenCheckSumArray(tmp,sizeof(LPWrapper));


	}

	void LPWrapper::singlePart(Context ctx, HighLevelRuntime* rt,LRWrapper lw,unsigned iCl)
	{
		*this = lw;

		Coloring cl;
		n_slice_dims = 0;

		cl[iCl].ranges.insert(std::pair<ptr_t,ptr_t>(0,ntotal-1));
//		IndexPartition ip;
		ip = rt->create_index_partition(ctx,lr.get_index_space(),cl,true);
		lp = rt->get_logical_partition(ctx,lr,ip);

		lDom = rt->get_index_partition_color_space(ctx,ip);

		const char* lr_name;
		rt->retrieve_name(lr,lr_name);
		rt->attach_name(lp,lr_name);
		rt->attach_name(ip,lr_name);

		single_part = true;
		GenCheckSum();



	}

	void LPWrapper::copiedPart(Context ctx, HighLevelRuntime* rt,LRWrapper lw,unsigned n)
	{
		*this = lw;

		n_slice_dims = 0;
		Coloring cl;
		for(unsigned i=0;i<n;i++)
			cl[i].ranges.insert(std::pair<ptr_t,ptr_t>(0,ntotal-1));
//		IndexPartition ip;
		ip = rt->create_index_partition(ctx,lr.get_index_space(),cl,false);
		lp = rt->get_logical_partition(ctx,lr,ip);

		lDom = rt->get_index_partition_color_space(ctx,ip);

		single_part = false;
		const char* lr_name;
		rt->retrieve_name(lr,lr_name);
		rt->attach_name(lp,lr_name);
		rt->attach_name(ip,lr_name);

		chksm = lw.chksm;
		GenCheckSum();

	}


	void LPWrapper::simpleSubPart(Context ctx, HighLevelRuntime* rt,LRWrapper lw,unsigned nSub)
	{
		*this = lw;
		Coloring cl;
		const unsigned lower_bound = ntotal/nSub;
		const unsigned upper_bound = lower_bound+1;
		const unsigned number_small = nSub - (ntotal % nSub);
		unsigned index = 0;
		for (unsigned n = 0; n< nSub; n++)
		{
			int num_elmts = n >= (nSub-number_small) ? lower_bound : upper_bound;
			assert((index+num_elmts) <= ntotal);

			cl[n].ranges.insert(std::pair<ptr_t,ptr_t>(index,index+num_elmts-1));
			index += num_elmts;
		}

		n_slice_dims = 1;
		slice_dims[0] = -1;
		slices[0].first = 0;
		slices[0].last = ntotal-1;
		slices[0].stride = lower_bound;
//		IndexPartition ip;

		ip = rt->create_index_partition(ctx,lr.get_index_space(),cl,true);
		lp = rt->get_logical_partition(ctx,lr,ip);

		lDom = rt->get_index_partition_color_space(ctx,ip);

		single_part = false;
		const char* lr_name;
		rt->retrieve_name(lr,lr_name);
		std::string lp_name = "LogicalPartition:"+std::string(lr_name);
		std::string ip_name = "IndexPartition:"+std::string(lr_name);
		rt->attach_name(lp,lp_name.c_str());
		rt->attach_name(ip,ip_name.c_str());

		GenCheckSum();


	}

	// Slice up each dimension such that there is one partition per element for the
	// requested dimensions
	void LPWrapper::simpleDimSlicePart(Context ctx, HighLevelRuntime* rt,LRWrapper lw,unsigned nDslice)
	{
		*this = lw;
		Coloring cl;
		n_slice_dims = nDslice;
		for(int i=0;i<nDslice;i++)
		{
			slice_dims[i] = ndims - nDslice + i;
			slices[i].first = 0;
			slices[i].last = dims[slice_dims[i]]-1;
			slices[i].stride = 1;
		}

		unsigned num_elmts = dims[0]-1;
		for(int i=1;i<ndims-nDslice;i++)
		{
			num_elmts += dims[i-1]*(dims[i]-1);
		}

		unsigned n = 0;
		for (unsigned i = 0; i<ntotal; i+=num_elmts)
		{
			assert((i+num_elmts) <= ntotal);

			cl[n].ranges.insert(std::pair<ptr_t,ptr_t>(i,i+num_elmts-1));
			n++;
		}
//		IndexPartition ip;
		ip = rt->create_index_partition(ctx,lr.get_index_space(),cl,true);
		lp = rt->get_logical_partition(ctx,lr,ip);

		lDom = rt->get_index_partition_color_space(ctx,ip);

		single_part = false;
		const char* lr_name;
		rt->retrieve_name(lr,lr_name);
		std::string lp_name = "LogicalPartition:"+std::string(lr_name);
		std::string ip_name = "IndexPartition:"+std::string(lr_name);
		rt->attach_name(lp,lp_name.c_str());
		rt->attach_name(ip,ip_name.c_str());

		GenCheckSum();

	}

	// Slice up each dimension such that there is one partition per element for the
	// requested dimensions
	void LPWrapper::customPart(Context ctx, HighLevelRuntime* rt,LRWrapper lw,Coloring cl)
	{
		*this = lw;
		n_slice_dims = 0;
//		IndexPartition ip;
		ip = rt->create_index_partition(ctx,lr.get_index_space(),cl,true);
		lp = rt->get_logical_partition(ctx,lr,ip);

		lDom = rt->get_index_partition_color_space(ctx,ip);

		single_part = false;
		const char* lr_name;
		rt->retrieve_name(lr,lr_name);
		std::string lp_name = "LogicalPartition:"+std::string(lr_name);
		std::string ip_name = "IndexPartition:"+std::string(lr_name);
		rt->attach_name(lp,lp_name.c_str());
		rt->attach_name(ip,ip_name.c_str());

		GenCheckSum();


	}


	void LPWrapper::GetMDColor(int iColor, unsigned* ids_out...)
	{
	    va_list args;
	    va_start(args,ids_out);

		unsigned prev_dims = 1;
		for(int i=0;i<n_slice_dims;i++)
		{
			unsigned tmp = (slices[i].last - slices[i].first)+1;
			unsigned nslices_dim = (tmp+slices[i].stride-1)/slices[i].stride;

			unsigned id = (iColor/prev_dims)%nslices_dim;
			unsigned& out_id = *(va_arg(args,unsigned*));
			out_id = id*slices[i].stride + slices[i].first;
			prev_dims *= nslices_dim;
		}

		 va_end(args);
	}

	void LPWrapper::GetMDColor(int iColor, unsigned out_ids[MAX_LEGION_MATRIX_DIMS])
	{


		unsigned prev_dims = 1;
		for(int i=0;i<n_slice_dims;i++)
		{
			unsigned tmp = (slices[i].last - slices[i].first)+1;
			unsigned nslices_dim = (tmp+slices[i].stride-1)/slices[i].stride;

			unsigned id = (iColor/prev_dims)%nslices_dim;
			out_ids[i] = id*slices[i].stride + slices[i].first;
			prev_dims *= nslices_dim;
		}

	}

	ColoredPoints<ptr_t> LPWrapper::calcColoringR(unsigned i0, unsigned i1, unsigned n,ColoredPoints<ptr_t> recurse)
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

	void LPWrapper::recurse_slice_dims(int iDim,
	                        std::map<unsigned,SliceParams> slice_map,
	                        ColoredPoints<ptr_t> recurse,
	                        Coloring& cl,
	                        unsigned& iCL)
	{

		if(iDim < 0)
		{
			cl[iCL] = recurse;
		}
		else
		{
			SliceParams slice = slice_map[iDim];
			unsigned first = slice.first;
			unsigned last = slice.last;
			unsigned stride = slice.stride;
			unsigned iCL_0= iCL;

			if(slice.b_slice)
			{ // Slice along this dim

				for(unsigned i=first;i<=last;i+=stride)
				{
					unsigned i_l = std::min(last,i+stride-1);

					ColoredPoints<ptr_t> pts_out = this->calcColoringR(i,i_l,dims[iDim],recurse);

//					printf("color %u dim %i from %u to %u\n",iCL,iDim,i,i_l);

					this->recurse_slice_dims(iDim-1,slice_map,pts_out,cl,iCL);

					iCL++;



				}
				iCL--;

//				printf("incrementing color from %i to %i\n", iCL_0, iCL);

			}
			else
			{
				ColoredPoints<ptr_t> pts_out = this->calcColoringR(first,last,dims[iDim],recurse);
				this->recurse_slice_dims(iDim-1,slice_map,pts_out,cl,iCL);
//				printf(" no slice color %u dim %i from %u to %u\n",iCL,iDim,first,last);

			}
		}
	}

	Coloring LPWrapper::mergeColorings(const Coloring _a,const Coloring _b)
	{


		Coloring a(_a);
		Coloring b(_b);

		auto pmin = [](uint ta, uint tb){return (uint)(ta < tb ? ta:tb);};
		auto pmax = [](uint ta, uint tb){return (uint)(ta < tb ? tb:ta);};

		auto li = [](uint ta){return (long long int)ta;};

		Coloring res(_a);
		Coloring b2(_b);

		for(auto cl_b : b)
		{

			auto cl_a = a.find(cl_b.first);
			if(cl_a != a.end())
			{

				ColoredPoints<ptr_t> cl_a2(cl_a->second);
				ColoredPoints<ptr_t> cl_b2(cl_b.second);
				// Check for points in A contained in ranges in B
				if(cl_a->second.points.size() > 0 && cl_b.second.ranges.size() > 0)
					for(auto pA : cl_a->second.points)
					{
						 ColoredPoints<ptr_t> cl_b3;
						cl_b3.ranges = cl_b2.ranges;
						for(auto rB : cl_b3.ranges)
							if((li(pA) <= li(rB.second)+1) && (li(pA) >= (li(rB.first)-1)))
							{
								printf("erasing point pA %u\n",(uint)pA);
								std::pair<ptr_t,ptr_t> r2(pmin(pA,rB.first),pmax(pA,rB.second));
								printf("inserting range rB %u - %u\n",(uint)r2.first,(uint)r2.second);

								cl_a2.points.erase(pA);
								if(r2 != rB)
								{
									cl_b2.ranges.erase(rB);
									cl_b2.ranges.insert(r2);
								}
							}
					}

				// Check for points in B contained in ranges in A
				if(cl_b.second.points.size() > 0 && cl_a->second.ranges.size() > 0)
					for(auto pB : cl_b.second.points)
					{
						ColoredPoints<ptr_t> cl_a3;
						cl_a3.ranges = cl_a2.ranges;
						for(auto rA : cl_a3.ranges)

							if((li(pB) <= (li(rA.second)+1)) && (li(pB) >= (li(rA.first)-1)))
							{
								std::pair<ptr_t,ptr_t> r2(pmin(pB,rA.first),pmax(pB,rA.second));

								printf("erasing point pB %u\n",(uint)pB);
								cl_b2.points.erase(pB);

								if(r2 != rA)
								{
									cl_a2.ranges.erase(rA);
									cl_a2.ranges.insert(r2);
								}
							}

					}

				// Check for overlapping ranges
				if(cl_b2.ranges.size() > 0 && cl_a2.ranges.size() > 0)
				{
					ColoredPoints<ptr_t> cl_b3;
					cl_b3.ranges = cl_b2.ranges;


					for(auto rB : cl_b3.ranges)
					{
						ColoredPoints<ptr_t> cl_a3;
						cl_a3.ranges = cl_a2.ranges;
						std::pair<ptr_t,ptr_t> o_left = rB;
						std::pair<ptr_t,ptr_t> o_right = rB;
						bool b_ins = false;

						for(auto rA : cl_a3.ranges)
						{
							std::pair<ptr_t,ptr_t> rB_merged(pmin(o_left.first,o_right.first),pmax(o_left.second,o_right.second));
							if((li(rB_merged.first) >= li(rA.first)) && (li(rB_merged.second) <= li(rA.second)))
							{
								cl_b2.ranges.erase(rB);
								b_ins = false;
								break;
							}
							else if((li(rB_merged.first) < li(rA.first)) && (li(rB_merged.second) > li(rA.second)))
							{	// range rB entirely contains range rA
								cl_a2.ranges.erase(rA);
								b_ins = true;
							}
							else if((li(rB_merged.first) >= (li(rA.first)-1)) && (li(rB_merged.first) <= (li(rA.second)+1)))
							{ // range rB starts in rA
								o_right = std::pair<ptr_t,ptr_t>(pmin(rA.first,rB_merged.first),pmax(rA.second,rB_merged.second));
								cl_a2.ranges.erase(rA);
								b_ins = true;
//								break;
							}
							else if((li(rB_merged.second) >= (li(rA.first)-1)) && (li(rB_merged.second) <= (li(rA.second)+1)))
							{
								o_left = std::pair<ptr_t,ptr_t>(pmin(rA.first,rB_merged.first),pmax(rA.second,rB_merged.second));
								cl_a2.ranges.erase(rA);

								b_ins = true;
//								break;
							}

						}

						if(b_ins)
						{
							cl_b2.ranges.erase(rB);
							cl_a2.ranges.insert(std::pair<ptr_t,ptr_t>(pmin(o_left.first,o_right.first),pmax(o_left.second,o_right.second)));

						}
					}

//					cl_b2 = cl_b3;
				}


				if(cl_b2.points.size() > 0)
					cl_a2.points.insert(cl_b2.points.begin(),cl_b2.points.end());

				if(cl_b2.ranges.size() > 0)
					cl_a2.ranges.insert(cl_b2.ranges.begin(),cl_b2.ranges.end());


				ColoredPoints<ptr_t> cl_out = cl_a2;

				// Check for points in B contained in ranges in A
				if(cl_out.points.size() > 0 && cl_out.ranges.size() > 0)
					for(auto pB : cl_a2.points)
					{
						ColoredPoints<ptr_t> cl_a3;
						cl_a3.ranges = cl_a2.ranges;
						for(auto rA : cl_a3.ranges)

							if((li(pB) <= (li(rA.second)+1)) && (li(pB) >= (li(rA.first)-1)))
							{
								std::pair<ptr_t,ptr_t> r2(pmin(pB,rA.first),pmax(pB,rA.second));

								printf("erasing point pB %u\n",(uint)pB);
								cl_out.points.erase(pB);

								if(r2 != rA)
								{
									cl_a2.ranges.erase(rA);
									cl_a2.ranges.insert(r2);
								}
							}

					}

				res[cl_b.first].points = cl_out.points;
				b2[cl_b.first].points = cl_b2.points;

				res[cl_b.first].ranges = cl_a2.ranges;
				b2[cl_b.first].ranges = cl_b2.ranges;

			}
			else
				res[cl_b.first] = cl_b.second;
		}

		return res;
	}


	ColoredPoints<ptr_t> LPWrapper::CalcColoring(int l,std::vector<std::pair<unsigned,unsigned>> range_map)
	const
	{
		assert(l < MAX_LEGION_MATRIX_DIMS);

		ColoredPoints<ptr_t> result;


		ColoredPoints<ptr_t> recurse;
		if(l<(ndims-1))
			recurse = CalcColoring(l+1,range_map);


		std::pair<unsigned,unsigned> my_range = range_map[l];
//		printf("working with dim %i, range %u to %u\n",l,my_range.first,my_range.second);

		if(l == ndims-1)
			result.ranges.insert(std::pair<ptr_t,ptr_t>(my_range.first,my_range.second));
		else
		{

		if(my_range.first == my_range.second)
		{
			unsigned ipoint = my_range.first;
			if(recurse.ranges.size() > 0)
			{
				for(auto range : recurse.ranges)
				{
					for(unsigned i=range.first;i<=(unsigned)range.second;i++)
						result.points.insert(ipoint+dims[l]*i);
				}
			}
			else if(recurse.points.size() > 0)
			{
				for(auto point : recurse.points)
				{
					result.points.insert(ipoint+dims[l]*(unsigned)point);
				}
			}
		}
		else
		{
			if(recurse.ranges.size() > 0)
			{
				for(auto range : recurse.ranges)
				{
					if(my_range.first == 0 && my_range.second == (dims[l]-1))
						result.ranges.insert(std::pair<ptr_t,ptr_t>(dims[l]*(unsigned)range.first,dims[l]-1+dims[l]*(unsigned)range.second));
					else
						for(unsigned i=range.first;i<=(unsigned)range.second;i++)
							result.ranges.insert(std::pair<ptr_t,ptr_t>(my_range.first+dims[l]*i,my_range.second+dims[l]*i));
				}
			}
			else if(recurse.points.size() > 0)
			{
				for(auto point : recurse.points)
				{
					result.ranges.insert(std::pair<ptr_t,ptr_t>(my_range.first+dims[l]*(unsigned)point,my_range.second+dims[l]*(unsigned)point));
				}
			}
			else
				result.ranges.insert(std::pair<ptr_t,ptr_t>(my_range.first,my_range.second));
		}
		}


		return result;
	}

} /* namespace Dragon */
