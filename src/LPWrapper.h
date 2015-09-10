/*
 * LRWrapper.h
 *
 *  Created on: Jul 7, 2015
 *      Author: payne
 */

#ifndef LPWRAPPER_H_
#define LPWRAPPER_H_

//#include <legion.h>
//#include <typeinfo>
//#ifdef USE_CUDA_FKOP
//#include <cuda.h>
//#include <cuda_runtime.h>
//#else
//#define __host__
//#define __device__
//#endif
//#include "legion_tasks.h"
//#include <sstream>
//#include <unistd.h>
//#include <stdio.h>
//#include <stdlib.h>
////#include "LRWrapper.h"

#include "wrapper_defines.h"


using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;
namespace Dragon
{
	__inline__
	unsigned atou(const char* a)
	{char* pEnd; return strtoul(a,&pEnd,10);}

	class LRWrapper;



	class LPWrapper;
	struct DimOptions
	{
		char c[25];
	};



//	template<int iDim>
//	class ColorExtractor;
//
//	template<>
//	class ColorExtractor<MAX_LEGION_MATRIX_DIMS>
//	{
//	public:
//		template<class... Args_out>
//		static ColoredPoints<ptr_t> prop_args(LPWrapper& lpw,std::map<unsigned,DimOptions> dim_options,Args_out... args);
//
//	};
//
////	template<>
////	class ColorExtractor<0>
////	{
////	public:
////		static ColoredPoints<ptr_t> prop_args(LPWrapper& lpw,std::map<unsigned,DimOptions> dim_options);
////
////	};
//
//	template<int iDim>
//	class ColorExtractor
//	{
//	public:
//
//
//		template<class... Args_out>
//		static ColoredPoints<ptr_t> prop_args(LPWrapper& lpw,std::map<unsigned,DimOptions> dim_options,Args_out... args);
//
//		static ColoredPoints<ptr_t> prop_args0(LPWrapper& lpw,std::map<unsigned,DimOptions> dim_options);
//
//	};

//	class LPWrapper;
//	template<int iDim,class... Args_out>
//	struct Bar
//	{
//		 static ColoredPoints<ptr_t> prop_args(LPWrapper& lpw,std::map<unsigned,DimOptions> dim_options,Args_out... args);
//	};
//
//	template<class... Args_out>
//	struct Bar<MAX_LEGION_MATRIX_DIMS,Args_out...>
//	{
//		 static ColoredPoints<ptr_t> prop_args(LPWrapper& lpw,std::map<unsigned,DimOptions> dim_options,Args_out... args);
//	};
//
//	template<int iDim>
//	struct Bar<iDim>
//	{
//		 static ColoredPoints<ptr_t> prop_args(LPWrapper& lpw,std::map<unsigned,DimOptions> dim_options);
//	};



	class LPWrapper
	{
	public:

		LPWrapper()
		{

			memset(this,0,sizeof(LPWrapper));
//			memset(slices,0,MAX_LEGION_MATRIX_DIMS*sizeof(SliceParams));
//			memset(slice_dims,0,MAX_LEGION_MATRIX_DIMS*sizeof(int8_t));
//			memset(dims,0,MAX_LEGION_MATRIX_DIMS*sizeof(size_t));
//			memset(fids,0,MAX_LEGION_MATRIX_DIMS*sizeof(uint8_t));
//			memset(f_types,0,MAX_LEGION_MATRIX_DIMS*sizeof(size_t));
//
			chksm = 0;
		}
//		LPWrapper(const LPWrapper& _in)
//		{
//			memset(this,0,sizeof(LPWrapper));
//
//			*this = _in;
//		}


		class SliceParams
		{
		public:
			unsigned first;
			unsigned last;
			unsigned stride;
			bool b_slice;
		};


		template<class... Args>
		void extract_options(int iDim,
		                     std::map<unsigned,SliceParams>& split_dims,
		                     const char* arg1,Args... rest)
		{
			// We split the dimension evenly
			const char* s = strchr(arg1,'%');
			if(s == arg1)
			{
				SliceParams pout;
				pout.first = 0;
				pout.last = dims[iDim]-1;
				pout.stride = atou(arg1+1);
				pout.b_slice = true;

				printf("stride[%i] = %u\n",iDim,pout.stride);

				split_dims[iDim] = pout;
			}
			else if(s == NULL) // Can't find the split flag
			{
				const char* second = strchr(arg1,':');
				unsigned ifirst;
				unsigned ilast;
				if(second == arg1)
					ifirst=0;
				else if(second == NULL)
					ifirst=atou(arg1);
				else
					ifirst=atou(arg1);

				if(second == arg1+(strlen(arg1)-1))
					ilast = dims[iDim] - 1;
				else if(second == NULL)
					ilast = ifirst;
				else
					ilast = atou(second+1);

				SliceParams pout;
				pout.first = ifirst;
				pout.last = ilast;
				pout.stride = ilast-ifirst+1;
				pout.b_slice = false;
				split_dims[iDim] = pout;

				printf("stride[%i] = %u\n",iDim,pout.stride);

			}
			else // Found a split flag, but we are splitting a specified range
			{
				char tmp[64];
				strcpy(tmp,arg1);
				tmp[s-arg1] = '\0';

				SliceParams pout;
				pout.first = std::max(atou(arg1),unsigned(0));
				pout.last = std::min((unsigned)atou(strchr(arg1,':')+1),(unsigned)(dims[iDim]-1));
				pout.stride = atou(s+1);
				pout.b_slice = true;
				split_dims[iDim] = pout;
				printf("stride[%i] = %u\n",iDim,pout.stride);

			}

			extract_options(iDim+1,split_dims,rest...);
		}

		template<class... Args>
		void extract_options(int iDim,
		                     std::map<unsigned,SliceParams>& split_dims,
		                     unsigned arg1,Args... rest)
		{
			SliceParams pout;
			pout.first = arg1;
			pout.last = arg1;
			pout.stride = 1;
			pout.b_slice = false;

			split_dims[iDim] = pout;
			extract_options(iDim+1,split_dims,rest...);
		}


		void extract_options(int iDim,
		                     std::map<unsigned,SliceParams>& split_dims,
		                     const char* arg1)
		{
			// We split the dimension evenly
			const char* s = strchr(arg1,'%');
			if(s == arg1)
			{
				SliceParams pout;
				pout.first = 0;
				pout.last = dims[iDim]-1;
				pout.stride = atou(arg1+1);
				pout.b_slice = true;

				printf("stride[%i] = %u\n",iDim,pout.stride);

				split_dims[iDim] = pout;
			}
			else if(s == NULL) // Can't find the split flag
			{
				const char* second = strchr(arg1,':');
				unsigned ifirst;
				unsigned ilast;
				if(second == arg1)
					ifirst=0;
				else if(second == NULL)
					ifirst=atou(arg1);
				else
					ifirst=atou(arg1);

				if(second == arg1+(strlen(arg1)-1))
					ilast = dims[iDim] - 1;
				else if(second == NULL)
					ilast = ifirst;
				else
					ilast = atou(second+1);

				SliceParams pout;
				pout.first = ifirst;
				pout.last = ilast;
				pout.stride = ilast-ifirst+1;
				pout.b_slice = false;

				split_dims[iDim] = pout;
				printf("stride[%i] = %u\n",iDim,pout.stride);



			}
			else // Found a split flag, but we are splitting a specified range
			{
				char tmp[64];
				strcpy(tmp,arg1);
				tmp[s-arg1] = '\0';

				SliceParams pout;
				pout.first = std::max(atou(arg1),unsigned(0));
				pout.last = std::min((unsigned)atou(strchr(arg1,':')+1),(unsigned)(dims[iDim]-1));
				pout.stride = atou(s+1);
				pout.b_slice = true;
				split_dims[iDim] = pout;
				printf("stride[%i] = %u\n",iDim,pout.stride);

			}

		}

		void extract_options(int iDim,
		                     std::map<unsigned,SliceParams>& split_dims,
		                     unsigned arg1)
		{
			SliceParams pout;
			pout.first = arg1;
			pout.last = arg1;
			pout.stride = 1;
			pout.b_slice = false;

			split_dims[iDim] = pout;
		}

		void recurse_slice_dims(int iDim,
		                        std::map<unsigned,SliceParams> slice_map,
		                        ColoredPoints<ptr_t> recurse,
		                        Coloring& cl,
		                        unsigned& iCL);

		ColoredPoints<ptr_t> calcColoringR(unsigned i0, unsigned i1, unsigned n,ColoredPoints<ptr_t> recurse);


		static Coloring mergeColorings(const Coloring _a,const Coloring _b);

		template<class... Args>
		Coloring slicedPart(Context ctx, HighLevelRuntime* rt,LRWrapper lw,Args... args)
		{

			*this = lw;
			Coloring cl;
			std::map<unsigned,SliceParams> split_dims;

			// parse the arguments
//			printf("Extracting Arguments\n");
			extract_options(0,split_dims,args...);

			{unsigned i = 0;
			for(auto slice : split_dims)
			{
				if(slice.second.b_slice)
				{
					slices[i] = slice.second;
					slice_dims[i] = slice.first;
					i++;
				}
			}
				n_slice_dims = i+1;
			}
			unsigned iColor = 0;
//			printf("Looping Colors\n");
			// Fill the coloring
			ColoredPoints<ptr_t> dummy;
			recurse_slice_dims(ndims-1,split_dims,dummy,cl,iColor);
//			IndexPartition ip;

			ip = rt->create_index_partition(ctx,lw.is,cl,true);
			lp = rt->get_logical_partition(ctx,lr,ip);

			lDom = rt->get_index_partition_color_space(ctx,ip);

			single_part = false;

			for(auto color : cl)
			{
				printf("color %i:",color.first);
				for(auto pt : color.second.points)
				{
					printf(" %u,",(unsigned)pt);
				}

				for(auto rng : color.second.ranges)
				{
					printf(" %u - %u,",(unsigned)rng.first,(unsigned)rng.second);
				}
				printf("\n");
			}
			const char* lr_name;
			rt->retrieve_name(lr,lr_name);

			std::string lp_name = "LogicalPartition:"+std::string(lr_name);
			std::string ip_name = "IndexPartition:"+std::string(lr_name);
			rt->attach_name(lp,lp_name.c_str());
			rt->attach_name(ip,ip_name.c_str());

			GenCheckSum();
			return cl;

		}

		template<class... Args>
		Coloring genColoring(Context ctx, HighLevelRuntime* rt,LRWrapper lw,Args... args)
		{
			*this = lw;
			Coloring cl;

			std::map<unsigned,SliceParams> split_dims;

			// parse the arguments
//			printf("Extracting Arguments\n");
			extract_options(0,split_dims,args...);

			{unsigned i = 0;
			for(auto slice : split_dims)
			{
				if(slice.second.b_slice)
				{
					slices[i] = slice.second;
					slice_dims[i] = slice.first;
					i++;
				}
			}
				n_slice_dims = i+1;
			}
			unsigned iColor = 0;
//			printf("Looping Colors\n");
			// Fill the coloring
			ColoredPoints<ptr_t> dummy;
			recurse_slice_dims(ndims-1,split_dims,dummy,cl,iColor);

			return cl;
		}

		void singlePart(Context ctx, HighLevelRuntime* rt,LRWrapper lw,unsigned iCl = 0);

		void copiedPart(Context ctx, HighLevelRuntime* rt,LRWrapper lw,unsigned n);

		void simpleSubPart(Context ctx, HighLevelRuntime* rt,LRWrapper lw,unsigned nSub);

		// Slice up each dimension such that there is one partition per element for the
		// requested dimensions
		void simpleDimSlicePart(Context ctx, HighLevelRuntime* rt,LRWrapper lw,unsigned nDslice);

		// Slice up each dimension such that there is one partition per element for the
		// requested dimensions
		void customPart(Context ctx, HighLevelRuntime* rt,LRWrapper lw,Coloring _cl);


		void GetMDColor(int iColor, unsigned* ids_out...);

		void GetMDColor(int iColor, unsigned out_ids[MAX_LEGION_MATRIX_DIMS]);

		void GetMDIndex(int idx, unsigned out_ids[MAX_LEGION_MATRIX_DIMS]);


		void GenCheckSum();

		LPWrapper& operator=(const LRWrapper& _in);

//		LPWrapper& operator=(const LPWrapper& _in);


		operator LRWrapper();

		LogicalRegion lr;
		LogicalPartition lp;
		IndexPartition ip;
//		Coloring _cl;
		Domain lDom;
		uint8_t ndims;
		int8_t n_slice_dims;
		uint8_t nfields;
		size_t ntotal;
		SliceParams slices[MAX_LEGION_MATRIX_DIMS];
		int8_t slice_dims[MAX_LEGION_MATRIX_DIMS];


		size_t dims[MAX_LEGION_MATRIX_DIMS];
		uint8_t fids[MAX_LEGION_MATRIX_FIELDS];
		size_t f_types[MAX_LEGION_MATRIX_FIELDS];

		bool single_part;
		long long chksm;
	private:
		bool lrCreated;

	public:

		bool operator<(const Dragon::LPWrapper& in)
		const
		{

//			printf("chksm = %lu == %lu\n",chksm,in.chksm);

			bool res = false;
			if(lr != in.lr)
				if(lr < in.lr)
					res = true;
				else
					res = false;
			else if(lp != in.lp)
				if(lp < in.lp)
					res = true;
				else
					res = false;
			else if(lDom != in.lDom)
				if(lDom < in.lDom)
					return true;
				else
					return false;
			else if(chksm < in.chksm)
				res = true;

//			printf("chksm = %lu == %lu, res = %u\n",chksm,in.chksm,res);

			return res;
		}
		bool operator>(const Dragon::LPWrapper& in)
		const
		{

//			printf("chksm = %lu == %lu\n",chksm,in.chksm);

			if(!((lr < in.lr) || (lr == in.lr)) )
				return true;
			else if(chksm > in.chksm)
				return true;
			else if(ndims > in.ndims)
				return true;
			else if(nfields > in.nfields)
				return true;
			else if(ntotal > in.ntotal)
				return true;
			else if(single_part > in.single_part)
				return true;

			return false;
		}

//		inline LPWrapper& operator=(const LPWrapper &rhs)
//		{
//			lr = rhs.lr;
//			lp = rhs.lp;
//			lDom = rhs.lDom;
//			ndims = rhs.ndims;
//			n_slice_dims = rhs.n_slice_dims;
//			nfields = rhs.nfields;
//			ntotal = rhs.ntotal;
//			memcpy(slices,rhs.slices,MAX_LEGION_MATRIX_DIMS*sizeof(SliceParams));
//			memcpy(slice_dims,rhs.slice_dims,MAX_LEGION_MATRIX_DIMS*sizeof(int8_t));
//
//
//			memcpy(dims,rhs.dims,MAX_LEGION_MATRIX_DIMS*sizeof(size_t));
//			memcpy(fids,rhs.fids,MAX_LEGION_MATRIX_DIMS*sizeof(uint8_t));
//			memcpy(f_types,rhs.f_types,MAX_LEGION_MATRIX_DIMS*sizeof(size_t));
//
//			lrCreated = rhs.lrCreated;
//			chksm = rhs.chksm;
//			single_part = rhs.single_part;
//			return *this;
//		}

		bool operator==(const Dragon::LPWrapper& in)
		const
		{
			bool res = ((lr == in.lr)  && (lDom == in.lDom) && (chksm == in.chksm) && (single_part == in.single_part));
//			printf("chksm = %lu == %lu, res = %u\n",chksm,in.chksm,res);
			return res;

		}

		bool operator==(const Dragon::LPWrapper& in)
		{
			bool res = ((lr == in.lr)  && (lDom == in.lDom) && (chksm == in.chksm) && (single_part == in.single_part));
//			printf("chksm = %lu == %lu, res = %u\n",chksm,in.chksm,res);
			return res;
		}

		template<class... Args> __host__ __device__
		int operator()(Args... rest)
		{
			return expand(0,rest...);
		}

		template<class... Args> __host__ __device__
		const int operator()(Args... rest)
		const
		{
			return expand(0,rest...);
		}

		int expand(int l,int i)
		{
			return i;
		}

		template<class... Args>
		int expand(int l,int i, Args... rest)
		{
			assert(l < MAX_LEGION_MATRIX_DIMS);
			return i+dims[l]*expand(l+1,rest...);
		}

		// This generates a coloring based on the arguments passed in
		// an argument of ":" means that the full range for that dimension will be used
		// an argument of int means that a single point for that dimension will be used
		template<class... Args>
		ColoredPoints<ptr_t> GetColoring(Args... args)
		const
		{
			return CalcColoring(0,args...);
		}

		template<class... Args>
		ColoredPoints<ptr_t> CalcColoring(int l,const char* whole_dim,Args... args)
		const
		{
			assert(l < MAX_LEGION_MATRIX_DIMS);

			ColoredPoints<ptr_t> recurse = CalcColoring(l+1,args...);

			ColoredPoints<ptr_t> result;

			if(recurse.ranges.size() > 0)
			{
				for(auto range : recurse.ranges)
				{
					result.ranges.insert(std::pair<ptr_t,ptr_t>(dims[l]*(unsigned)range.first,dims[l]-1+dims[l]*(unsigned)range.second));
				}
			}
			else if(recurse.points.size() > 0)
			{
				for(auto point : recurse.points)
				{
					result.ranges.insert(std::pair<ptr_t,ptr_t>(dims[l]*(unsigned)point,dims[l]-1+dims[l]*(unsigned)point));
				}
			}

			return result;

		}

		template<class... Args>
		ColoredPoints<ptr_t> CalcColoring(int l,unsigned ipoint,Args... args)
		const
		{
			assert(l < MAX_LEGION_MATRIX_DIMS);

			ColoredPoints<ptr_t> recurse = CalcColoring(l+1,args...);

			ColoredPoints<ptr_t> result;

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

			return result;
		}

		template<class... Args>
		ColoredPoints<ptr_t> CalcColoring(int l,std::initializer_list<unsigned> nums,Args... args)
		const
		{
			assert(l < MAX_LEGION_MATRIX_DIMS);
			assert(nums.size() > 1);

			ColoredPoints<ptr_t> recurse = CalcColoring(l+1,args...);

			ColoredPoints<ptr_t> result;

			if(recurse.ranges.size() > 0)
			{
				for(auto range : recurse.ranges)
				{
					for(unsigned i=range.first;i<=(unsigned)range.second;i++)
						result.ranges.insert(std::pair<ptr_t,ptr_t>(nums.begin()[0]+dims[l]*i,nums.begin()[1]+dims[l]*i));
				}
			}
			else if(recurse.points.size() > 0)
			{
				for(auto point : recurse.points)
				{
					result.ranges.insert(std::pair<ptr_t,ptr_t>(nums.begin()[0]+dims[l]*(unsigned)point,nums.begin()[1]+dims[l]*(unsigned)point));
				}
			}

			return result;
		}

		ColoredPoints<ptr_t> CalcColoring(int l,const char* whole_dim)
		const
		{
			assert(l < MAX_LEGION_MATRIX_DIMS);

			ColoredPoints<ptr_t> result;

			result.ranges.insert(std::pair<ptr_t,ptr_t>(0,dims[l]-1));

			return result;
		}

		ColoredPoints<ptr_t> CalcColoring(int l,unsigned ipoint)
		const
		{
			assert(l < MAX_LEGION_MATRIX_DIMS);

			ColoredPoints<ptr_t> result;

			result.points.insert(ipoint);

			return result;
		}

		ColoredPoints<ptr_t> CalcColoring(int l,std::initializer_list<unsigned> nums)
		const
		{
			assert(l < MAX_LEGION_MATRIX_DIMS);
			assert(nums.size() > 1);

			ColoredPoints<ptr_t> result;

			result.ranges.insert(std::pair<ptr_t,ptr_t>(nums.begin()[0],nums.begin()[1]));

			return result;
		}



		ColoredPoints<ptr_t> CalcColoring(int l,std::vector<std::pair<unsigned,unsigned>> range_map)const;

		~LPWrapper(){};



	};

	struct LPArg
	{
	public:

		LPArg(){}
		LPArg(LPWrapper _lr, FieldID _fid,PrivilegeMode _priv,CoherenceProperty _co,RegionFlags _flags):
			lp(_lr),fid(_fid),priv(_priv),co(_co),flags(_flags){}
		LPWrapper lp;
		FieldID fid;
		PrivilegeMode priv;
		CoherenceProperty co;
		RegionFlags flags;
	};



} /* namespace Dragon */

#endif /* LPWRAPPER_H_ */
