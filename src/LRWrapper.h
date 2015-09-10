/*
 * LRWrapper.h
 *
 *  Created on: Jul 7, 2015
 *      Author: payne
 */

#ifndef LRWRAPPER_H_
#define LRWRAPPER_H_


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

#include "wrapper_defines.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;
namespace Dragon
{

	class LPWrapper;

	class LRWrapper
	{
	public:


		// Creates a logical region
		// note: field_stuff is basically a pair in the form of "int fid, T()"
		// so an example would be wrapper.create(ctx,rt,{nx,ny,nz},x,double(),i,int());
		// where 'x' and 'i' are enums or integers.
		template<class ...Args>
		void create(Context ctx, HighLevelRuntime* rt,const char* name,const std::initializer_list<int> _dims,
		            Args... field_stuff)
		{
			create(ctx,rt,_dims,field_stuff...);
			rt->attach_name(lr,name);
			rt->attach_name(is,name);

		}

		// Creates a logical region
		// note: field_stuff is basically a pair in the form of "int fid, T()"
		// so an example would be wrapper.create(ctx,rt,{nx,ny,nz},x,double(),i,int());
		// where 'x' and 'i' are enums or integers.
		template<class ...Args>
		void create(Context ctx, HighLevelRuntime* rt,const std::initializer_list<int> _dims, Args... field_stuff)
		{
			{
				{
			int i=0;
			ntotal=1;
			for(auto d : _dims)
			{
				// Make sure that our dimension is greater than 0
				assert(d > 0);
				// make sure we don't have too many dimensions
				assert(i < MAX_LEGION_MATRIX_DIMS);
				dims[i] = d;
				ntotal *= d;
				i++;
			}
				}
			ndims = _dims.size();


			is = rt->create_index_space(ctx,ntotal);

	// Allocate the index space
			{
				IndexAllocator ptcl_allocator = rt->create_index_allocator(ctx, is);
				ptcl_allocator.alloc(ntotal);

			}

			FieldSpace fs = rt->create_field_space(ctx);

			FieldAllocator allocator =
			  rt->create_field_allocator(ctx, fs);

			AllocateFields(ctx,rt,allocator,0,field_stuff...);

			printf("Creating Logical Region with %lu total points\n",ntotal*nfields);


			lr = rt->create_logical_region(ctx, is, fs);
			GenCheckSum();
//			size_t dim_sm = 0;
//			size_t fid_sm = 0;
//			size_t ft_sm = 0;
//			size_t prm_sm = 0;
//			for(int i=0;i<ndims;i++)
//			{
//				dim_sm ^= dims[i];
//			}
//			for(int i=0;i<ndims;i++)
//			{
//				fid_sm ^= fids[i];
//				ft_sm ^= f_types[i];
//			}
//
//			prm_sm ^= ndims;
//			prm_sm ^= nfields;
//			prm_sm ^= ntotal;
//
//			chksm = 0;
//			chksm ^= (dim_sm << __builtin_clzll(dim_sm));
//			chksm ^= (fid_sm << __builtin_clzll(fid_sm));
//			chksm ^= (ft_sm << __builtin_clzll(ft_sm));
//			chksm ^= (prm_sm << __builtin_clzll(prm_sm));



			}
		}

		// Creates a deep copy of an LRWrapper, same dims
		void createCopy(Context ctx, HighLevelRuntime* rt,LRWrapper _in, std::vector<FieldID> _fids)
		{

			FieldAllocator allocator;
			nfields = _fids.size();
			unsigned l =0;
			for(auto fid : _fids)
			{
				size_t f_size = rt->get_field_size(ctx,_in.lr.get_field_space(),fid);

				allocator.allocate_field(f_size,fid);
				fids[l] = fid;

				int l_in = -1;
				for(int l2 = 0;l2<_in.nfields;l2++)
					if(_in.fids[l2] == fid)
					{	l_in = l2; break;}

				assert(l_in >= 0);
				f_types[l] = _in.f_types[l_in];
				l++;
			}
		}

		template<class T,class... Args>
		void AllocateFields(Context ctx,HighLevelRuntime* rt,
		                    FieldAllocator& allocator,
		                    int l,int fid,T x,Args... rest)
		{
			allocator.allocate_field(sizeof(T),fid);
			// Make sure we don't have too many fields
			assert(l < MAX_LEGION_MATRIX_FIELDS);
			fids[l] = fid;
			f_types[l] = typeid(T).hash_code();

			AllocateFields(ctx,rt,allocator,l+1,rest...);
		}

		template<class T>
		void AllocateFields(Context ctx,HighLevelRuntime* rt,
		                    FieldAllocator& allocator,
		                    int l,int fid,T x)
		{
			allocator.allocate_field(sizeof(T),fid);
			fids[l] = fid;
			f_types[l] = typeid(T).hash_code();
			nfields = l+1;
		}

		LogicalRegion lr;
		IndexSpace is;
		uint8_t ndims;
		uint8_t nfields;
		size_t ntotal;
		size_t dims[MAX_LEGION_MATRIX_DIMS];
		uint8_t fids[MAX_LEGION_MATRIX_FIELDS];
		size_t f_types[MAX_LEGION_MATRIX_FIELDS];

	private:
		bool lrCreated;
	public: long long chksm;
	public:

		void GenCheckSum();


		bool operator<(const Dragon::LRWrapper& in)
		const
		{
			if(lr != in.lr)
				if(lr < in.lr)
					return true;
				else
					return false;
			else if(chksm < in.chksm)
				return true;

			return false;
		}

		bool operator==(const Dragon::LRWrapper& in)
		const
		{
			return (lr == in.lr) && (chksm == in.chksm);

		}

		bool operator==(const Dragon::LRWrapper& in)
		{
			return (lr == in.lr) && (chksm == in.chksm);

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


//		template<class... Args>
//		ColoredPoints<ptr_t> CalcColoring(int l,const char* whole_dim,Args... args)
//		const
//		{
//			assert(l < MAX_LEGION_MATRIX_DIMS);
//
//			ColoredPoints<ptr_t> recurse = CalcColoring(l+1,args...);
//
//			ColoredPoints<ptr_t> result;
//
//			if(recurse.ranges.size() > 0)
//			{
//				for(auto range : recurse.ranges)
//				{
//					result.ranges.insert(std::pair<ptr_t,ptr_t>(dims[l]*(unsigned)range.first,dims[l]-1+dims[l]*(unsigned)range.second));
//				}
//			}
//			else if(recurse.points.size() > 0)
//			{
//				for(auto point : recurse.points)
//				{
//					result.ranges.insert(std::pair<ptr_t,ptr_t>(dims[l]*(unsigned)point,dims[l]-1+dims[l]*(unsigned)point));
//				}
//			}
//
//			return result;
//
//		}
//
//		template<class... Args>
//		ColoredPoints<ptr_t> CalcColoring(int l,unsigned ipoint,Args... args)
//		const
//		{
//			assert(l < MAX_LEGION_MATRIX_DIMS);
//
//			ColoredPoints<ptr_t> recurse = CalcColoring(l+1,args...);
//
//			ColoredPoints<ptr_t> result;
//
//			if(recurse.ranges.size() > 0)
//			{
//				for(auto range : recurse.ranges)
//				{
//					for(unsigned i=range.first;i<=(unsigned)range.second;i++)
//						result.points.insert(ipoint+dims[l]*i);
//				}
//			}
//			else if(recurse.points.size() > 0)
//			{
//				for(auto point : recurse.points)
//				{
//					result.points.insert(ipoint+dims[l]*(unsigned)point);
//				}
//			}
//
//			return result;
//		}
//
//		template<class... Args>
//		ColoredPoints<ptr_t> CalcColoring(int l,std::initializer_list<unsigned> nums,Args... args)
//		const
//		{
//			assert(l < MAX_LEGION_MATRIX_DIMS);
//			assert(nums.size() > 1);
//
//			ColoredPoints<ptr_t> recurse = CalcColoring(l+1,args...);
//
//			ColoredPoints<ptr_t> result;
//
//			if(recurse.ranges.size() > 0)
//			{
//				for(auto range : recurse.ranges)
//				{
//					for(unsigned i=range.first;i<=(unsigned)range.second;i++)
//						result.ranges.insert(std::pair<ptr_t,ptr_t>(nums.begin()[0]+dims[l]*i,nums.begin()[1]+dims[l]*i));
//				}
//			}
//			else if(recurse.points.size() > 0)
//			{
//				for(auto point : recurse.points)
//				{
//					result.ranges.insert(std::pair<ptr_t,ptr_t>(nums.begin()[0]+dims[l]*(unsigned)point,nums.begin()[1]+dims[l]*(unsigned)point));
//				}
//			}
//
//			return result;
//		}
//
//		ColoredPoints<ptr_t> CalcColoring(int l,const char* whole_dim)
//		const
//		{
//			assert(l < MAX_LEGION_MATRIX_DIMS);
//
//			ColoredPoints<ptr_t> result;
//
//			result.ranges.insert(std::pair<ptr_t,ptr_t>(0,dims[l]-1));
//
//			return result;
//		}
//
//		ColoredPoints<ptr_t> CalcColoring(int l,unsigned ipoint)
//		const
//		{
//			assert(l < MAX_LEGION_MATRIX_DIMS);
//
//			ColoredPoints<ptr_t> result;
//
//			result.points.insert(ipoint);
//
//			return result;
//		}
//
//		ColoredPoints<ptr_t> CalcColoring(int l,std::initializer_list<unsigned> nums)
//		const
//		{
//			assert(l < MAX_LEGION_MATRIX_DIMS);
//			assert(nums.size() > 1);
//
//			ColoredPoints<ptr_t> result;
//
//			result.ranges.insert(std::pair<ptr_t,ptr_t>(nums.begin()[0],nums.begin()[1]));
//
//			return result;
//		}

		~LRWrapper(){};

		LRWrapper& operator=(const LPWrapper& _in);


	};

	template<class ...Args>
	LRWrapper createLRWrapper(Context ctx, HighLevelRuntime* rt,const std::initializer_list<int> _dims, Args... field_stuff)
	{
		LRWrapper result;
		result.create(ctx,rt,"unnamed",_dims,field_stuff...);
		return result;
	}

	template<class ...Args>
	LRWrapper createLRWrapper(Context ctx, HighLevelRuntime* rt,const char* name,const std::initializer_list<int> _dims, Args... field_stuff)
	{
		LRWrapper result;
		result.create(ctx,rt,name,_dims,field_stuff...);
		return result;
	}



} /* namespace Dragon */

#endif /* LRWRAPPER_H_ */
