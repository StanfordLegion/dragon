/*
 * LegionMatrix.h
 *
 *  Created on: Jul 9, 2015
 *      Author: payne
 *
 * Copyright (c) 2014-2015 Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the  LANL Contributions to Legion (C15091) project.
 * See the LICENSE.txt file at the top-level directory of this distribution.
 */
#include "LRWrapper.h"

#ifndef LEGION_MATRIX_H_
#define LEGION_MATRIX_H_

namespace Dragon
{

	template<typename T>
	class lrInterface
	{
	public:
		RegionAccessor<AccessorType::SOA<sizeof(T)>,T> Taccessor;
		const ptr_t aPoint;

		__host__ __device__
		lrInterface(RegionAccessor<AccessorType::SOA<sizeof(T)>,T> _accessor,const ptr_t _a) :
			Taccessor(_accessor),aPoint(_a)
		{

		}



		__host__ __device__
		lrInterface<T>& operator=(const T& b)
		{
			Taccessor.write(aPoint,b);
			return *this;
		}

		__host__ __device__
		operator T() const { return Taccessor.read(aPoint);}

		__host__ __device__
		const T cast()const{ return Taccessor.read(aPoint);}

		__host__ __device__
		const T* ptr()const{ return Taccessor.ptr(aPoint);}

		__host__ __device__
		T* ptr(){ return Taccessor.ptr(aPoint);}
	};

	template<class T>
	class lrAccessor
	{
	public:

		 RegionAccessor<AccessorType::SOA<sizeof(T)>,T> accessor;
		 ptr_t first_elem;

		__host__ __device__
		lrAccessor(){}

		lrAccessor(const PhysicalRegion &region, const int FID);


		__host__ __device__
		lrInterface<T> operator()(int i)
		{
	//		DomainPoint apoint = DomainPoint::from_point<1>(Point<1>(i));
			return  lrInterface<T>(accessor,first_elem+i);
		}

		__host__ __device__
		const lrInterface<T> operator()(int i)
		const
		{
	//		DomainPoint apoint = DomainPoint::from_point<1>(Point<1>(i));

			return  lrInterface<T>(accessor,first_elem+i);
		}
	};

	template<class T>
	class LegionMatrix
	{
	public:
		const int fid;
		size_t dims[MAX_LEGION_MATRIX_DIMS];

		LegionMatrix() : fid(-1) {}

		LegionMatrix(const LRWrapper& wrapper,const PhysicalRegion &region,const int _FID) : fid(_FID), acc(region,_FID)
		{

			printf("Creating %s of field id %i with %i dims\n",typeid(decltype(*this)).name(),_FID,wrapper.ndims);
			memcpy(dims,wrapper.dims,wrapper.ndims*sizeof(size_t));
		}

		LegionMatrix(Context ctx,const LRWrapper& wrapper,const int _FID) : fid(_FID)
		{
			Task* task = ctx->as_mappable_task();
			PhysicalRegion pr_angFluxes;
			memcpy(dims,wrapper.dims,wrapper.ndims*sizeof(size_t));

			for(int i=0;i<task->regions.size();i++)
			{
				if(task->regions[i].region == wrapper.lr || task->regions[i].parent == wrapper.lr)
				{
					if(task->regions[i].has_field_privilege(_FID))
					{
					acc = lrAccessor<T>(ctx->get_physical_region(i),_FID);
					break;
					}
				}
			}
		}

		LegionMatrix(Context ctx,const LRWrapper& wrapper,const int _FID,PrivilegeMode priv, CoherenceProperty co) : fid(_FID)
		{
			Task* task = ctx->as_mappable_task();
			PhysicalRegion pr;
			memcpy(dims,wrapper.dims,wrapper.ndims*sizeof(size_t));

			for(int i=0;i<task->regions.size();i++)
			{
				if(task->regions[i].region == wrapper.lr || task->regions[i].parent == wrapper.lr)
				{
					if(task->regions[i].has_field_privilege(_FID))
					{
						pr = ctx->get_physical_region(i);
					acc = lrAccessor<T>(pr,_FID);
					break;
					}
				}
			}

			if(!pr.is_mapped())
			{
				// We have to map it
				Processor proc = Processor::get_executing_processor();
				HighLevelRuntime* rt = HighLevelRuntime::get_runtime(proc);

				RegionRequirement rr(wrapper.lr,priv,co,wrapper.lr);
				rr.add_field(fid);
				pr = rt->map_region(ctx,rr);
				acc = lrAccessor<T>(pr,_FID);
			}
		}




		LegionMatrix(const LRWrapper& wrapper,lrAccessor<T> _acc) : fid(0),acc(_acc)
		{
			memcpy(dims,wrapper.dims,wrapper.ndims*sizeof(size_t));
		}



		lrAccessor<T> acc;

		template<class... Args> __host__ __device__
		lrInterface<T> operator()(Args... rest)
		{
			return lrInterface<T>(acc.accessor,ptr_t(expand(0,rest...)));
		}

		template<class... Args> __host__ __device__
		const lrInterface<T> operator()(Args... rest)
		const
		{
			return lrInterface<T>(acc.accessor,ptr_t(expand(0,rest...)));
		}
		__host__ __device__
		int expand(int l,int i)
		{
			return i;
		}

		template<class... Args> __host__ __device__
		int expand(int l,int i, Args... rest)
		{
			assert(l < MAX_LEGION_MATRIX_DIMS);
			return i+dims[l]*expand(l+1,rest...);
		}

	};

	template<>__inline__
	lrAccessor<float>::lrAccessor(const PhysicalRegion &region, const int FID)
	{
		assert(region.get_field_accessor(FID).typeify<float>().can_convert<AccessorType::SOA<0>>());
		accessor = region.get_field_accessor(FID).typeify<float>().convert<AccessorType::SOA<sizeof(float)>>();
		//first_elem = IndexIterator(region.get_logical_region()).next();//-region.get_logical_region().get_index_space().get_valid_mask().first_element;
		first_elem = ptr_t(0);
	}

	template<>__inline__
	lrAccessor<int>::lrAccessor(const PhysicalRegion &region, const int FID)
	{
		assert(region.get_field_accessor(FID).typeify<int>().can_convert<AccessorType::SOA<0>>());

		accessor = region.get_field_accessor(FID).typeify<int>().convert<AccessorType::SOA<sizeof(int)>>();

		first_elem = ptr_t(0);//IndexIterator(region.get_logical_region()).next();//-region.get_logical_region().get_index_space().get_valid_mask().first_element;
	}

	template<>__inline__
	lrAccessor<double>::lrAccessor(const PhysicalRegion &region, const int FID)
	{
		assert(region.get_field_accessor(FID).typeify<double>().can_convert<AccessorType::SOA<0>>());

		accessor = region.get_field_accessor(FID).typeify<double>().convert<AccessorType::SOA<sizeof(double)>>();
		//first_elem = IndexIterator(region.get_logical_region()).next();//-region.get_logical_region().get_index_space().get_valid_mask().first_element;
		first_elem = ptr_t(0);
	}

	template<>__inline__
	lrAccessor<char>::lrAccessor(const PhysicalRegion &region, const int FID)
	{
		assert(region.get_field_accessor(FID).typeify<char>().can_convert<AccessorType::SOA<0>>());

		accessor = region.get_field_accessor(FID).typeify<char>().convert<AccessorType::SOA<sizeof(char)>>();
		//first_elem = IndexIterator(region.get_logical_region()).next();//-region.get_logical_region().get_index_space().get_valid_mask().first_element;
		first_elem = ptr_t(0);
	}

	template<>__inline__
	lrAccessor<LRWrapper>::lrAccessor(const PhysicalRegion &region, const int FID)
	{
		assert(region.get_field_accessor(FID).typeify<LRWrapper>().can_convert<AccessorType::SOA<0>>());

		accessor = region.get_field_accessor(FID).typeify<LRWrapper>().convert<AccessorType::SOA<sizeof(LRWrapper)>>();
		//first_elem = IndexIterator(region.get_logical_region()).next();//-region.get_logical_region().get_index_space().get_valid_mask().first_element;
		first_elem = ptr_t(0);
	}

	template<>__inline__
	lrAccessor<bool>::lrAccessor(const PhysicalRegion &region, const int FID)
	{
		assert(region.get_field_accessor(FID).typeify<bool>().can_convert<AccessorType::SOA<0>>());
		accessor = region.get_field_accessor(FID).typeify<bool>().convert<AccessorType::SOA<sizeof(bool)>>();
		//first_elem = IndexIterator(region.get_logical_region()).next();//-region.get_logical_region().get_index_space().get_valid_mask().first_element;
		first_elem = ptr_t(0);
	}

	typedef lrAccessor<float> float_lr;
	typedef lrAccessor<int> int_lr;
	typedef lrAccessor<double> double_lr;
	typedef lrAccessor<char> char_lr;

} /* namespace Dragon */

#endif /* LEGION_MATRIX_H_ */
