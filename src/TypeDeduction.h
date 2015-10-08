/*
 * TypeDeduction.h
 *
 *  Created on: Jul 7, 2015
 *      Author: payne
 *
 * Copyright (c) 2014-2015 Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the Dragon project. See the LICENSE.txt file at the
 * top-level directory of this distribution.
 */

#ifndef TYPEDEDUCTION_H_
#define TYPEDEDUCTION_H_
#include <legion.h>
#include <typeinfo>
#include <type_traits>
#ifdef USE_CUDA_FKOP
#include <cuda.h>
#include <cuda_runtime.h>
#else
#define __host__
#define __device__
#endif
#include "legion_tasks.h"
#include <sstream>
#include <unistd.h>
#include "LRWrapper.h"



using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

namespace Dragon
{

	// ClassList contains a list of types
	template<class... cB>
	class ClassList
	{
	};

	template<class cA,class... cB>
	ClassList<cB...> reduceCList(ClassList<cA,cB...> a)
	{
		return ClassList<cB...>();
	}

	// Contains a class list as an object, possibly unnecessary
	template<typename... args>
	class typeList
	{
	public:
		ClassList<args...> aList;
	};


	// This class allows us to catalog a list of templated accessor types and the type they are templated on.
	// This is the basic version so that we can do specialization on other versions
	template<class... args>
	class Wrapper2
	{
		static const int NARGS = 0;
	};

	// Specialized top-level version for wrapping more than one argument
	template<template<class>class Accessor,class Arg1, class... args>
	class Wrapper2<Accessor<Arg1>,args...>
	{
	public:
		typedef typename Wrapper2<typeList<Accessor<Arg1>>,args...>::tList tList;
		static const int NARGS = 1+Wrapper2<typeList<Accessor<Arg1>>,args...>::NARGS;

	};

	// Specialized version for wrapping a single argument
	template<template<class>class Accessor,class Arg1>
	class Wrapper2<Accessor<Arg1>>
	{
	public:
		typedef typeList<Accessor<Arg1>> tList;
		static const int NARGS = 1;
	};

	// Specialized mid-level version for wrapping multiple arguments.
	template<template<class>class Accessor,class Arg1, class... tListTail,class... args>
	class Wrapper2<typeList<tListTail...>,Accessor<Arg1>,args...>
	{
	public:
		typedef typename Wrapper2<typeList<tListTail...,Accessor<Arg1>>,args...>::tList tList;
		static const int NARGS = 1+Wrapper2<typeList<tListTail...,Accessor<Arg1>>,args...>::NARGS;

	};

	// Specialized bottom-level version for wrapping multiple arguments.
	template<template<class>class Accessor,class Arg1, class... tListTail>
	class Wrapper2<typeList<tListTail...>,Accessor<Arg1>>
	{
	public:
		typedef typeList<tListTail...,Accessor<Arg1>> tList;
		static const int NARGS = 1;

	};


	// Default version of a class to extract the arguments and return value from a function pointer.
	template <class T>
	class ArgExctractor
	{
	public:
		typedef typename ArgExctractor<T>::tList tList;
	};

	// Specialized ArgExtractor for functions with only accessors as arguments
	template<class rVal,class... args>
	class ArgExctractor<rVal(*)(args...)>
	{
	public:
		typedef typename Wrapper2<args...>::tList tList;
		typedef std::false_type cuda_capable;
		static const int NARGS = Wrapper2<args...>::NARGS;
		static const bool CPU_BASE_LEAF = true;

	};

	// Specialized ArgExtractor for functions with an integer input and accessors as arguments
	template<class rVal,class... args>
	class ArgExctractor<rVal(*)(int,args...)>
	{
	public:
		typedef typename Wrapper2<args...>::tList tList;
		typedef std::true_type cuda_capable;
		static const int NARGS = Wrapper2<args...>::NARGS;
		static const bool CPU_BASE_LEAF = true;

	};

	// Specialized ArgExtractor for functions with full context and task information as arguments
	template<class rVal,class... args>
	class ArgExctractor<rVal(*)(const Task*,Context,HighLevelRuntime*,args...)>
	{
	public:
		typedef typename Wrapper2<args...>::tList tList;
		typedef std::false_type cuda_capable;
		static const int NARGS = Wrapper2<args...>::NARGS;
		static const bool CPU_BASE_LEAF = false;

	};


	template <class T>
	class RetExctractor
	{
	public:
		typedef void rT;

	};

	// Class for extracting the return type from a function pointer type
	template<class rVal,class... args>
	class RetExctractor<rVal(*)(args...)>
	{
	public:
		typedef rVal rT;
	};



	// Class to check functor class for a "start" member function
	template< typename T>
	struct has_on_start_method
	{
	    /* SFINAE foo-has-correct-sig :) */
	    template<typename A>
	    static std::true_type test(void (A::*)()) {
	        return std::true_type();
	    }

	    /* SFINAE foo-exists :) */
	    template<typename A>
	    static decltype(test(&A::on_start))
	    test(decltype(&A::on_start),void*) {
	        /* foo exists. What about sig? */
	        typedef decltype(test(&A::foo)) return_type;
	        return return_type();
	    }

	    /* SFINAE game over :( */
	    template<typename A>
	    static std::false_type test(...) {
	        return std::false_type();
	    }

	    /* This will be either `std::true_type` or `std::false_type` */
	    typedef decltype(test<T>(0,0)) type;

	    static const bool value = type::value; /* Which is it? */

	    /*  `eval(T const &,std::true_type)`
	        delegates to `T::foo()` when `type` == `std::true_type`
	    */
	    static void eval(T  & t, std::true_type) {
	        t.on_start();
	    }
	    /* `eval(...)` is a no-op for otherwise unmatched arguments */
	    template<class... Args2>
	    static void eval(Args2... ar){

	    }

	    /* `eval(T const & t)` delegates to :-
	        - `eval(t,type()` when `type` == `std::true_type`
	        - `eval(...)` otherwise
	    */
	    static void eval(T  & t) {
	        eval(t,type());
	    }
	};

	// Class to check functor class for a "finish" member function
	template< typename T>
	struct has_on_finish_method
	{
	    /* SFINAE foo-has-correct-sig :) */
	    template<typename A>
	    static std::true_type test(void (A::*)()) {
	        return std::true_type();
	    }

	    /* SFINAE foo-exists :) */
	    template<typename A>
	    static decltype(test(&A::on_finish))
	    test(decltype(&A::on_finish),void*) {
	        /* foo exists. What about sig? */
	        typedef decltype(test(&A::foo)) return_type;
	        return return_type();
	    }

	    /* SFINAE game over :( */
	    template<typename A>
	    static std::false_type test(...) {
	        return std::false_type();
	    }

	    /* This will be either `std::true_type` or `std::false_type` */
	    typedef decltype(test<T>(0,0)) type;

	    static const bool value = type::value; /* Which is it? */

	    /*  `eval(T const &,std::true_type)`
	        delegates to `T::foo()` when `type` == `std::true_type`
	    */
	    static void eval(T  & t, std::true_type) {
	        t.on_finish();
	    }
	    /* `eval(...)` is a no-op for otherwise unmatched arguments */
	    template<class... Args2>
	    static void eval(Args2... ar){

	    }

	    /* `eval(T const & t)` delegates to :-
	        - `eval(t,type()` when `type` == `std::true_type`
	        - `eval(...)` otherwise
	    */
	    static void eval(T  & t) {
	        eval(t,type());
	    }
	};


	// Class to check for an "evaluate" member function matching some required sig
	template< typename T,class B>
	struct find_method
	{

	};

	template< typename T,class rVal,class ...Args>
	struct find_method<T,rVal(*)(Args...)>
	{
	    /* SFINAE foo-has-correct-sig :) */
	    template<typename A>
	    static std::true_type test(rVal (A::*)(Args...)) {
	        return std::true_type();
	    }

	    /* SFINAE foo-exists :) */
	    template<typename A>
	    static decltype(test(&A::evaluate))
	    test(decltype(&A::evaluate),void*) {
	        /* foo exists. What about sig? */
	        typedef decltype(test(&A::foo)) return_type;
	        return return_type();
	    }

	    /* SFINAE game over :( */
	    template<typename A>
	    static std::false_type test(...) {
	        return std::false_type();
	    }

	    /* This will be either `std::true_type` or `std::false_type` */
	    typedef decltype(test<T>(0,0)) type;

	    static const bool value = type::value; /* Which is it? */

	    /*  `eval(T const &,std::true_type)`
	        delegates to `T::foo()` when `type` == `std::true_type`
	    */
	    static rVal eval(T  & t, std::true_type,Args... ar) {
	        return t.evaluate(ar...);
	    }
	    /* `eval(...)` is a no-op for otherwise unmatched arguments */
	    template<class... Args2>
	    static rVal eval(Args2... ar){
	        // This output for demo purposes. Delete
	        std::cout << "T::foo() not called" << std::endl;
	        return rVal();
	    }

	    /* `eval(T const & t)` delegates to :-
	        - `eval(t,type()` when `type` == `std::true_type`
	        - `eval(...)` otherwise
	    */
	    static rVal eval(T  & t,Args... ar) {
	        return eval(t,type(),ar...);
	    }
	};


//	// Class to check for a "gpu_impl" member function.
//	template< typename T,class B>
//	struct has_gpu_impl
//	{
//
//	};
//
//	template< typename T,class rVal,class ...Args>
//	struct has_gpu_impl<T,rVal(*)(Args...)>
//	{
//	    /* SFINAE foo-has-correct-sig :) */
//	    template<typename A>
//	    static std::true_type test(rVal (A::*)(Args...)) {
//	        return std::true_type();
//	    }
//
//	    /* SFINAE foo-exists :) */
//	    template<typename A>
//	    static decltype(test(&A::evaluate))
//	    test(decltype(&A::evaluate),void*) {
//	        /* foo exists. What about sig? */
//	        typedef decltype(test(&A::foo)) return_type;
//	        return return_type();
//	    }
//
//	    /* SFINAE game over :( */
//	    template<typename A>
//	    static std::false_type test(...) {
//	        return std::false_type();
//	    }
//
//	    /* This will be either `std::true_type` or `std::false_type` */
//	    typedef decltype(test<T>(0,0)) type;
//
//	    static const bool value = type::value; /* Which is it? */
//
//	    /*  `eval(T const &,std::true_type)`
//	        delegates to `T::foo()` when `type` == `std::true_type`
//	    */
//	    static rVal eval(T  & t, std::true_type,Args... ar) {
//	        return t.evaluate(ar...);
//	    }
//	    /* `eval(...)` is a no-op for otherwise unmatched arguments */
//	    template<class... Args2>
//	    static rVal eval(Args2... ar){
//	        // This output for demo purposes. Delete
//	        std::cout << "T::foo() not called" << std::endl;
//	        return rVal();
//	    }
//
//	    /* `eval(T const & t)` delegates to :-
//	        - `eval(t,type()` when `type` == `std::true_type`
//	        - `eval(...)` otherwise
//	    */
//	    static rVal eval(T  & t,Args... ar) {
//	        return eval(t,type(),ar...);
//	    }
//	};

	// Dummy function for generating specific function sigs.
	template<class rT,class ...Args>
	rT DummyFoo(Args... args){return rT();};


	struct ArgMapInfo
	{
		uint8_t req,fid,wrap;
	};

	template<class O>
	struct DragonOpInfo
	{
	public:
		DragonOpInfo(O _op) : op(_op) {}
		O op;
		typedef typename ArgExctractor<decltype(&(O::evaluate_s))>::tList tList;
		typedef typename RetExctractor<decltype(&(O::evaluate_s))>::rT rT;
		static const int NARGS = ArgExctractor<decltype(&(O::evaluate_s))>::NARGS + !std::is_void<rT>::value;

		FieldID w_fid;

		unsigned narg;
		unsigned nwrap;


		ArgMapInfo arg_req_map[NARGS];

		LRWrapper wrappers[NARGS];
	};




	class FieldPrivlages
	{
	public:
		FieldPrivlages(FieldID _fid,PrivilegeMode _priv) :fid(_fid),priv(_priv){}
		FieldID fid;
		PrivilegeMode priv;
	};

	__inline__
	uint16_t GenCheckSum16(char* addr, unsigned count)
	{
		register uint16_t sum = 0;

		while(count > 1)
		{
			sum +=  sum ^ (*((uint16_t*)addr));
			addr = (char*)(((uint16_t*)addr)+1);
			count -= 2;
		}

		if(count > 0)
			sum += sum ^ (*((uint8_t*)addr));

		  // Fold 32-bit sum to 16 bits
		  while (sum>>16)
		    sum = (sum & 0xFFFF) + (sum >> 16);

		  return(~sum);
	}

	__inline__
	uint32_t GenCheckSum32(char* addr, unsigned count)
	{
		register uint64_t sum = 0;

		while(count > 3)
		{
			sum +=  sum ^ (*((uint32_t*)addr));
			addr = (char*)(((uint32_t*)addr)+1);
			count -= 4;
		}

		if(count > 0)
			sum += sum ^ GenCheckSum16(addr,count);

		  // Fold 32-bit sum to 16 bits
		  while (sum>>32)
		    sum = (sum & 0xFFFFFFFF) + (sum >> 32);

		  return(~sum);
	}

	__inline__
	uint64_t GenCheckSum64(char* addr, unsigned count)
	{
		register uint64_t sum = 0;

		while(count > 7)
		{
			sum +=  sum ^ (*((size_t*)addr));
			addr = (char*)(((size_t*)addr)+1);
			count -= 8;
		}

		if(count > 0)
			sum += sum ^ GenCheckSum32(addr,count);

		return sum;
	}

	template<class T>__inline__
	uint64_t GenCheckSumArray(T* addr, unsigned count)
	{
		if(sizeof(T) > 4)
			return GenCheckSum64((char*)addr,count*sizeof(T));
	}



};


#endif /* TYPEDEDUCTION_H_ */
