#include <iostream>
#include <typeinfo>
#include <sstream>
#include <unistd.h>
#include <stdio.h>
#include <string.h>


/*! The template `has_void_foo_no_args_const<T>` exports a
    boolean constant `value` that is true iff `T` provides
    `void foo() const`

    It also provides `static void eval(T const & t)`, which
    invokes void `T::foo() const` upon `t` if such a public member
    function exists and is a no-op if there is no such member.
*/
template< typename T,typename B>
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
    static decltype(test(&A::foo))
    test(decltype(&A::foo),void*) {
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
    rVal eval(T  & t, std::true_type,Args... ar) {
        return t.foo(ar...);
    }
    /* `eval(...)` is a no-op for otherwise unmatched arguments */
    rVal eval(...){
        // This output for demo purposes. Delete
        std::cout << "T::foo() not called" << std::endl;
    }

    /* `eval(T const & t)` delegates to :-
        - `eval(t,type()` when `type` == `std::true_type`
        - `eval(...)` otherwise
    */
    rVal eval(T  & t,Args... ar) {
        return eval(t,type(),ar...);
    }
};

//template< typename T,class rVal>
//struct find_method<T,rVal(*)()>
//{
//    /* SFINAE foo-has-correct-sig :) */
//    template<typename A>
//    static std::true_type test(rVal (A::*)()) {
//        return std::true_type();
//    }
//
//    /* SFINAE foo-exists :) */
//    template<typename A>
//    static decltype(test(&A::foo))
//    test(decltype(&A::foo),void*) {
//        /* foo exists. What about sig? */
//        typedef decltype(test(&A::foo)) return_type;
//        return return_type();
//    }
//
//    /* SFINAE game over :( */
//    template<typename A>
//    static std::false_type test(...) {
//        return std::false_type();
//    }
//
//    /* This will be either `std::true_type` or `std::false_type` */
//    typedef decltype(test<T>(0,0)) type;
//
//    static const bool value = type::value; /* Which is it? */
//
//    /*  `eval(T const &,std::true_type)`
//        delegates to `T::foo()` when `type` == `std::true_type`
//    */
//    rVal eval(T  & t, std::true_type) {
//        return t.foo();
//    }
//    /* `eval(...)` is a no-op for otherwise unmatched arguments */
//    rVal eval(...){
//        // This output for demo purposes. Delete
//        std::cout << "T::foo() not called" << std::endl;
//    }
//
//    /* `eval(T const & t)` delegates to :-
//        - `eval(t,type()` when `type` == `std::true_type`
//        - `eval(...)` otherwise
//    */
//    rVal eval(T  & t) {
//         return eval(t,type());
//    }
//};

//template<class T,class rVal,class... args>
//bool findMethod(T op,rVal(*)(args...))
//{
//	return has_void_foo_no_args_const<T,rVal,args...>::value;
//}


template<class... Args>
class PArgWrapper
{

};

template<class Arg>
class PArgWrapper<Arg>
{
public:
	Arg ar;
};


template<class Arg1,class... Args>
class PArgWrapper<Arg1,Args...>
{
public:
	Arg1 ar;
	PArgWrapper<Args...> tail;
};

int foobar()
{
	return 0;
}

// For testing
struct AA {
    void foo() const {
        std::cout << "AA::foo() called" << std::endl;
    }
};

// For testing
struct BB {
    void foo() {
        std::cout << "BB::foo() called" << std::endl;
    }
};

// For testing
struct CC {
    int foo() {
        std::cout << "CC::foo() called" << std::endl;
        return 0;
    }
};

// This is the desired implementation of `void f(T const& val)`
//template<class T>
//void f(T const& val) {
//    has_void_foo_no_args_const<T>::eval(val);
//}

template<class T,class rVal,class... args>
rVal f(T & val,rVal(*)(args...)) {
	find_method<T,rVal(*)(args...)>().eval(val);
}

int main() {
    AA aa;
    std::cout << (find_method<AA,decltype(&foobar)>::value ?
        "AA has void foo() const" : "AA does not have void foo() const")
        << std::endl;
    f(aa,&foobar);
    BB bb;
    std::cout << (find_method<BB,decltype(&foobar)>::value?
        "BB has void foo() const" : "BB does not have void foo() const")
        << std::endl;
    f(bb,&foobar);
    CC cc;
    std::cout << (find_method<CC,decltype(&foobar)>::value ?
        "CC has void foo() const" : "CC does not have void foo() const")
        << std::endl;
    f(cc,&foobar);

    double dd;
    std::cout << (find_method<double,decltype(&foobar)>::value ?
        "Double has void foo() const" : "Double does not have void foo() const")
        << std::endl;
    f(dd,&foobar);

     const char* dummy = "%1234:5678";
     const char* dummy2 = "1234:5678%9";

	char tmp[64];
	strcpy(tmp,dummy2);
	const char* s = strchr(dummy2,'%');
	tmp[s-dummy2] = '\0';

	std::cout << tmp << " s = " << atoi(s+1) << std::endl;

     std::cout << (strchr(dummy,'%') == dummy) << std::endl;

     std::cout << atoi(strchr(dummy,':')+1) << std::endl;
     return 0;


//    AA aa;
//    std::cout << (has_void_foo_no_args_const<AA>::value ?
//        "AA has void foo() const" : "AA does not have void foo() const")
//        << std::endl;
//    f(aa);
//    BB bb;
//    std::cout << (has_void_foo_no_args_const<BB>::value ?
//        "BB has void foo() const" : "BB does not have void foo() const")
//        << std::endl;
//    f(bb);
//    CC cc;
//    std::cout << (has_void_foo_no_args_const<CC>::value ?
//        "CC has void foo() const" : "CC does not have void foo() const")
//        << std::endl;
//    f(cc);
//    std::cout << (has_void_foo_no_args_const<double>::value ?
//        "Double has void foo() const" : "Double does not have void foo() const")
//        << std::endl;
//    f(3.14);
//            return 0;

}
