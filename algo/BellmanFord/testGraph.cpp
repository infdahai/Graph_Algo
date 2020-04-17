#include <type_traits>
// 设置当opencl出错时抛出异常
#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
// gcc 下如果定义了__STRICT_ANSI__，就没办法使用别名访问vector向量类型(cl_int2,cl_float8...)
#if defined( __GNUC__) && defined( __STRICT_ANSI__ )
#define __STRICT_ANSI__DEFINED__
// 删除__STRICT_ANSI__定义
#undef __STRICT_ANSI__
#endif
#include <CL/cl.hpp>
#ifdef __STRICT_ANSI__DEFINED__
#undef __STRICT_ANSI__DEFINED__
// 恢复__STRICT_ANSI__定义
#define __STRICT_ANSI__
#endif
namespace cl{
/* 根据向量元素类型和向量长度返回opencl向量类型，
 * 如 cl_vector_type<2,cl_int>::type 为 cl_int2
 */
template<int SIZE,typename T>
struct cl_vector_type{
    template<int _SIZE,typename _T>static typename std::enable_if< 2== _SIZE&&std::is_same<_T,cl_char>::value, cl_char2>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if< 4== _SIZE&&std::is_same<_T,cl_char>::value, cl_char4>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if< 8== _SIZE&&std::is_same<_T,cl_char>::value, cl_char8>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if<16== _SIZE&&std::is_same<_T,cl_char>::value,cl_char16>::type vector_type(_T);

    template<int _SIZE,typename _T>static typename std::enable_if< 2== _SIZE&&std::is_same<_T,cl_uchar>::value, cl_uchar2>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if< 4== _SIZE&&std::is_same<_T,cl_uchar>::value, cl_uchar4>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if< 8== _SIZE&&std::is_same<_T,cl_uchar>::value, cl_uchar8>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if<16== _SIZE&&std::is_same<_T,cl_uchar>::value,cl_uchar16>::type vector_type(_T);

    template<int _SIZE,typename _T>static typename std::enable_if< 2== _SIZE&&std::is_same<_T,cl_short>::value, cl_short2>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if< 4== _SIZE&&std::is_same<_T,cl_short>::value, cl_short4>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if< 8== _SIZE&&std::is_same<_T,cl_short>::value, cl_short8>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if<16== _SIZE&&std::is_same<_T,cl_short>::value,cl_short16>::type vector_type(_T);

    template<int _SIZE,typename _T>static typename std::enable_if< 2== _SIZE&&std::is_same<_T,cl_ushort>::value, cl_ushort2>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if< 4== _SIZE&&std::is_same<_T,cl_ushort>::value, cl_ushort4>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if< 8== _SIZE&&std::is_same<_T,cl_ushort>::value, cl_ushort8>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if<16== _SIZE&&std::is_same<_T,cl_ushort>::value,cl_ushort16>::type vector_type(_T);

    template<int _SIZE,typename _T>static typename std::enable_if< 2== _SIZE&&std::is_same<_T,cl_int>::value, cl_int2>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if< 4== _SIZE&&std::is_same<_T,cl_int>::value, cl_int4>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if< 8== _SIZE&&std::is_same<_T,cl_int>::value, cl_int8>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if<16== _SIZE&&std::is_same<_T,cl_int>::value,cl_int16>::type vector_type(_T);

    template<int _SIZE,typename _T>static typename std::enable_if< 2== _SIZE&&std::is_same<_T,cl_uint>::value, cl_uint2>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if< 4== _SIZE&&std::is_same<_T,cl_uint>::value, cl_uint4>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if< 8== _SIZE&&std::is_same<_T,cl_uint>::value, cl_uint8>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if<16== _SIZE&&std::is_same<_T,cl_uint>::value,cl_uint16>::type vector_type(_T);

    template<int _SIZE,typename _T>static typename std::enable_if< 2== _SIZE&&std::is_same<_T,cl_long>::value, cl_long2>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if< 4== _SIZE&&std::is_same<_T,cl_long>::value, cl_long4>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if< 8== _SIZE&&std::is_same<_T,cl_long>::value, cl_long8>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if<16== _SIZE&&std::is_same<_T,cl_long>::value,cl_long16>::type vector_type(_T);

    template<int _SIZE,typename _T>static typename std::enable_if< 2== _SIZE&&std::is_same<_T,cl_ulong>::value, cl_ulong2>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if< 4== _SIZE&&std::is_same<_T,cl_ulong>::value, cl_ulong4>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if< 8== _SIZE&&std::is_same<_T,cl_ulong>::value, cl_ulong8>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if<16== _SIZE&&std::is_same<_T,cl_ulong>::value,cl_ulong16>::type vector_type(_T);

    template<int _SIZE,typename _T>static typename std::enable_if< 2== _SIZE&&std::is_same<_T,cl_float>::value, cl_float2>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if< 4== _SIZE&&std::is_same<_T,cl_float>::value, cl_float4>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if< 8== _SIZE&&std::is_same<_T,cl_float>::value, cl_float8>::type vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if<16== _SIZE&&std::is_same<_T,cl_float>::value,cl_float16>::type vector_type(_T);

    template<int _SIZE,typename _T>static typename std::enable_if< 2== _SIZE&&std::is_same<_T,cl_double>::value, cl_double2>::type  vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if< 4== _SIZE&&std::is_same<_T,cl_double>::value, cl_double4>::type  vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if< 8== _SIZE&&std::is_same<_T,cl_double>::value, cl_double8>::type  vector_type(_T);
    template<int _SIZE,typename _T>static typename std::enable_if<16== _SIZE&&std::is_same<_T,cl_double>::value,cl_double16>::type  vector_type(_T);

    template<int _SIZE,typename _T>static typename std::enable_if<2!= _SIZE && 4 != _SIZE && 8 != _SIZE &&16 != _SIZE>::type vector_type(...);
    using type=decltype(vector_type<SIZE>(std::declval<T>()));
};
/*
 * 根据opencl 向量类型返回向量的元素类型和向量长度，
 * 如is_cl_vector<cl_int2>::type 为 cl_int
 *  is_cl_vector<cl_int2>::value 为true 是opencl向量类型
 *  is_cl_vector<cl_int2>::size 为向量长度
 */
template<typename T>
struct is_cl_vector{
    template<typename _T, typename ENABLE=void>
    struct vector_size{
        enum{size=-1};
    };
    template<typename _T>
    struct vector_size<_T, typename std::enable_if<
        std::is_same<_T, cl_char2>::value
        || std::is_same<_T, cl_uchar2>::value
        || std::is_same<_T, cl_short2>::value
        || std::is_same<_T, cl_ushort2>::value
        || std::is_same<_T, cl_int2>::value
        || std::is_same<_T, cl_uint2>::value
        || std::is_same<_T, cl_long2>::value
        || std::is_same<_T, cl_ulong2>::value
        || std::is_same<_T, cl_float2>::value
        || std::is_same<_T, cl_double2>::value
    >::type>{
        enum{size=2};
    };
    template<typename _T>
    struct vector_size<_T, typename std::enable_if<
        std::is_same<_T, cl_char4>::value
        || std::is_same<_T, cl_uchar4>::value
        || std::is_same<_T, cl_short4>::value
        || std::is_same<_T, cl_ushort4>::value
        || std::is_same<_T, cl_int4>::value
        || std::is_same<_T, cl_uint4>::value
        || std::is_same<_T, cl_long4>::value
        || std::is_same<_T, cl_ulong4>::value
        || std::is_same<_T, cl_float4>::value
        || std::is_same<_T, cl_double4>::value
    >::type>{
        enum{size=4};
    };
    template<typename _T>
    struct vector_size<_T, typename std::enable_if<
        std::is_same<_T, cl_char8>::value
        || std::is_same<_T, cl_uchar8>::value
        || std::is_same<_T, cl_short8>::value
        || std::is_same<_T, cl_ushort8>::value
        || std::is_same<_T, cl_int8>::value
        || std::is_same<_T, cl_uint8>::value
        || std::is_same<_T, cl_long8>::value
        || std::is_same<_T, cl_ulong8>::value
        || std::is_same<_T, cl_float8>::value
        || std::is_same<_T, cl_double8>::value
    >::type>{
        enum{size=8};
    };
    template<typename _T>
    struct vector_size<_T, typename std::enable_if<
        std::is_same<_T, cl_char16>::value
        || std::is_same<_T, cl_uchar16>::value
        || std::is_same<_T, cl_short16>::value
        || std::is_same<_T, cl_ushort16>::value
        || std::is_same<_T, cl_int16>::value
        || std::is_same<_T, cl_uint16>::value
        || std::is_same<_T, cl_long16>::value
        || std::is_same<_T, cl_ulong16>::value
        || std::is_same<_T, cl_float16>::value
        || std::is_same<_T, cl_double16>::value
    >::type>{
        enum{size=16};
    };

    static cl_char check(cl_char2);
    static cl_char check(cl_char4);
    static cl_char check(cl_char8);
    static cl_char check(cl_char16);

    static cl_uchar check(cl_uchar2);
    static cl_uchar check(cl_uchar4);
    static cl_uchar check(cl_uchar8);
    static cl_uchar check(cl_uchar16);

    static cl_short check(cl_short2);
    static cl_short check(cl_short4);
    static cl_short check(cl_short8);
    static cl_short check(cl_short16);

    static cl_ushort check(cl_ushort2);
    static cl_ushort check(cl_ushort4);
    static cl_ushort check(cl_ushort8);
    static cl_ushort check(cl_ushort16);

    static cl_int check(cl_int2);
    static cl_int check(cl_int4);
    static cl_int check(cl_int8);
    static cl_int check(cl_int16);

    static cl_uint check(cl_uint2);
    static cl_uint check(cl_uint4);
    static cl_uint check(cl_uint8);
    static cl_uint check(cl_uint16);

    static cl_long check(cl_long2);
    static cl_long check(cl_long4);
    static cl_long check(cl_long8);
    static cl_long check(cl_long16);

    static cl_ulong check(cl_ulong2);
    static cl_ulong check(cl_ulong4);
    static cl_ulong check(cl_ulong8);
    static cl_ulong check(cl_ulong16);

    static cl_float check(cl_float2);
    static cl_float check(cl_float4);
    static cl_float check(cl_float8);
    static cl_float check(cl_float16);

    static cl_double check(cl_double2);
    static cl_double check(cl_double4);
    static cl_double check(cl_double8);
    static cl_double check(cl_double16);
    static void check(...);
    using type=decltype(check(std::declval<T>()));
    enum{value=!std::is_void<type>::value,size=vector_size<T>::size};
};

template<typename T,typename C=float
        ,typename RET=typename std::enable_if<std::is_arithmetic<T>::value,typename std::conditional<std::is_floating_point<T>::value,T,C>::type>::type>
inline
RET
length(const T &x,const T &y){
    return RET(std::sqrt(std::pow(RET(x),2)+std::pow(RET(y),2)));
}
/*
 * 递归计算向量所有元素的平方和（递归结束）
 * */
template<typename T,typename C=float,typename VI=is_cl_vector<T>
        ,typename RET=typename std::conditional<std::is_floating_point<typename VI::type>::value,T,C>::type>
inline
typename std::enable_if<2==VI::size&&VI::value,RET>::type
square_sum(const T &pos){
    return RET(std::pow(RET(pos.s[0]),2)+std::pow(RET(pos.s[1]),2));
}
/*
 * 递归计算向量所有元素的平方和
 * */
template<typename T,typename C=float,typename VI=is_cl_vector<T>
    ,typename RET=typename std::conditional<std::is_floating_point<typename VI::type>::value,T,C>::type>
inline
typename std::enable_if<2<VI::size&&VI::value,RET>::type
square_sum(const T &pos){
    return square_sum(pos.hi)+square_sum(pos.lo);
}
template<typename T,typename C=float,typename VI=is_cl_vector<T>
        ,typename RET=typename std::enable_if<VI::value,typename std::conditional<std::is_floating_point<typename VI::type>::value,T,C>::type>::type>
inline
RET
length(const T &pos){
    return RET(std::sqrt(square_sum(pos)));
}

template<typename T,typename C=float,typename VI = cl::is_cl_vector<T>
        ,typename RET=typename std::enable_if<VI::value,typename std::conditional<std::is_floating_point<typename VI::type>::value,T,C>::type>::type>
inline
RET
distance(const T &p0,const T &p1){
    return length(p0-p1);
}
}/* namespace cl */

////////////////begin global namespace//////////////
/*
 * (递归结束)向量减法操作符
 */
template<typename T, typename VI = cl::is_cl_vector<T>
    , typename ENABLE = typename std::enable_if<VI::value>::type
    , typename RET= typename cl::cl_vector_type<VI::size, decltype(declval<T>().s[0] - declval<T>().s[0])>::type>
inline
typename std::enable_if<2 == VI::size, RET>::type
operator-(const T &p1, const T &p2) {
    return{ p1.s[0] - p2.s[0],p1.s[1] - p2.s[1] };
}
/*
 * (递归)向量减法操作符
 */
template<typename T, typename VI = cl::is_cl_vector<T>
    , typename ENABLE = typename std::enable_if<VI::value>::type
    , typename RET = typename cl::cl_vector_type<VI::size, decltype(declval<T>().s[0] - declval<T>().s[0])>::type>
inline
typename std::enable_if<2<VI::size, RET>::type
operator-(const T &p1, const T &p2) {
    RET r;
    r.hi = p1.hi - p2.hi;
    r.lo = p1.lo - p2.lo;
    return r;
}
/*
 * (递归结束)向量减法操作符,第二个操作数非向量
 */
template<typename T, typename VI = cl::is_cl_vector<T>
    , typename ENABLE = typename std::enable_if<VI::value>::type
    , typename RET= typename cl::cl_vector_type<VI::size, decltype(declval<T>().s[0] - declval<T>().s[0])>::type>
inline
typename std::enable_if<2 == VI::size, RET>::type
operator-(const T &p1, const typename VI::type &p2) {
    return{ p1.s[0] - p2,p1.s[1] - p2};
}
/*
 * (递归)向量减法操作符,第二个操作数非向量
 */
template<typename T, typename VI = cl::is_cl_vector<T>
    , typename ENABLE = typename std::enable_if<VI::value>::type
    , typename RET = typename cl::cl_vector_type<VI::size, decltype(declval<T>().s[0] - declval<T>().s[0])>::type>
inline
typename std::enable_if<2<VI::size, RET>::type
operator-(const T &p1, const typename VI::type &p2) {
    RET r;
    r.hi = p1.hi - p2;
    r.lo = p1.lo - p2;
    return r;
}
/*
* (递归结束)向量减法操作符,第一个操作数非向量
*/
template<typename N, typename T, typename VI = cl::is_cl_vector<T>
    , typename ENABLE = typename std::enable_if<VI::value&&std::is_same<N, typename VI::type>::value>::type
    , typename RET = typename cl::cl_vector_type<VI::size, decltype(declval<T>().s[0] - declval<T>().s[0])>::type>
    inline
    typename std::enable_if<2==VI::size, RET>::type
    operator-(const N &p1, const  T&p2) {
    return{ p1 - p2.s[0],p1 - p2.s[1] };
}

/*
 * (递归)向量减法操作符,第一个操作数非向量
 */
template<typename N,typename T, typename VI = cl::is_cl_vector<T>
    , typename ENABLE = typename std::enable_if<VI::value&&std::is_same<N, typename VI::type>::value>::type
    , typename RET = typename cl::cl_vector_type<VI::size, decltype(declval<T>().s[0] - declval<T>().s[0])>::type>
inline
typename std::enable_if<2<VI::size, RET>::type
operator-(const N &p1, const  T&p2) {
    RET r;
    r.hi = p1 - p2.hi;
    r.lo = p1 - p2.hi;
    return r;
}
/*
 * (递归结束)向量加法操作符
 */
template<typename T, typename VI = cl::is_cl_vector<T>
    , typename ENABLE = typename std::enable_if<VI::value>::type
    , typename RET= typename cl::cl_vector_type<VI::size, decltype(declval<T>().s[0] + declval<T>().s[0])>::type>
inline
typename std::enable_if<2 == VI::size, RET>::type
operator+(const T &p1, const T &p2) {
    return{ p1.s[0] + p2.s[0],p1.s[1] + p2.s[1] };
}
/*
 * (递归)向量加法操作符
 */
template<typename T, typename VI = cl::is_cl_vector<T>
    , typename ENABLE = typename std::enable_if<VI::value>::type
    , typename RET = typename cl::cl_vector_type<VI::size, decltype(declval<T>().s[0] + declval<T>().s[0])>::type>
inline
typename std::enable_if<2<VI::size, RET>::type
operator+(const T &p1, const T &p2) {
    RET r;
    r.hi = p1.hi + p2.hi;
    r.lo = p1.lo + p2.lo;
    return r;
}
/*
 * (递归结束)向量加法操作符,第二个操作数非向量
 */
template<typename T, typename VI = cl::is_cl_vector<T>
    , typename ENABLE = typename std::enable_if<VI::value>::type
    ,typename RET=typename cl::cl_vector_type<VI::size, decltype(declval<T>().s[0] + declval<T>().s[0])>::type>
inline
typename std::enable_if<2 == VI::size, RET>::type
operator+(const T &p1, const typename VI::type &p2) {
    return{ p1.s[0] + p2,p1.s[1] + p2 };
}
/*
 * (递归)向量加法操作符,第二个操作数非向量
 */
template<typename T, typename VI = cl::is_cl_vector<T>
    , typename ENABLE = typename std::enable_if<VI::value>::type
    , typename RET = typename cl::cl_vector_type<VI::size, decltype(declval<T>().s[0] + declval<T>().s[0])>::type>
inline
typename std::enable_if<2<VI::size, RET>::type
operator+(const T &p1, const typename VI::type &p2) {
    RET r;
    r.hi = p1.hi + p2;
    r.lo = p1.lo + p2;
    return r;
}

/*
 * 向量加法操作符,第一个操作数非向量
 */
template<typename N,typename T, typename VI = cl::is_cl_vector<T>
    , typename ENABLE = typename std::enable_if<VI::value&&std::is_same<N, typename VI::type>::value>::type
    , typename RET = typename cl::cl_vector_type<VI::size, decltype(declval<T>().s[0] + declval<T>().s[0])>::type>
inline
RET
operator+(const N &p1, const  T&p2) {
    return p2+p1;
}


////////////////end global namespace//////////////
// ————————————————
// 版权声明：本文为CSDN博主「10km」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
// 原文链接：https://blog.csdn.net/10km/java/article/details/51121642