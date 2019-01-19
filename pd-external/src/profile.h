/*************************************************************//**
*
*	@file	profile.cpp
*	@date	28/10/2018
*	@author Martin Fouilleul
*
*****************************************************************/

#ifndef _PROFILE_H_
#define _PROFILE_H_

#include<inttypes.h>
#include<sys/time.h>
#include<stdio.h>

typedef int8_t	int8;
typedef int16_t	int16;
typedef int32_t	int32;
typedef int64_t	int64;

typedef uint8_t	 uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

typedef float	float32;
typedef double	float64;

//-----------------------------------------------------------------------
// Precise CPU Timing - unfortunately x86_64 specific...
//-----------------------------------------------------------------------

#ifdef PROFILE
#ifdef __x86_64__

uint64 TimeStamp()
{
	uint32 high, low;

	asm volatile("rdtsc\n"
		     :"=a"(low), "=d"(high));

	return(((uint64)high)<<32 | (uint64)low);
}

float64 GetTime()
{
	struct timeval tp;
	gettimeofday(&tp, 0);
	return(tp.tv_sec + tp.tv_usec*1.e-6);
}

void Serialize()
{
	asm volatile("xorl %%eax,%%eax\n"
		     "cpuid"
		     ::: "%eax", "%ebx", "%ecx", "%edx");
}


#else // __x86_64__

#warning "Profiling not supported for your platform - timing values will be erroneous"
inline void Serialize(){}
inline uint64 TimeStamp(){return(0);}

#endif	// __x86_64__

#define PROFILE_POST(s, ...) printf(s, ##__VA_ARGS__)

#define CYCLES_BLOCK_START()	\
{				\
	Serialize();		\
	uint64 __time_stamp_start__ = TimeStamp(); \

#define CYCLES_BLOCK_END(name)	\
				\
	Serialize();		\
	PROFILE_POST("Timed block %s : %llu\n", name, TimeStamp() - __time_stamp_start__); \
}				\

#define TIME_BLOCK_START()	\
{				\
	Serialize();		\
	float64 __time_start__ = GetTime(); \

#define TIME_BLOCK_END(name)	\
				\
	Serialize();		\
	PROFILE_POST("Timed block %s : %f\n", name, GetTime() - __time_start__); \
}				\


#else	// PROFILE

#define TIME_BLOCK_START()
#define TIME_BLOCK_END(var)
#define PROFILE_POST(s, ...)

#endif // PROFILE



#endif // _PROFILE_H_
