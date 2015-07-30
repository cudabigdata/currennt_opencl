/******************************************************************************
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
 *
 * This file is part of CURRENNT.
 *
 * CURRENNT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CURRENNT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CURRENNT.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#ifndef TYPESCL_HPP
#define TYPESCL_HPP

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <iostream>

#define PATTYPE_NONE   0 ///< pattern does not belong to the sequence
#define PATTYPE_FIRST  1 ///< first pattern/timestep in the sequence
#define PATTYPE_NORMAL 2 ///< pattern/timestep with a sequence (not first/last)
#define PATTYPE_LAST   3 ///< last pattern/timestep in the sequence


/*************************************************************************//**
 * The floating point type used for all computations
 *****************************************************************************/
typedef float real_t;

typedef struct d_mem_real
{
	cl_mem mem;
	unsigned int size;

	unsigned int numBytes()
	{
		return size * sizeof(real_t);
	}
	d_mem_real()
	{
		size = 0;
		mem = 0;
	}


	void free(){

		if (mem !=0 && size > 0){
			cl_int error = clReleaseMemObject(mem);
		}
		mem = 0;
		size = 0;
	}
	~d_mem_real(){
		mem = 0;
		size = 0;

	}


} d_mem_real;

typedef struct d_mem_int
{
	cl_mem mem;
	unsigned int size;


	unsigned int numBytes()
	{
		return size * sizeof(int);
	}
	d_mem_int()
	{
		size = 0;
		mem = 0;
	}



	void free(){
		if (mem !=0)
			clReleaseMemObject(mem);
		mem = 0;
		size = 0;
	}
	~d_mem_int(){
		mem = 0;
		size = 0;
	}

} d_mem_int;

typedef struct d_mem_char
{
	cl_mem mem;
	unsigned int size;

	unsigned int numBytes()
	{
		return size * sizeof(char);
	}
	d_mem_char()
	{
		size = 0;
		mem = 0;
	}

	void free(){
		if (mem !=0)
			clReleaseMemObject(mem);
		mem = 0;
		size = 0;
	}
	~d_mem_char(){
		mem = 0;
		size = 0;
	}

} d_mem_char;


#endif // TYPESCL_HPP
