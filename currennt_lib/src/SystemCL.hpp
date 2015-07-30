/*
 * SystemCL.hpp
 *
 *  Created on: Jan 10, 2015
 *      Author: tamnguyen
 */

#ifndef CURRENNT_LIB_SRC_SYSTEMCL_HPP_
#define CURRENNT_LIB_SRC_SYSTEMCL_HPP_

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "TypesCL.hpp"
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <vector>


class SystemCL
{
private:

//	int platform;
//	int device;
	cl_platform_id platform_id;
	cl_device_id device_id;
	cl_uint num_devices;
	cl_uint num_platforms;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;

	cl_kernel memset_float_k;
    cl_kernel mm_tranpose_afn_k;
    cl_kernel mm_tranpose_bfn_k;
    cl_kernel mm_tranpose_fn_k;
    cl_kernel mma_tranpose_afn_k;
    cl_kernel mma_tranpose_bfn_k;
    cl_kernel mma_tranpose_fn_k;
    cl_kernel r_plus_k;
    cl_kernel weightedSSeFn_k;
    cl_kernel outputErrorFn_k;
    cl_kernel ffl_computeOutputFn_k;
    cl_kernel ffl_computeDeltaFn_k;
    cl_kernel ffl_computeBiasWeightUpdateFn_k;
    cl_kernel spol_computeSseFn_k;
    cl_kernel spol_computeOutputErrorFn_k;
    cl_kernel bcl_countCorrectClassificationsFn_k;
    cl_kernel bcl_computeCrossEntropyErrorFn_k;
    cl_kernel bcl_computeOutputErrorFn_k;
    cl_kernel cpol_computeCeFn_k;
    cl_kernel cpol_computeOutputErrorFn_k;
    cl_kernel mcl_countCorrectClassificationsFn_k;
    cl_kernel mcl_computeCrossEntropyErrorFn_k;
    cl_kernel mcl_computeOutputErrorFn_k;
    cl_kernel rpol_calculateError_k;
    cl_kernel rpol_computeRmseFn_k;
    cl_kernel rpol_computeOutputErrorFn_k;
    cl_kernel sml_calculateOffsetFn_k;
    cl_kernel sml_calculateExpFn_k;
    cl_kernel sml_sumUpOutputsFn_k;
    cl_kernel sml_normalizeOutputsFn_k;
    cl_kernel sml_calculateErrorOffsetFn_k;
    cl_kernel sml_calculateErrorsFn_k;
    cl_kernel smpol_computeSseMaskFn_k;
    cl_kernel smpol_computeOutputErrorFn_k;
    cl_kernel ll_computeBlockOutputFn_k;
    cl_kernel ll_resortOutputsFn_k;
    cl_kernel ll_resortOutputErrorsFn_k;
    cl_kernel ll_computeBlockErrorsFn_k;
    cl_kernel ll_computeWeightUpdateFn_k;
    cl_kernel sdo_updateWeightFn_k;
    ~SystemCL(){
        cl_int ret = clReleaseProgram(program);

        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);
    }
	SystemCL(){


	    platform_id = NULL;
	    device_id = NULL;

	    cl_int ret = clGetPlatformIDs(1, &platform_id, &num_platforms);
	    CheckError(ret);
	    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1,
	            &device_id, &num_devices);
	    CheckError(ret);
	    // Create an OpenCL context
	    context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
	    CheckError(ret);
	    // Create a command queue
	    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	    CheckError(ret);
	    cl_int errNum;


		std::ifstream kernelFile("currennt_kernel.cl", std::ios::in);
		if (!kernelFile.is_open())
		{
			std::cerr << "Failed to currennt_kernel.cl file for reading: " << std::endl;
			std::exit(0);
		}

		std::ostringstream oss;
		oss << kernelFile.rdbuf();

		std::string srcStdStr = oss.str();
		const char *srcStr = srcStdStr.c_str();
		program = clCreateProgramWithSource(context, 1,
											(const char**)&srcStr,
											NULL, &errNum);
		CheckError(errNum);
		if (program == NULL)
		{
			std::cerr << "Failed to create CL program from source." << std::endl;
			std::exit(0);
		}

		errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
		if (errNum != CL_SUCCESS)
		{
			// Determine the reason for the error
			char buildLog[16384];
			clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
								  sizeof(buildLog), buildLog, NULL);

			std::cerr << "Error in kernel: " << std::endl;
			std::cerr << buildLog;
			clReleaseProgram(program);
			std::exit(0);
		}
		// Load kernels
		memset_float_k = clCreateKernel(program, "memset_float", &ret);
		CheckError(ret);
		mm_tranpose_afn_k = clCreateKernel(program, "mm_tranpose_AFn", &ret);
		CheckError(ret);
		mm_tranpose_fn_k = clCreateKernel(program, "mm_tranpose_Fn", &ret);
		CheckError(ret);
		mm_tranpose_bfn_k = clCreateKernel(program, "mm_tranpose_BFn", &ret);
		CheckError(ret);
		mma_tranpose_afn_k = clCreateKernel(program, "mma_tranpose_AFn", &ret);
		CheckError(ret);
		mma_tranpose_fn_k = clCreateKernel(program, "mma_tranpose_Fn", &ret);
		CheckError(ret);
		mma_tranpose_bfn_k = clCreateKernel(program, "mma_tranpose_BFn", &ret);
		CheckError(ret);
		r_plus_k = clCreateKernel(program, "r_plus", &ret);
		CheckError(ret);
		weightedSSeFn_k =  clCreateKernel(program, "compute_weightedSSeFn", &ret);
		CheckError(ret);
		outputErrorFn_k =  clCreateKernel(program, "compute_OutputErrorFn", &ret);
		CheckError(ret);
		ffl_computeOutputFn_k =  clCreateKernel(program, "ffl_computeOutputFn", &ret);
		CheckError(ret);
		ffl_computeDeltaFn_k =  clCreateKernel(program, "ffl_computeDeltaFn", &ret);
		CheckError(ret);
		ffl_computeBiasWeightUpdateFn_k = clCreateKernel(program, "ffl_computeBiasWeightUpdateFn", &ret);
		CheckError(ret);
		spol_computeSseFn_k = clCreateKernel(program, "spol_computeSseFn", &ret);
		CheckError(ret);
		spol_computeOutputErrorFn_k = clCreateKernel(program, "spol_computeOutputErrorFn", &ret);
		CheckError(ret);
		bcl_countCorrectClassificationsFn_k = clCreateKernel(program, "bcl_countCorrectClassificationsFn", &ret);
		CheckError(ret);
		bcl_computeCrossEntropyErrorFn_k = clCreateKernel(program, "bcl_computeCrossEntropyErrorFn", &ret);
		CheckError(ret);
		bcl_computeOutputErrorFn_k = clCreateKernel(program, "bcl_computeOutputErrorFn", &ret);
		CheckError(ret);
		cpol_computeCeFn_k = clCreateKernel(program, "cpol_computeCeFn", &ret);
		CheckError(ret);
		cpol_computeOutputErrorFn_k= clCreateKernel(program, "cpol_computeOutputErrorFn", &ret);
		CheckError(ret);
		mcl_countCorrectClassificationsFn_k= clCreateKernel(program, "mcl_countCorrectClassificationsFn", &ret);
		CheckError(ret);
		mcl_computeCrossEntropyErrorFn_k= clCreateKernel(program, "mcl_computeCrossEntropyErrorFn", &ret);
		CheckError(ret);
		mcl_computeOutputErrorFn_k= clCreateKernel(program, "mcl_computeOutputErrorFn", &ret);
		CheckError(ret);
		rpol_calculateError_k = clCreateKernel(program, "rpol_calculateError", &ret);
		CheckError(ret);
		rpol_computeRmseFn_k  = clCreateKernel(program, "rpol_computeRmseFn", &ret);
		CheckError(ret);
		rpol_computeOutputErrorFn_k = clCreateKernel(program, "rpol_computeOutputErrorFn", &ret);
		CheckError(ret);
		sml_calculateOffsetFn_k = clCreateKernel(program, "sml_calculateOffsetFn", &ret);
		CheckError(ret);
		sml_calculateExpFn_k  = clCreateKernel(program, "sml_calculateExpFn", &ret);
		CheckError(ret);
		sml_sumUpOutputsFn_k  = clCreateKernel(program, "sml_sumUpOutputsFn", &ret);
		CheckError(ret);
		sml_normalizeOutputsFn_k = clCreateKernel(program, "sml_normalizeOutputsFn", &ret);
		CheckError(ret);
		sml_calculateErrorOffsetFn_k= clCreateKernel(program, "sml_calculateErrorOffsetFn", &ret);
		CheckError(ret);
		sml_calculateErrorsFn_k = clCreateKernel(program, "sml_calculateErrorsFn", &ret);
		CheckError(ret);
		smpol_computeSseMaskFn_k = clCreateKernel(program, "smpol_computeSseMaskFn", &ret);
		CheckError(ret);
		smpol_computeOutputErrorFn_k  = clCreateKernel(program, "smpol_computeOutputErrorFn", &ret);
		CheckError(ret);
		ll_computeBlockOutputFn_k = clCreateKernel(program, "ll_computeBlockOutputFn", &ret);
		CheckError(ret);
		ll_resortOutputsFn_k = clCreateKernel(program, "ll_resortOutputsFn", &ret);
		CheckError(ret);
		ll_resortOutputErrorsFn_k = clCreateKernel(program, "ll_resortOutputErrorsFn", &ret);
		CheckError(ret);
		ll_computeBlockErrorsFn_k = clCreateKernel(program, "ll_computeBlockErrorsFn", &ret);
		CheckError(ret);
		ll_computeWeightUpdateFn_k = clCreateKernel(program, "ll_computeWeightUpdateFn", &ret);
		CheckError(ret);
		sdo_updateWeightFn_k  = clCreateKernel(program, "sdo_updateWeightFn", &ret);
		CheckError(ret);
	};

public:

	cl_context & Context(){

		return context;
	}
	cl_command_queue Queue(){
		return command_queue;
	}
    cl_kernel & K_memset_float(){
    	return memset_float_k;
    }
    cl_kernel & K_mm_tranpose_afn(){
    	return mm_tranpose_afn_k;
    }
    cl_kernel & K_mm_tranpose_fn(){
    	return mm_tranpose_fn_k;
    }
    cl_kernel & K_mm_tranpose_bfn(){
    	return mm_tranpose_bfn_k;
    }

    cl_kernel & K_mma_tranpose_afn(){
    	return mma_tranpose_afn_k;
    }
    cl_kernel & K_mma_tranpose_fn(){
    	return mma_tranpose_fn_k;
    }
    cl_kernel & K_mma_tranpose_bfn(){
    	return mma_tranpose_bfn_k;
    }

    cl_kernel & K_r_plus()
    {
    	return r_plus_k;
    }
    cl_kernel & K_sseFn(){
    	return weightedSSeFn_k;
    }

    cl_kernel & K_outputErrorFn(){
    	return outputErrorFn_k;
    }

    cl_kernel & K_ffl_computeOutputFn(){
    	return ffl_computeOutputFn_k;
    }

    cl_kernel & K_ffl_computeDeltaFn(){
    	return ffl_computeDeltaFn_k;
    }
    cl_kernel & K_ffl_computeBiasWeightUpdateFn(){
    	return ffl_computeBiasWeightUpdateFn_k;
    }
    cl_kernel & K_spol_computeSseFn(){
    	return spol_computeSseFn_k;
    }
    cl_kernel & K_spol_computeOutputErrorFn(){
    	return spol_computeOutputErrorFn_k;
    }
    cl_kernel & K_bcl_countCorrectClassificationsFn(){
    	return bcl_countCorrectClassificationsFn_k;
    }
    cl_kernel & K_bcl_computeCrossEntropyErrorFn(){
    	return bcl_computeCrossEntropyErrorFn_k;
    }
    cl_kernel & K_bcl_computeOutputErrorFn_(){
    	return bcl_computeOutputErrorFn_k;
    }
    cl_kernel & K_cpol_computeCeFn(){
    	return cpol_computeCeFn_k;
    }
    cl_kernel & K_cpol_computeOutputErrorFn(){
    	return cpol_computeOutputErrorFn_k;
    }
    cl_kernel & K_mcl_countCorrectClassificationsFn(){
    	return mcl_countCorrectClassificationsFn_k;
    }
    cl_kernel & K_mcl_computeCrossEntropyErrorFn(){
    	return mcl_computeCrossEntropyErrorFn_k;
    }
    cl_kernel & K_mcl_computeOutputErrorFn(){
    	return mcl_computeOutputErrorFn_k;
    }
    cl_kernel & K_rpol_calculateError(){
    	return rpol_calculateError_k;
    }
    cl_kernel & K_rpol_computeRmseFn(){
    	return rpol_computeRmseFn_k;
    }
    cl_kernel & K_rpol_computeOutputErrorFn(){
    	return rpol_computeOutputErrorFn_k;
    }
    cl_kernel & K_sml_calculateOffsetFn(){
    	return sml_calculateOffsetFn_k;
    }
    cl_kernel & K_sml_calculateExpFn(){
    	return sml_calculateExpFn_k;
    }

    cl_kernel & K_sml_sumUpOutputsFn(){
    	return sml_sumUpOutputsFn_k;
    }
    cl_kernel & K_sml_normalizeOutputsFn(){
    	return sml_normalizeOutputsFn_k;
    }
    cl_kernel & K_sml_calculateErrorOffsetFn(){
    	return sml_calculateErrorOffsetFn_k;
    }
    cl_kernel & K_sml_calculateErrorsFn(){
    	return sml_calculateErrorsFn_k;
    }
    cl_kernel & K_smpol_computeSseMaskFn(){
    	return smpol_computeSseMaskFn_k;
    }
    cl_kernel & K_smpol_computeOutputErrorFn(){
    	return smpol_computeOutputErrorFn_k;
    }
    cl_kernel & K_ll_computeBlockOutputFn(){
    	return ll_computeBlockOutputFn_k;
    }
    cl_kernel & K_ll_resortOutputsFn(){
    	return ll_resortOutputsFn_k;
    }
    cl_kernel & K_ll_resortOutputErrorsFn(){
    	return ll_resortOutputErrorsFn_k;
    }
    cl_kernel & K_ll_computeBlockErrorsFn(){
    	return ll_computeBlockErrorsFn_k;
    }
    cl_kernel & K_ll_computeWeightUpdateFn(){
    	return ll_computeWeightUpdateFn_k;
    }
    cl_kernel & K_sdo_updateWeightFn(){
    	return sdo_updateWeightFn_k;
    }



    static void malloc_real(d_mem_real & rt, int size){

		rt.size = size;

		cl_int error;
		rt.mem =  clCreateBuffer (SystemCL::inst().Context(),
				CL_MEM_READ_WRITE, rt.numBytes(),NULL, &error);
		CheckError(error);


	}
    static void fill(const d_mem_real & a, real_t val){
    	if (a.size < 1)
    		return;

	    // Set the arguments of the kernel
	    cl_int ret = clSetKernelArg(SystemCL::inst().K_memset_float(), 0, sizeof(a.mem), (void *)&a.mem);
	    CheckError(ret);
	    ret = clSetKernelArg(SystemCL::inst().K_memset_float(), 1, sizeof(val), (void *)&val);
	    CheckError(ret);

	    // Execute the OpenCL kernel on the list
	    size_t global_item_size = a.size; // Process the entire lists

	    ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_memset_float(), 1, NULL,
	            &global_item_size, NULL, 0, NULL, NULL);
	    CheckError(ret);
    }

	static void malloc_real(d_mem_real & rt, int size, real_t val){
		rt.free();
		if ( size < 1)
			return ;
		rt.size = size;
		cl_int error;
		rt.mem =  clCreateBuffer (SystemCL::inst().Context(),
				CL_MEM_READ_WRITE, rt.numBytes(),NULL, &error);
		CheckError(error);

	    // Set the arguments of the kernel
	    cl_int ret = clSetKernelArg(SystemCL::inst().K_memset_float(), 0, sizeof(rt.mem), (void *)&rt.mem);
	    CheckError(ret);
	    ret = clSetKernelArg(SystemCL::inst().K_memset_float(), 1, sizeof(val), (void *)&val);
	    CheckError(ret);

	    // Execute the OpenCL kernel on the list
	    size_t global_item_size = size; // Process the entire lists

	    ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_memset_float(), 1, NULL,
	            &global_item_size, NULL, 0, NULL, NULL);
	    CheckError(ret);
	}

	static void malloc_char(d_mem_char & rt, int size){
		rt.free();
		if ( size < 1)
				return ;
		rt.size = size;
		cl_int error;
		rt.mem =  clCreateBuffer (SystemCL::inst().Context(),
				CL_MEM_READ_WRITE, rt.numBytes(),NULL, &error);
		CheckError(error);

	}

	static void malloc_int(d_mem_int & rt, int size){
		rt.free();
		if ( size < 1)
			return ;
		rt.size = size;
		cl_int error;
		rt.mem =  clCreateBuffer (SystemCL::inst().Context(),
				CL_MEM_READ_WRITE, rt.numBytes(),NULL, &error);
		CheckError(error);

	}
    static void  copy_char( d_mem_char & rt, std::vector<char> hvec){
		rt.size = hvec.size();
		cl_int error;
		rt.mem =  clCreateBuffer (SystemCL::inst().Context(),
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, rt.numBytes(),hvec.data(), &error);
		CheckError(error);
    }


    static void copy_real(d_mem_real & rt, std::vector<real_t> hvec){

    	if (hvec.size() < 1) return;

     	if (rt.mem  == 0 || rt.size == 0 ) malloc_real(rt, hvec.size());

		rt.size = hvec.size();
		cl_int error;
		error = clEnqueueWriteBuffer(SystemCL::inst().Queue(), rt.mem,
				CL_TRUE, 0,(size_t) rt.numBytes(), hvec.data(), 0, NULL, NULL);
		CheckError(error);

    }
    static void copy_real(d_mem_real& dest,const d_mem_real &src){
    	cl_int error = clEnqueueCopyBuffer(SystemCL::inst().Queue(),
    			src.mem, dest.mem, 0, 0, src.size * sizeof(float), 0 , NULL, NULL);
    	CheckError(error);
    }
    static void assign_real(d_mem_real& dest,const d_mem_real &src){
    	if (src.size < 1) return;
		cl_int error;
		dest.free();
		dest.size = src.size;
		dest.mem =  clCreateBuffer (SystemCL::inst().Context(),
				CL_MEM_READ_WRITE , src.size * sizeof(real_t),NULL, &error);
		CheckError(error);

		copy_real(dest, src);


    }
    static void copy_int(d_mem_int & rt, std::vector<int> hvec){
		rt.size = hvec.size();

		cl_int error;
		rt.mem =  clCreateBuffer (SystemCL::inst().Context(),
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, rt.numBytes(), hvec.data(), &error);
		CheckError(error);
    }

    static  void copy_int_real(d_mem_real & rt, std::vector<int> hvec){
		rt.size = hvec.size();

		cl_int error;
		rt.mem =  clCreateBuffer (SystemCL::inst().Context(),
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, rt.numBytes(), hvec.data(), &error);
		CheckError(error);
    }


    static std::vector<real_t> copy_real(const d_mem_real & d){
    	std::vector<real_t> rt(d.size);

		cl_int error;
		error = clEnqueueReadBuffer(inst().Queue(), d.mem, CL_TRUE, 0,
		            d.size * sizeof(real_t) , rt.data(), 0, NULL, NULL);
		CheckError(error);

    	return rt;
    }
    static std::vector<real_t> copy_real(d_mem_real & d, int start, int end){
    	int size = (end - start) + 1 ;
    	std::vector<real_t> rt(size);

		cl_int error;
		error = clEnqueueReadBuffer(inst().Queue(), d.mem, CL_TRUE, start,
		            size * sizeof(real_t) , rt.data(), 0, NULL, NULL);
		CheckError(error);
    	return rt;
    }

    static std::vector<char> copy_char(const d_mem_char  &d){
    	std::vector<char> rt(d.size);

		cl_int error;
		error = clEnqueueReadBuffer(inst().Queue(), d.mem, CL_TRUE, 0,
		            d.size * sizeof(char) , rt.data(), 0, NULL, NULL);
		CheckError(error);
    	return rt;
    }
    static void mm_tranpose_AFn(int rowsA, int rowsB, int colsA, int colsB,
    							int A_Offset, int B_offset,
    							cl_mem a, cl_mem b,
								int rowsC, int colsC, cl_mem c, int Offset
								)
    {
    	// Set the arguments of the kernel
		cl_int ret;
		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_afn(), 0, sizeof(rowsA), (void *)&rowsA);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_afn(), 1, sizeof(rowsB), (void *)&rowsB);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_afn(), 2, sizeof(colsA), (void *)&colsA);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_afn(), 3, sizeof(colsB), (void *)&colsB);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_afn(), 4, sizeof(A_Offset), (void *)&A_Offset);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_afn(), 5, sizeof(B_offset), (void *)&B_offset);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_afn(), 6, sizeof(a), (void *)&a);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_afn(), 7, sizeof(b), (void *)&b);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_afn(), 8, sizeof(c), (void *)&c);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_afn(), 9, sizeof(Offset), (void *)&Offset);
		CheckError(ret);
		// Execute the OpenCL kernel on the list
		size_t global_item_size =  rowsC * colsC; // Process the entire lists

		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_mm_tranpose_afn(), 1, NULL,
				&global_item_size, NULL, 0, NULL, NULL);
		CheckError(ret);





    }
    static void mm_tranpose_Fn(int rowsA, int rowsB, int colsA, int colsB,
    							int A_offset, int B_offset,
     							cl_mem a, cl_mem b,
 								int rowsC, int colsC, cl_mem c, int Offset
 								)
     {


     	// Set the arguments of the kernel
 		cl_int ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_fn(), 0, sizeof(rowsA), (void *)&rowsA);
 		CheckError(ret);
 		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_fn(), 1, sizeof(rowsB), (void *)&rowsB);
 		CheckError(ret);
 		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_fn(), 2, sizeof(colsA), (void *)&colsA);
 		CheckError(ret);
 		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_fn(), 3, sizeof(colsB), (void *)&colsB);
 		CheckError(ret);
 		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_fn(), 4, sizeof(A_offset), (void *)&A_offset);
 		CheckError(ret);
 		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_fn(), 5, sizeof(B_offset), (void *)&B_offset);
 		CheckError(ret);
 		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_fn(), 6, sizeof(a), (void *)&a);
 		CheckError(ret);
 		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_fn(), 7, sizeof(b), (void *)&b);
 		CheckError(ret);
 		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_fn(), 8, sizeof(c), (void *)&c);
 		CheckError(ret);
 		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_fn(), 9, sizeof(Offset), (void *)&Offset);
 		CheckError(ret);
 		// Execute the OpenCL kernel on the list
 		size_t global_item_size =  rowsC * colsC; // Process the entire lists

 		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_mm_tranpose_fn(), 1, NULL,
 				&global_item_size, NULL, 0, NULL, NULL);
 		CheckError(ret);


     }
    static void mm_tranpose_BFn(int rowsA, int rowsB, int colsA, int colsB,
    		                     int A_offset, int B_offset,
      							cl_mem a, cl_mem b,
  								int rowsC, int colsC, cl_mem c, int Offset
  								)
      {


      	// Set the arguments of the kernel
  		cl_int ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_bfn(), 0, sizeof(rowsA), (void *)&rowsA);
  		CheckError(ret);
  		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_bfn(), 1, sizeof(rowsB), (void *)&rowsB);
  		CheckError(ret);
  		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_bfn(), 2, sizeof(colsA), (void *)&colsA);
  		CheckError(ret);
  		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_bfn(), 3, sizeof(colsB), (void *)&colsB);
  		CheckError(ret);
 		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_bfn(), 4, sizeof(A_offset), (void *)&A_offset);
 		CheckError(ret);
 		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_bfn(), 5, sizeof(B_offset), (void *)&B_offset);
 		CheckError(ret);
  		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_bfn(), 6, sizeof(a), (void *)&a);
  		CheckError(ret);
  		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_bfn(), 7, sizeof(b), (void *)&b);
  		CheckError(ret);
  		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_bfn(), 8, sizeof(c), (void *)&c);
  		CheckError(ret);
  		ret = clSetKernelArg(SystemCL::inst().K_mm_tranpose_bfn(), 9, sizeof(Offset), (void *)&Offset);
  		CheckError(ret);
  		// Execute the OpenCL kernel on the list
  		size_t global_item_size =  rowsC * colsC; // Process the entire lists

  		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_mm_tranpose_bfn(), 1, NULL,
  				&global_item_size, NULL, 0, NULL, NULL);
  		CheckError(ret);


      }

    static void mma_tranpose_AFn(int rowsA, int rowsB, int colsA, int colsB,
    	                     	 int A_offset, int B_offset,
    							cl_mem a, cl_mem b,
								int rowsC, int colsC, cl_mem c, int Offset)  {

    	// Set the arguments of the kernel
		cl_int ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_afn(), 0, sizeof(rowsA), (void *)&rowsA);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_afn(), 1, sizeof(rowsB), (void *)&rowsB);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_afn(), 2, sizeof(colsA), (void *)&colsA);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_afn(), 3, sizeof(colsB), (void *)&colsB);
		CheckError(ret);
 		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_afn(), 4, sizeof(A_offset), (void *)&A_offset);
 		CheckError(ret);
 		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_afn(), 5, sizeof(B_offset), (void *)&B_offset);
 		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_afn(), 6, sizeof(a), (void *)&a);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_afn(), 7, sizeof(b), (void *)&b);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_afn(), 8, sizeof(c), (void *)&c);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_afn(), 9, sizeof(Offset), (void *)&Offset);
		CheckError(ret);
		// Execute the OpenCL kernel on the list
		size_t global_item_size =  rowsC * colsC; // Process the entire lists

		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_mma_tranpose_afn(), 1, NULL,
				&global_item_size, NULL, 0, NULL, NULL);
		CheckError(ret);


    }
    static void mma_tranpose_BFn(int rowsA, int rowsB, int colsA, int colsB,
    		                   int A_offset, int B_offset,
      							cl_mem a, cl_mem b,
  								int rowsC, int colsC, cl_mem c, int Offset){
      	// Set the arguments of the kernel
  		cl_int ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_bfn(), 0, sizeof(rowsA), (void *)&rowsA);
  		CheckError(ret);
  		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_bfn(), 1, sizeof(rowsB), (void *)&rowsB);
  		CheckError(ret);
  		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_bfn(), 2, sizeof(colsA), (void *)&colsA);
  		CheckError(ret);
  		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_bfn(), 3, sizeof(colsB), (void *)&colsB);
  		CheckError(ret);
 		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_bfn(), 4, sizeof(A_offset), (void *)&A_offset);
 		CheckError(ret);
 		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_bfn(), 5, sizeof(B_offset), (void *)&B_offset);
 		CheckError(ret);
  		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_bfn(), 6, sizeof(a), (void *)&a);
  		CheckError(ret);
  		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_bfn(), 7, sizeof(b), (void *)&b);
  		CheckError(ret);
  		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_bfn(), 8, sizeof(c), (void *)&c);
  		CheckError(ret);
  		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_bfn(), 9, sizeof(Offset), (void *)&Offset);
  		CheckError(ret);
  		// Execute the OpenCL kernel on the list
  		size_t global_item_size =  rowsC * colsC; // Process the entire lists

  		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_mma_tranpose_bfn(), 1, NULL,
  				&global_item_size, NULL, 0, NULL, NULL);
  		CheckError(ret);


      }
    static void mma_tranpose_Fn(int rowsA, int rowsB, int colsA, int colsB,
    		                     int A_offset, int B_offset,
       							cl_mem a, cl_mem b,
   								int rowsC, int colsC, cl_mem c, int Offset
   								) {
       	// Set the arguments of the kernel
   		cl_int ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_fn(), 0, sizeof(rowsA), (void *)&rowsA);
   		CheckError(ret);
   		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_fn(), 1, sizeof(rowsB), (void *)&rowsB);
   		CheckError(ret);
   		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_fn(), 2, sizeof(colsA), (void *)&colsA);
   		CheckError(ret);
   		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_fn(), 3, sizeof(colsB), (void *)&colsB);
   		CheckError(ret);
 		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_fn(), 4, sizeof(A_offset), (void *)&A_offset);
 		CheckError(ret);
 		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_fn(), 5, sizeof(B_offset), (void *)&B_offset);
 		CheckError(ret);
   		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_fn(), 6, sizeof(a), (void *)&a);
   		CheckError(ret);
   		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_fn(), 7, sizeof(b), (void *)&b);
   		CheckError(ret);
   		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_fn(), 8, sizeof(c), (void *)&c);
   		CheckError(ret);
   		ret = clSetKernelArg(SystemCL::inst().K_mma_tranpose_fn(), 9, sizeof(Offset), (void *)&Offset);
   		CheckError(ret);
   		// Execute the OpenCL kernel on the list
   		size_t global_item_size =  rowsC * colsC; // Process the entire lists

   		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_mma_tranpose_fn(), 1, NULL,
   				&global_item_size, NULL, 0, NULL, NULL);
   		CheckError(ret);


       }

    static void r_plus(cl_mem  a, cl_mem  b, int size	)
       {

       	// Set the arguments of the kernel
   		cl_int ret = clSetKernelArg(SystemCL::inst().K_r_plus(), 0, sizeof(a), (void *)&a);
   		CheckError(ret);
   		ret = clSetKernelArg(SystemCL::inst().K_r_plus(), 1, sizeof(b), (void *)&b);
   		CheckError(ret);
   		// Execute the OpenCL kernel on the list
   		size_t global_item_size =  size; // Process the entire lists

   		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_r_plus(), 1, NULL,
   				&global_item_size, NULL, 0, NULL, NULL);
   		CheckError(ret);

       }

    static float weighedSSeFn(int layerSize, const d_mem_char &patTypes, d_mem_real &targets,
    								d_mem_real& outputs, int n){
    	float rt = 0;
    	cl_int ret;
       	// Set the arguments of the kernel
        size_t block_size = 256;
		cl_mem mem =  clCreateBuffer (SystemCL::inst().Context(),
				CL_MEM_READ_WRITE, sizeof(real_t), NULL, &ret);
		CheckError(ret);

   		ret = 		clSetKernelArg(SystemCL::inst().K_sseFn(), 0, sizeof(layerSize), (void *)&layerSize);
   		CheckError(ret);
   		ret =        clSetKernelArg(SystemCL::inst().K_sseFn(), 1, sizeof(patTypes.mem), (void *)&patTypes.mem);
   		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_sseFn(), 2, sizeof(targets.mem), (void *)&targets.mem);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_sseFn(), 3, sizeof(outputs.mem), (void *)&outputs.mem);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_sseFn(), 4,block_size * sizeof(real_t), NULL);
		CheckError(ret);

		ret =        clSetKernelArg(SystemCL::inst().K_sseFn(), 5, sizeof(mem), (void *) &mem);
		CheckError(ret);

		ret =        clSetKernelArg(SystemCL::inst().K_sseFn(), 6,  sizeof(n), (void *)&n);
		CheckError(ret);

   		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_sseFn(), 1, NULL,
   				&block_size, &block_size, 0, NULL, NULL);
   		CheckError(ret);


   		ret = clEnqueueReadBuffer(inst().Queue(), mem, CL_TRUE, 0,
		            sizeof(real_t) , &rt, 0, NULL, NULL);
		CheckError(ret);

		clReleaseMemObject(mem);

    	return rt;
    }

    static void outputErrorFn(int layerSize,const d_mem_char &patTypes, d_mem_real & targets,
			d_mem_real & outputs, d_mem_real & errorOutput, int n){

    	cl_int ret;

   		ret = 		clSetKernelArg(SystemCL::inst().K_outputErrorFn(), 0, sizeof(layerSize), (void *)&layerSize);
   		CheckError(ret);
   		ret = 		clSetKernelArg(SystemCL::inst().K_outputErrorFn(), 1, sizeof(patTypes.mem), (void *)&patTypes.mem);
   		CheckError(ret);
   		ret = 		clSetKernelArg(SystemCL::inst().K_outputErrorFn(), 2, sizeof(targets.mem), (void *)&targets.mem);
   		CheckError(ret);
   		ret = 		clSetKernelArg(SystemCL::inst().K_outputErrorFn(), 3, sizeof(outputs.mem), (void *)&outputs.mem);
   		CheckError(ret);
   		ret = 		clSetKernelArg(SystemCL::inst().K_outputErrorFn(), 4, sizeof(errorOutput.mem), (void *)&errorOutput.mem);
   		CheckError(ret);

   		size_t global_size = n;
   		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_outputErrorFn(), 1, NULL,
   				&global_size, NULL, 0, NULL, NULL);
   		CheckError(ret);

    }

    static void ffl_computeOutputFn(int layerSize, float bias, d_mem_real  &biasWeights, int biasOffset,
    			d_mem_real & ouput, int n, int typeFunction)
    {
   		cl_int ret ;
   		ret = clSetKernelArg(SystemCL::inst().K_ffl_computeOutputFn(), 0, sizeof(layerSize), (void *)&layerSize);
   		CheckError(ret);
   		ret = clSetKernelArg(SystemCL::inst().K_ffl_computeOutputFn(), 1, sizeof(bias), (void *)&bias);
   		CheckError(ret);
   		ret = clSetKernelArg(SystemCL::inst().K_ffl_computeOutputFn(), 2, sizeof(biasWeights.mem), (void *)&biasWeights.mem);
   		CheckError(ret);
   		ret = clSetKernelArg(SystemCL::inst().K_ffl_computeOutputFn(), 3, sizeof(biasOffset), (void *)&biasOffset);
   		CheckError(ret);
   		ret = clSetKernelArg(SystemCL::inst().K_ffl_computeOutputFn(), 4, sizeof(ouput.mem), (void *)&ouput.mem);
   		CheckError(ret);
   		ret = clSetKernelArg(SystemCL::inst().K_ffl_computeOutputFn(), 5, sizeof(typeFunction), (void *)&typeFunction);
   		CheckError(ret);

   		size_t global_item_size = n;
   		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_ffl_computeOutputFn(), 1, NULL,
   				&global_item_size, NULL, 0, NULL, NULL);
   		CheckError(ret);


    }

    static void ffl_computeDeltaFn(d_mem_real &outputErrors, d_mem_real& outputs, int n, int typeAF){


   		cl_int ret ;
   		ret = clSetKernelArg(SystemCL::inst().K_ffl_computeDeltaFn(), 0, sizeof(outputErrors.mem), (void *)&outputErrors.mem);
   		CheckError(ret);
   		ret = clSetKernelArg(SystemCL::inst().K_ffl_computeDeltaFn(), 1, sizeof(outputs.mem), (void *)&outputs.mem);
   		CheckError(ret);
   		ret = clSetKernelArg(SystemCL::inst().K_ffl_computeDeltaFn(), 2, sizeof(typeAF), (void *)&typeAF);
   		CheckError(ret);
   		size_t global_item_size = n;
   		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_ffl_computeDeltaFn(), 1, NULL,
   				&global_item_size, NULL, 0, NULL, NULL);
   		CheckError(ret);

    }

    static void ffl_computeBiasWeightUpdateFn(int layerSize,int patternCount,
    		                   float bias, d_mem_real& deltas,
    		                   const d_mem_real& out, int offset, int n ){

   		cl_int ret ;
   		ret = clSetKernelArg(SystemCL::inst().K_ffl_computeBiasWeightUpdateFn(), 0,
   				sizeof(layerSize), (void *)&layerSize);
   		CheckError(ret);
   		ret = clSetKernelArg(SystemCL::inst().K_ffl_computeBiasWeightUpdateFn(), 1,
   				sizeof(patternCount), (void *)&patternCount);
   		CheckError(ret);
   		ret = clSetKernelArg(SystemCL::inst().K_ffl_computeBiasWeightUpdateFn(), 2,
   				sizeof(bias), (void *)&bias);
   		ret = clSetKernelArg(SystemCL::inst().K_ffl_computeBiasWeightUpdateFn(), 3,
   				sizeof(deltas.mem), (void *)&deltas.mem);
   		CheckError(ret);
   		ret = clSetKernelArg(SystemCL::inst().K_ffl_computeBiasWeightUpdateFn(), 4,
   				sizeof(out.mem), (void *)&out.mem);
   		CheckError(ret);
   		ret = clSetKernelArg(SystemCL::inst().K_ffl_computeBiasWeightUpdateFn(), 5,
   				sizeof(offset), (void *)&offset);
   		CheckError(ret);

   		size_t global_item_size = n;
   		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_ffl_computeBiasWeightUpdateFn(), 1, NULL,
   				&global_item_size, NULL, 0, NULL, NULL);
   		CheckError(ret);


    }

    static float spol_computeSseFn(int layerSize,const d_mem_char &patTypes, d_mem_real& targets,
    		  d_mem_real &actualOuput, int n){
    	float rt = 0;

		cl_int ret;
		// Set the arguments of the kernel
		size_t block_size = 256;
		cl_mem mem =  clCreateBuffer (SystemCL::inst().Context(),
				CL_MEM_READ_WRITE, sizeof(real_t), NULL, &ret);
		CheckError(ret);

		ret = 		clSetKernelArg(SystemCL::inst().K_spol_computeSseFn(), 0, sizeof(layerSize), (void *)&layerSize);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_spol_computeSseFn(), 1, sizeof(patTypes.mem), (void *)&patTypes.mem);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_spol_computeSseFn(), 2, sizeof(targets.mem), (void *)&targets.mem);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_spol_computeSseFn(), 3, sizeof(actualOuput.mem), (void *)&actualOuput.mem);
		CheckError(ret);

		ret =        clSetKernelArg(SystemCL::inst().K_spol_computeSseFn(), 4, sizeof(mem), (void *) &mem);
		CheckError(ret);

		ret =        clSetKernelArg(SystemCL::inst().K_spol_computeSseFn(), 5,block_size * sizeof(real_t), NULL);
		CheckError(ret);

		ret =        clSetKernelArg(SystemCL::inst().K_spol_computeSseFn(), 6,  sizeof(n), (void *)&n);
		CheckError(ret);

		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_spol_computeSseFn(), 1, NULL,
				&block_size, &block_size, 0, NULL, NULL);
		CheckError(ret);


		ret = clEnqueueReadBuffer(inst().Queue(), mem, CL_TRUE, 0,
					sizeof(real_t) , &rt, 0, NULL, NULL);
		CheckError(ret);

		clReleaseMemObject(mem);

    	return rt;
    }

    static void spol_computeOutputErrorFn(int layerSize, const d_mem_char & patTypes,d_mem_real &actualOutputs,
    		d_mem_real&targets, d_mem_real& outputErrors, int n ){
    	cl_int ret ;
		ret = clSetKernelArg(SystemCL::inst().K_spol_computeOutputErrorFn(), 0, sizeof(layerSize), (void *)&layerSize);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_spol_computeOutputErrorFn(), 1, sizeof(patTypes.mem), (void *)&patTypes.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_spol_computeOutputErrorFn(), 2, sizeof(actualOutputs.mem), (void *)&actualOutputs.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_spol_computeOutputErrorFn(), 3, sizeof(targets.mem), (void *)&targets.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_spol_computeOutputErrorFn(), 4, sizeof(outputErrors.mem), (void *)&outputErrors.mem);
		CheckError(ret);
   		size_t global_item_size = n;
   		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_spol_computeOutputErrorFn(), 1, NULL,
   				&global_item_size, NULL, 0, NULL, NULL);
   		CheckError(ret);


    }


    static int bcl_countCorrectClassificationsFn(d_mem_real & targets, d_mem_real & actualOutputs,const d_mem_char & patTypes, int n){
    	int count = 0;
		cl_int ret;
		// Set the arguments of the kernel
		size_t block_size = 256;
		cl_mem mem =  clCreateBuffer (SystemCL::inst().Context(),
				CL_MEM_READ_WRITE, sizeof(int), NULL, &ret);
		CheckError(ret);


		ret = 		clSetKernelArg(SystemCL::inst().K_bcl_countCorrectClassificationsFn(), 0, sizeof(targets.mem), (void *)&targets.mem);
		CheckError(ret);
		ret = 		clSetKernelArg(SystemCL::inst().K_bcl_countCorrectClassificationsFn(), 1, sizeof(actualOutputs.mem), (void *)&actualOutputs.mem);
		CheckError(ret);
		ret = 		clSetKernelArg(SystemCL::inst().K_bcl_countCorrectClassificationsFn(), 2, sizeof(patTypes.mem), (void *)&patTypes.mem);
		CheckError(ret);
		ret = 		clSetKernelArg(SystemCL::inst().K_bcl_countCorrectClassificationsFn(), 3, sizeof(mem), (void *)&mem);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_bcl_countCorrectClassificationsFn(), 4,block_size * sizeof(int), NULL);
		CheckError(ret);
		ret = 		clSetKernelArg(SystemCL::inst().K_bcl_countCorrectClassificationsFn(), 5, sizeof(n), (void *)&n);
		CheckError(ret);


		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_bcl_countCorrectClassificationsFn(), 1, NULL,
				&block_size, &block_size, 0, NULL, NULL);
		CheckError(ret);


		ret = clEnqueueReadBuffer(inst().Queue(), mem, CL_TRUE, 0,
					sizeof(int) , &count, 0, NULL, NULL);
		CheckError(ret);

		clReleaseMemObject(mem);



    	return count;
    }
    static float bcl_computeCrossEntropyErrorFn(const d_mem_char &patTypes,d_mem_real &targets,
    		                  d_mem_real & actualOutputs, int n){
    	float rt;
    	cl_int ret;
    			// Set the arguments of the kernel
		size_t block_size = 256;
		cl_mem mem =  clCreateBuffer (SystemCL::inst().Context(),
				CL_MEM_READ_WRITE, sizeof(real_t), NULL, &ret);
		CheckError(ret);

		ret = 		clSetKernelArg(SystemCL::inst().K_bcl_computeCrossEntropyErrorFn(), 0, sizeof(patTypes.mem), (void *)&patTypes.mem);
		CheckError(ret);

		ret = 		clSetKernelArg(SystemCL::inst().K_bcl_computeCrossEntropyErrorFn(), 1, sizeof(targets.mem), (void *)&targets.mem);
		CheckError(ret);

		ret = 		clSetKernelArg(SystemCL::inst().K_bcl_computeCrossEntropyErrorFn(), 2, sizeof(actualOutputs.mem), (void *)&actualOutputs.mem);
		CheckError(ret);

		ret =        clSetKernelArg(SystemCL::inst().K_bcl_computeCrossEntropyErrorFn(), 3, sizeof(mem), (void *) &mem);
		CheckError(ret);

		ret =        clSetKernelArg(SystemCL::inst().K_bcl_computeCrossEntropyErrorFn(), 4,block_size * sizeof(real_t), NULL);
		CheckError(ret);

		ret =        clSetKernelArg(SystemCL::inst().K_bcl_computeCrossEntropyErrorFn(), 5,  sizeof(n), (void *)&n);
		CheckError(ret);

		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_bcl_computeCrossEntropyErrorFn(), 1, NULL,
				&block_size, &block_size, 0, NULL, NULL);
		CheckError(ret);

		ret = clEnqueueReadBuffer(inst().Queue(), mem, CL_TRUE, 0,
					sizeof(real_t) , &rt, 0, NULL, NULL);
		CheckError(ret);

		clReleaseMemObject(mem);
    	return rt;
    }
    static void bcl_computeOutputErrorFn( const d_mem_char &patTypes, d_mem_real& outputErrors,
    		       d_mem_real& targets, d_mem_real &actualOutputs,int n){
    	cl_int ret ;
		ret = clSetKernelArg(SystemCL::inst().K_bcl_computeCrossEntropyErrorFn(), 0, sizeof(patTypes.mem), (void *)&patTypes.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_bcl_computeCrossEntropyErrorFn(), 1, sizeof(targets.mem), (void *)&targets.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_bcl_computeCrossEntropyErrorFn(), 2, sizeof(outputErrors.mem), (void *)&outputErrors.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_bcl_computeCrossEntropyErrorFn(), 3, sizeof(actualOutputs.mem), (void *)&actualOutputs.mem);
		CheckError(ret);


		size_t global_item_size = n;
		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_bcl_computeCrossEntropyErrorFn(), 1, NULL,
				&global_item_size, NULL, 0, NULL, NULL);
		CheckError(ret);

    }

    static float cpol_computeCeFn(int layerSize,const d_mem_char& patTypes, d_mem_real& targets, d_mem_real& actualOutputs,int n){
    	float rt = 0;

		cl_int ret;
		// Set the arguments of the kernel
		size_t block_size = 256;
		cl_mem mem =  clCreateBuffer (SystemCL::inst().Context(),
				CL_MEM_READ_WRITE, sizeof(real_t), NULL, &ret);
		CheckError(ret);

		ret = 		clSetKernelArg(SystemCL::inst().K_cpol_computeCeFn(), 0, sizeof(layerSize), (void *)&layerSize);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_cpol_computeCeFn(), 1, sizeof(patTypes.mem), (void *)&patTypes.mem);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_cpol_computeCeFn(), 2, sizeof(targets.mem), (void *)&targets.mem);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_cpol_computeCeFn(), 3, sizeof(actualOutputs.mem), (void *)&actualOutputs.mem);
		CheckError(ret);

		ret =        clSetKernelArg(SystemCL::inst().K_cpol_computeCeFn(), 4, sizeof(mem), (void *) &mem);
		CheckError(ret);

		ret =        clSetKernelArg(SystemCL::inst().K_cpol_computeCeFn(), 5,block_size * sizeof(real_t), NULL);
		CheckError(ret);

		ret =        clSetKernelArg(SystemCL::inst().K_cpol_computeCeFn(), 6,  sizeof(n), (void *)&n);
		CheckError(ret);

		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_cpol_computeCeFn(), 1, NULL,
				&block_size, &block_size, 0, NULL, NULL);
		CheckError(ret);

		ret = clEnqueueReadBuffer(inst().Queue(), mem, CL_TRUE, 0,
					sizeof(real_t) , &rt, 0, NULL, NULL);
		CheckError(ret);

		clReleaseMemObject(mem);
    	return rt;
    }
    static void cpol_computeOutputErrorFn(int layerSize, const d_mem_char & patTypes, d_mem_real & actualOutputs,
    	       d_mem_real &targets,d_mem_real &outputErrors, int n){
    	cl_int ret ;
		ret = clSetKernelArg(SystemCL::inst().K_cpol_computeOutputErrorFn(), 0, sizeof(layerSize), (void *)&layerSize);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_cpol_computeOutputErrorFn(), 1, sizeof(patTypes.mem), (void *)&patTypes.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_cpol_computeOutputErrorFn(), 2, sizeof(actualOutputs.mem), (void *)&actualOutputs.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_cpol_computeOutputErrorFn(), 3, sizeof(targets.mem), (void *)&targets.mem);
		CheckError(ret);


		ret = clSetKernelArg(SystemCL::inst().K_cpol_computeOutputErrorFn(), 4, sizeof(outputErrors.mem), (void *)&outputErrors.mem);
		CheckError(ret);
		size_t global_item_size = n;
		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_cpol_computeOutputErrorFn(), 1, NULL,
				&global_item_size, NULL, 0, NULL, NULL);
		CheckError(ret);
    }

    static int mcl_countCorrectClassificationsFn(int layerSize, d_mem_real &actualOutputs,d_mem_int & m_patTargetClasses,int  n){
    	int rt;

		cl_int ret;
		// Set the arguments of the kernel
		size_t block_size = 256;
		cl_mem mem =  clCreateBuffer (SystemCL::inst().Context(),
				CL_MEM_READ_WRITE, sizeof(int), NULL, &ret);
		CheckError(ret);

		ret = 		clSetKernelArg(SystemCL::inst().K_mcl_countCorrectClassificationsFn(), 0, sizeof(layerSize), (void *)&layerSize);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_mcl_countCorrectClassificationsFn(), 1, sizeof(actualOutputs.mem), (void *)&actualOutputs.mem);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_mcl_countCorrectClassificationsFn(), 2, sizeof(m_patTargetClasses.mem), (void *)&m_patTargetClasses.mem);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_mcl_countCorrectClassificationsFn(), 3, sizeof(mem), (void *) &mem);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_mcl_countCorrectClassificationsFn(), 4, block_size * sizeof(int), NULL);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_mcl_countCorrectClassificationsFn(), 5,  sizeof(n), (void *)&n);
		CheckError(ret);

		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_mcl_countCorrectClassificationsFn(), 1, NULL,
				&block_size, &block_size, 0, NULL, NULL);
		CheckError(ret);

		ret = clEnqueueReadBuffer(inst().Queue(), mem, CL_TRUE, 0,
					sizeof(int) , &rt, 0, NULL, NULL);
		CheckError(ret);

		clReleaseMemObject(mem);
    	return rt;
    }

    static float mcl_computeCrossEntropyErrorFn(int layerSize,d_mem_real & actualOutputs,
            		d_mem_int & m_patTargetClasses, int n){
    	float rt;
    	cl_int ret;
		// Set the arguments of the kernel
		size_t block_size = 256;
		cl_mem mem =  clCreateBuffer (SystemCL::inst().Context(),
				CL_MEM_READ_WRITE, sizeof(real_t), NULL, &ret);
		CheckError(ret);
		ret = 		clSetKernelArg(SystemCL::inst().K_mcl_computeCrossEntropyErrorFn(), 0, sizeof(layerSize), (void *)&layerSize);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_mcl_computeCrossEntropyErrorFn(), 1, sizeof(actualOutputs.mem), (void *)&actualOutputs.mem);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_mcl_computeCrossEntropyErrorFn(), 2, sizeof(m_patTargetClasses.mem), (void *)&m_patTargetClasses.mem);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_mcl_computeCrossEntropyErrorFn(), 3, sizeof(mem), (void *) &mem);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_mcl_computeCrossEntropyErrorFn(), 4, block_size * sizeof(real_t), NULL);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_mcl_computeCrossEntropyErrorFn(), 5,  sizeof(n), (void *)&n);
		CheckError(ret);

		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_mcl_computeCrossEntropyErrorFn(), 1, NULL,
				&block_size, &block_size, 0, NULL, NULL);
		CheckError(ret);

		ret = clEnqueueReadBuffer(inst().Queue(), mem, CL_TRUE, 0,
					sizeof(real_t) , &rt, 0, NULL, NULL);
		CheckError(ret);

		clReleaseMemObject(mem);
    	return rt;
    }

    static void mcl_computeOutputErrorFn(int layerSize,d_mem_real & actualOutputs,d_mem_real& outputErrors, d_mem_int & m_patTargetClasses,int n ){
    	cl_int ret ;
		ret = clSetKernelArg(SystemCL::inst().K_mcl_computeOutputErrorFn(), 0, sizeof(layerSize), (void *)&layerSize);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_mcl_computeOutputErrorFn(), 1, sizeof(actualOutputs.mem), (void *)&actualOutputs.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_mcl_computeOutputErrorFn(), 2, sizeof(outputErrors.mem), (void *)&outputErrors.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_mcl_computeOutputErrorFn(), 3, sizeof(m_patTargetClasses.mem), (void *)&m_patTargetClasses.mem);
		CheckError(ret);

		size_t global_item_size = n;
		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_mcl_computeOutputErrorFn(), 1, NULL,
				&global_item_size, NULL, 0, NULL, NULL);
		CheckError(ret);

    }


	static void fill(d_mem_real & mem, int size, float value ){
		int ret = 		clSetKernelArg(SystemCL::inst().K_memset_float(), 0, sizeof(mem.mem), (void *)&mem.mem);
		CheckError(ret);
		ret =           clSetKernelArg(SystemCL::inst().K_memset_float(), 1, sizeof(value), (void *)&value);
		CheckError(ret);
		size_t global_item_size = size;
		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_memset_float(), 1, NULL,
				&global_item_size, NULL, 0, NULL, NULL);
		CheckError(ret);

    }

	static float rpol_calculateError(d_mem_real &m_rmses, int n){
		float rt;
	    	cl_int ret;
			// Set the arguments of the kernel
			size_t block_size = 256;
			cl_mem mem =  clCreateBuffer (SystemCL::inst().Context(),
					CL_MEM_READ_WRITE, sizeof(real_t), NULL, &ret);
			CheckError(ret);
			ret =        clSetKernelArg(SystemCL::inst().K_rpol_calculateError(), 0, sizeof(m_rmses.mem), (void *)&m_rmses.mem);
			CheckError(ret);
			ret =        clSetKernelArg(SystemCL::inst().K_rpol_calculateError(), 1, sizeof(mem), (void *) &mem);
			CheckError(ret);
			ret =        clSetKernelArg(SystemCL::inst().K_rpol_calculateError(), 2, block_size * sizeof(real_t), NULL);
			CheckError(ret);
			ret =        clSetKernelArg(SystemCL::inst().K_rpol_calculateError(), 3,  sizeof(n), (void *)&n);
			CheckError(ret);

			ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_rpol_calculateError(), 1, NULL,
					&block_size, &block_size, 0, NULL, NULL);
			CheckError(ret);

			ret = clEnqueueReadBuffer(inst().Queue(), mem, CL_TRUE, 0,
						sizeof(real_t) , &rt, 0, NULL, NULL);
			CheckError(ret);

			clReleaseMemObject(mem);
			return rt;
	}
	static void rpol_computeRmseFn(int layerSize,d_mem_real& actualOutputs,
    		d_mem_real & targets, const d_mem_char& patTypes, d_mem_real& m_rmses , int n){

    	cl_int ret ;
		ret = clSetKernelArg(SystemCL::inst().K_rpol_computeRmseFn(), 0, sizeof(layerSize), (void *)&layerSize);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_rpol_computeRmseFn(), 1, sizeof(actualOutputs.mem), (void *)&actualOutputs.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_rpol_computeRmseFn(), 2, sizeof(targets.mem), (void *)&targets.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_rpol_computeRmseFn(), 3, sizeof(patTypes.mem), (void *)&patTypes.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_rpol_computeRmseFn(), 4, sizeof(m_rmses.mem), (void *)&m_rmses.mem);
		CheckError(ret);

		size_t global_item_size = n;
		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_rpol_computeRmseFn(), 1, NULL,
				&global_item_size, NULL, 0, NULL, NULL);
		CheckError(ret);

	}
	static void rpol_computeOutputErrorFn(int layerSize,d_mem_real& m_rmses,
			d_mem_real& actualOutputs, d_mem_real &targets,d_mem_real& outputErrors,int n){

    	cl_int ret ;
		ret = clSetKernelArg(SystemCL::inst().K_rpol_computeOutputErrorFn(), 0, sizeof(layerSize), (void *)&layerSize);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_rpol_computeOutputErrorFn(), 1, sizeof(actualOutputs.mem), (void *)&actualOutputs.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_rpol_computeOutputErrorFn(), 2, sizeof(targets.mem), (void *)&targets.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_rpol_computeOutputErrorFn(), 3, sizeof(outputErrors.mem), (void *)&outputErrors.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_rpol_computeOutputErrorFn(), 4, sizeof(m_rmses.mem), (void *)&m_rmses.mem);
		CheckError(ret);

		size_t global_item_size = n;
		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_rpol_computeOutputErrorFn(), 1, NULL,
				&global_item_size, NULL, 0, NULL, NULL);
		CheckError(ret);

	}

	static void sml_calculateOffsetFn(int layerSize, d_mem_real &outputs,
			const d_mem_char& patTypes, d_mem_real &m_patTmp,int n){

    	cl_int ret ;
		ret = clSetKernelArg(SystemCL::inst().K_sml_calculateOffsetFn(), 0, sizeof(layerSize), (void *)&layerSize);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_sml_calculateOffsetFn(), 1, sizeof(outputs.mem), (void *)&outputs.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_sml_calculateOffsetFn(), 2, sizeof(patTypes.mem), (void *)&patTypes.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_sml_calculateOffsetFn(), 3, sizeof(m_patTmp.mem), (void *)&m_patTmp.mem);
		CheckError(ret);


		size_t global_item_size = n;
		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_sml_calculateOffsetFn(), 1, NULL,
				&global_item_size, NULL, 0, NULL, NULL);
		CheckError(ret);
	}
	static void sml_calculateExpFn(int layerSize, d_mem_real &m_patTmp,
			d_mem_real& outputs,int n ){
    	cl_int ret ;
		ret = clSetKernelArg(SystemCL::inst().K_sml_calculateExpFn(), 0, sizeof(layerSize), (void *)&layerSize);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_sml_calculateExpFn(), 1, sizeof(m_patTmp.mem), (void *)&m_patTmp.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_sml_calculateExpFn(), 2, sizeof(outputs.mem), (void *)&outputs.mem);
		CheckError(ret);



		size_t global_item_size = n;
		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_sml_calculateExpFn(), 1, NULL,
				&global_item_size, NULL, 0, NULL, NULL);
		CheckError(ret);
	}

	static void sml_sumUpOutputsFn(int layerSize,d_mem_real & outputs,d_mem_real& m_patTmp,int  n){
		cl_int ret ;
		ret = clSetKernelArg(SystemCL::inst().K_sml_sumUpOutputsFn(), 0, sizeof(layerSize), (void *)&layerSize);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_sml_sumUpOutputsFn(), 1, sizeof(outputs.mem), (void *)&outputs.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_sml_sumUpOutputsFn(), 2, sizeof(m_patTmp.mem), (void *)&m_patTmp.mem);
		CheckError(ret);

		size_t global_item_size = n;
		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_sml_sumUpOutputsFn(), 1, NULL,
				&global_item_size, NULL, 0, NULL, NULL);
		CheckError(ret);
	}
    static void sml_normalizeOutputsFn(int layerSize,d_mem_real& m_patTmp,d_mem_real& outputs,int  n){
    	cl_int ret ;
		ret = clSetKernelArg(SystemCL::inst().K_sml_normalizeOutputsFn(), 0, sizeof(layerSize), (void *)&layerSize);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_sml_normalizeOutputsFn(), 1, sizeof(m_patTmp.mem), (void *)&m_patTmp.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_sml_normalizeOutputsFn(), 2, sizeof(outputs.mem), (void *)&outputs.mem);
		CheckError(ret);
		size_t global_item_size = n;
		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_sml_normalizeOutputsFn(), 1, NULL,
				&global_item_size, NULL, 0, NULL, NULL);
		CheckError(ret);

    }
    static void sml_calculateErrorOffsetFn(int layerSize,d_mem_real& outputs,d_mem_real &outputErrors,
    		const d_mem_char& patTypes, d_mem_real &m_patTmp, int n){
    	cl_int ret ;
		ret = clSetKernelArg(SystemCL::inst().K_sml_calculateErrorOffsetFn(), 0, sizeof(layerSize), (void *)&layerSize);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_sml_calculateErrorOffsetFn(), 1, sizeof(outputs.mem), (void *)&outputs.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_sml_calculateErrorOffsetFn(), 2, sizeof(outputErrors.mem), (void *)&outputErrors.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_sml_calculateErrorOffsetFn(), 3, sizeof(patTypes.mem), (void *)&patTypes.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_sml_calculateErrorOffsetFn(), 4, sizeof(m_patTmp.mem), (void *)&m_patTmp.mem);
		CheckError(ret);

		size_t global_item_size = n;
		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_sml_calculateErrorOffsetFn(), 1, NULL,
				&global_item_size, NULL, 0, NULL, NULL);
		CheckError(ret);

    }
    static void sml_calculateErrorsFn(int layerSize,d_mem_real & m_patTmp , d_mem_real &outputErrors,
    		d_mem_real& outputs, int n){
    	cl_int ret ;
		ret = clSetKernelArg(SystemCL::inst().K_sml_calculateErrorsFn(), 0, sizeof(layerSize), (void *)&layerSize);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_sml_calculateErrorsFn(), 1, sizeof(m_patTmp.mem), (void *)&m_patTmp.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_sml_calculateErrorsFn(), 2, sizeof(outputErrors.mem), (void *)&outputErrors.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_sml_calculateErrorsFn(), 3, sizeof(outputs.mem), (void *)&outputs.mem);
		CheckError(ret);

		size_t global_item_size = n;
		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_sml_calculateErrorsFn(), 1, NULL,
				&global_item_size, NULL, 0, NULL, NULL);
		CheckError(ret);
    }

    static float smpol_computeSseMaskFn(int layerSize,const d_mem_char& patTypes,
			d_mem_real & targets, d_mem_real& actualOutputs,int n){
    	float rt;
		cl_int ret;
		// Set the arguments of the kernel
		size_t block_size = 256;
		cl_mem mem =  clCreateBuffer (SystemCL::inst().Context(),
				CL_MEM_READ_WRITE, sizeof(real_t), NULL, &ret);
		CheckError(ret);

		ret = 		clSetKernelArg(SystemCL::inst().K_smpol_computeSseMaskFn(), 0, sizeof(layerSize), (void *)&layerSize);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_smpol_computeSseMaskFn(), 1, sizeof(patTypes.mem), (void *)&patTypes.mem);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_smpol_computeSseMaskFn(), 2, sizeof(targets.mem), (void *)&targets.mem);
		CheckError(ret);
		ret =        clSetKernelArg(SystemCL::inst().K_smpol_computeSseMaskFn(), 3, sizeof(actualOutputs.mem), (void *)&actualOutputs.mem);
		CheckError(ret);

		ret =        clSetKernelArg(SystemCL::inst().K_smpol_computeSseMaskFn(), 4, sizeof(mem), (void *) &mem);
		CheckError(ret);

		ret =        clSetKernelArg(SystemCL::inst().K_smpol_computeSseMaskFn(), 5,block_size * sizeof(real_t), NULL);
		CheckError(ret);

		ret =        clSetKernelArg(SystemCL::inst().K_smpol_computeSseMaskFn(), 6,  sizeof(n), (void *)&n);
		CheckError(ret);

		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_smpol_computeSseMaskFn(), 1, NULL,
				&block_size, &block_size, 0, NULL, NULL);
		CheckError(ret);


		ret = clEnqueueReadBuffer(inst().Queue(), mem, CL_TRUE, 0,
					sizeof(real_t) , &rt, 0, NULL, NULL);
		CheckError(ret);

		clReleaseMemObject(mem);
    	return rt;
    }

    static void smpol_computeOutputErrorFn(int layerSize,const d_mem_char& patTypes,d_mem_real &targets,
    		d_mem_real &actualOutputs, d_mem_real& outputErrors, int n){

    	cl_int ret ;
		ret = clSetKernelArg(SystemCL::inst().K_smpol_computeOutputErrorFn(), 0, sizeof(layerSize), (void *)&layerSize);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_smpol_computeOutputErrorFn(), 1, sizeof(patTypes.mem), (void *)&patTypes.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_smpol_computeOutputErrorFn(), 2, sizeof(targets.mem), (void *)&targets.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_smpol_computeOutputErrorFn(), 3, sizeof(actualOutputs.mem), (void *)&actualOutputs.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_smpol_computeOutputErrorFn(), 4, sizeof(outputErrors.mem), (void *)&outputErrors.mem);
		CheckError(ret);
		size_t global_item_size = n;
		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_smpol_computeOutputErrorFn(), 1, NULL,
				&global_item_size, NULL, 0, NULL, NULL);
		CheckError(ret);
    }

    static void ll_computeBlockOutputFn(int effLayerCLSize,int prevOutputDistance,
    		float bias,const d_mem_char & patTypes,
			d_mem_real & xweight,
			int niBiasWeights_offset,
			int igBiasWeights_offset,
			int fgBiasWeights_offset,
			int ogBiasWeights_offset,
			int igPeepWeights_offset,
			int fgPeepWeights_offset,
			int ogPeepWeights_offset,
			d_mem_real & cellStates,d_mem_real &niActs,d_mem_real & igActs,
			d_mem_real &fgActs,d_mem_real & ogActs, int firstCall,int checkPatType,
			d_mem_real &tmpOutputs, int  offset,int n){
    	cl_int ret ;
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 0, sizeof(effLayerCLSize), (void *)&effLayerCLSize);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 1, sizeof(prevOutputDistance), (void *)&prevOutputDistance);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 2, sizeof(bias), (void *)&bias);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 3, sizeof(xweight.mem), (void *)&xweight.mem);
		CheckError(ret);

		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 4, sizeof(niBiasWeights_offset), (void *)&niBiasWeights_offset);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 5, sizeof(igBiasWeights_offset), (void *)&igBiasWeights_offset);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 6, sizeof(fgBiasWeights_offset), (void *)&fgBiasWeights_offset);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 7, sizeof(ogBiasWeights_offset), (void *)&ogBiasWeights_offset);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 8, sizeof(igPeepWeights_offset), (void *)&igPeepWeights_offset);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 9, sizeof(fgPeepWeights_offset), (void *)&fgPeepWeights_offset);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 10, sizeof(ogPeepWeights_offset), (void *)&ogPeepWeights_offset);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 11, sizeof(patTypes.mem), (void *)&patTypes.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 12, sizeof(cellStates.mem), (void *)&cellStates.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 13, sizeof(niActs.mem), (void *)&niActs.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 14, sizeof(igActs.mem), (void *)&igActs.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 15, sizeof(fgActs.mem), (void *)&fgActs.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 16, sizeof(ogActs.mem), (void *)&ogActs.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 17, sizeof(firstCall), (void *)&firstCall);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 18, sizeof(checkPatType), (void *)&checkPatType);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 19, sizeof(tmpOutputs.mem), (void *)&tmpOutputs.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockOutputFn(), 20, sizeof(offset), (void *)&offset);
		CheckError(ret);

		size_t global_item_size = n;
		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_ll_computeBlockOutputFn(), 1, NULL,
				&global_item_size, NULL, 0, NULL, NULL);
		CheckError(ret);


    }
    static void ll_resortOutputsFn(int layerSize, int effLayerCLSize, d_mem_real& fwOutputs, d_mem_real& bwOutputs, d_mem_real& outputs, int n){
		cl_int ret;
    	ret = clSetKernelArg(SystemCL::inst().K_ll_resortOutputsFn(), 0, sizeof(layerSize), (void *)&layerSize);
		CheckError(ret);
    	ret = clSetKernelArg(SystemCL::inst().K_ll_resortOutputsFn(), 1, sizeof(effLayerCLSize), (void *)&effLayerCLSize);
		CheckError(ret);
    	ret = clSetKernelArg(SystemCL::inst().K_ll_resortOutputsFn(), 2, sizeof(fwOutputs.mem), (void *)&fwOutputs.mem);
		CheckError(ret);
    	ret = clSetKernelArg(SystemCL::inst().K_ll_resortOutputsFn(), 3, sizeof(bwOutputs.mem), (void *)&bwOutputs.mem);
		CheckError(ret);
    	ret = clSetKernelArg(SystemCL::inst().K_ll_resortOutputsFn(), 4, sizeof(outputs.mem), (void *)&outputs.mem);
		CheckError(ret);

		size_t global_item_size = n;
		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_ll_resortOutputsFn(), 1, NULL,
				&global_item_size, NULL, 0, NULL, NULL);
		CheckError(ret);

    }

    static void ll_resortOutputErrorsFn(int layerSize,int effLayerCLSize,d_mem_real &fwOutputErrors,
    		d_mem_real & bwOutputErrors,
			d_mem_real & outputErrors,int n){
		cl_int ret;
    	ret = clSetKernelArg(SystemCL::inst().K_ll_resortOutputErrorsFn(), 0, sizeof(layerSize), (void *)&layerSize);
		CheckError(ret);
    	ret = clSetKernelArg(SystemCL::inst().K_ll_resortOutputErrorsFn(), 1, sizeof(effLayerCLSize), (void *)&effLayerCLSize);
		CheckError(ret);
    	ret = clSetKernelArg(SystemCL::inst().K_ll_resortOutputErrorsFn(), 2, sizeof(fwOutputErrors.mem), (void *)&fwOutputErrors.mem);
		CheckError(ret);
    	ret = clSetKernelArg(SystemCL::inst().K_ll_resortOutputErrorsFn(), 3, sizeof(bwOutputErrors.mem), (void *)&bwOutputErrors.mem);
		CheckError(ret);
    	ret = clSetKernelArg(SystemCL::inst().K_ll_resortOutputErrorsFn(), 4, sizeof(outputErrors.mem), (void *)&outputErrors.mem);
		CheckError(ret);

		size_t global_item_size = n;
		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_ll_resortOutputErrorsFn(), 1, NULL,
				&global_item_size, NULL, 0, NULL, NULL);
		CheckError(ret);
    }

    static void ll_computeBlockErrorsFn(int effLayerCLSize,int prevOutputDistance, const d_mem_char& patTypes, d_mem_real &XWeights,
    		int igPeepWeights_Offset ,int fgPeepWeights_Offset,int ogPeepWeights_Offset,
			d_mem_real &cellStates,
			d_mem_real &niActs,
			d_mem_real &igActs,
			d_mem_real &fgActs,
			d_mem_real &ogActs,
			d_mem_real &cellStateErrors,
			d_mem_real &niDeltas,
			d_mem_real &igDeltas,
			d_mem_real &fgDeltas,
			d_mem_real &ogDeltas,
			d_mem_real &tmpOutputErrors, int offset, int firstCall,int lastCall,int checkPatType,int n){

       	cl_int ret ;
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 0, sizeof(effLayerCLSize), (void *)&effLayerCLSize);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 1, sizeof(prevOutputDistance), (void *)&prevOutputDistance);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 2, sizeof(patTypes.mem), (void *)&patTypes.mem);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 3, sizeof(igPeepWeights_Offset), (void *)&igPeepWeights_Offset);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 4, sizeof(fgPeepWeights_Offset), (void *)&fgPeepWeights_Offset);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 5, sizeof(ogPeepWeights_Offset), (void *)&ogPeepWeights_Offset);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 6, sizeof(XWeights.mem), (void *)&XWeights.mem);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 7, sizeof(cellStates.mem), (void *)&cellStates.mem);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 8, sizeof(niActs.mem), (void *)&niActs.mem);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 9, sizeof(igActs.mem), (void *)&igActs.mem);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 10, sizeof(fgActs.mem), (void *)&fgActs.mem);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 11, sizeof(ogActs.mem), (void *)&ogActs.mem);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 12, sizeof(cellStateErrors.mem), (void *)&cellStateErrors.mem);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 13, sizeof(niDeltas.mem), (void *)&niDeltas.mem);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 14, sizeof(igDeltas.mem), (void *)&igDeltas.mem);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 15, sizeof(fgDeltas.mem), (void *)&fgDeltas.mem);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 16, sizeof(ogDeltas.mem), (void *)&ogDeltas.mem);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 17, sizeof(tmpOutputErrors.mem), (void *)&tmpOutputErrors.mem);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 18, sizeof(offset), (void *)&offset);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 19, sizeof(firstCall), (void *)&firstCall);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 20, sizeof(lastCall), (void *)&lastCall);
    		CheckError(ret);
    		ret = clSetKernelArg(SystemCL::inst().K_ll_computeBlockErrorsFn(), 21, sizeof(checkPatType), (void *)&checkPatType);
    		CheckError(ret);


    		size_t global_item_size = n;
    		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_ll_computeBlockErrorsFn(), 1, NULL,
    				&global_item_size, NULL, 0, NULL, NULL);
    		CheckError(ret);

    }

    static void ll_computeWeightUpdateFn(int layerSize,int effLayerCLSize, int precLayerCLSize,int timestepDistance,
			int parallelSequences,int  patternsCount,int biasWeightsOffset,int internalWeightsOffset,int peepholeWeightsOffset,
			float bias,
			d_mem_real & ploutputs, d_mem_real &fwOutputs, d_mem_real &bwOutputs,
			d_mem_real &fwcellStates, d_mem_real& bwcellStates,
			d_mem_real &fwniDeltas, d_mem_real& bwniDeltas,
			d_mem_real &fwigDeltas, d_mem_real& bwigDeltas,
			d_mem_real &fwfgDeltas, d_mem_real& bwfgDeltas,
			d_mem_real &fwogDeltas, d_mem_real& bwogDeltas,
			d_mem_real &Output, int n){
    	int ret;
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 0, sizeof(layerSize), (void *)&layerSize);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 1, sizeof(effLayerCLSize), (void *)&effLayerCLSize);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 2, sizeof(precLayerCLSize), (void *)&precLayerCLSize);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 3, sizeof(timestepDistance), (void *)&timestepDistance);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 4, sizeof(parallelSequences), (void *)&parallelSequences);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 5, sizeof(patternsCount), (void *)&patternsCount);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 6, sizeof(biasWeightsOffset), (void *)&biasWeightsOffset);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 7, sizeof(internalWeightsOffset), (void *)&internalWeightsOffset);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 8, sizeof(peepholeWeightsOffset), (void *)&peepholeWeightsOffset);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 9, sizeof(bias), (void *)&bias);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 10, sizeof(ploutputs.mem), (void *)&ploutputs.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 11, sizeof(fwOutputs.mem), (void *)&fwOutputs.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 12, sizeof(bwOutputs.mem), (void *)&bwOutputs.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 13, sizeof(fwcellStates.mem), (void *)&fwcellStates.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 14, sizeof(bwcellStates.mem), (void *)&bwcellStates.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 15, sizeof(fwniDeltas.mem), (void *)&fwniDeltas.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 16, sizeof(bwniDeltas.mem), (void *)&bwniDeltas.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 17, sizeof(fwigDeltas.mem), (void *)&fwigDeltas.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 18, sizeof(bwigDeltas.mem), (void *)&bwigDeltas.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 19, sizeof(fwfgDeltas.mem), (void *)&fwfgDeltas.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 20, sizeof(bwfgDeltas.mem), (void *)&bwfgDeltas.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 21, sizeof(fwogDeltas.mem), (void *)&fwogDeltas.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 22, sizeof(bwogDeltas.mem), (void *)&bwogDeltas.mem);
		CheckError(ret);
		ret = clSetKernelArg(SystemCL::inst().K_ll_computeWeightUpdateFn(), 23, sizeof(Output.mem), (void *)&Output.mem);
		CheckError(ret);
		size_t global_item_size = n;
		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_ll_computeWeightUpdateFn(), 1, NULL,
				&global_item_size, NULL, 0, NULL, NULL);
		CheckError(ret);


    }

    static void sdo_updateWeightFn(float momentum, float learningRate,d_mem_real &weights,
            		const d_mem_real& weightUpdates, d_mem_real& weightDeltas, d_mem_real& output, int n){
   		int ret;
    	ret = clSetKernelArg(SystemCL::inst().K_sdo_updateWeightFn(), 0, sizeof(learningRate), (void *)&learningRate);
    	CheckError(ret);
        ret = clSetKernelArg(SystemCL::inst().K_sdo_updateWeightFn(), 1, sizeof(momentum), (void *)&momentum);
        CheckError(ret);
        ret = clSetKernelArg(SystemCL::inst().K_sdo_updateWeightFn(), 2, sizeof(weights.mem), (void *)&weights.mem);
        CheckError(ret);
        ret = clSetKernelArg(SystemCL::inst().K_sdo_updateWeightFn(), 3, sizeof(weightUpdates.mem), (void *)&weightUpdates.mem);
        CheckError(ret);
        ret = clSetKernelArg(SystemCL::inst().K_sdo_updateWeightFn(), 4, sizeof(weightDeltas.mem), (void *)&weightDeltas.mem);
        CheckError(ret);
        ret = clSetKernelArg(SystemCL::inst().K_sdo_updateWeightFn(), 5, sizeof(output.mem), (void *)&output.mem);
        CheckError(ret);
		size_t global_item_size = n;
		ret = clEnqueueNDRangeKernel(SystemCL::inst().Queue(), SystemCL::inst().K_sdo_updateWeightFn(), 1, NULL,
				&global_item_size, NULL, 0, NULL, NULL);
		CheckError(ret);
    }
    static void print(const d_mem_real& d){
//    	if (d.size < 1) return;
//		float * rt = new float[d.size];
//		int ret = clEnqueueReadBuffer(inst().Queue(), d.mem, CL_TRUE, 0,
//					sizeof(real_t) * d.size , rt, 0, NULL, NULL);
//		int size = d.size > 50 ? 50: d.size;
//		for (int i = 0; i < size ; i ++)
//			std::cout << rt[i] << " ";
//		std::cout << std::endl;
//
//		CheckError(ret);
    }
    static void print(const d_mem_char& d){
//    	if (d.size < 1) return;
//		char * rt = new char[d.size];
//		int ret = clEnqueueReadBuffer(inst().Queue(), d.mem, CL_TRUE, 0,
//					sizeof(char) * d.size , rt, 0, NULL, NULL);
//		int size = d.size > 50 ? 50: d.size;
//		for (int i = 0; i < size ; i ++)
//			std::cout << rt[i] << " ";
//		std::cout << std::endl;
//
//		CheckError(ret);
    }

    static void print(d_mem_real& d, int offset){
//    	if (d.size < 1) return;
//		float * rt = new float[d.size];
//		int ret = clEnqueueReadBuffer(inst().Queue(), d.mem, CL_TRUE, 0,
//					sizeof(real_t) * d.size , rt, 0, NULL, NULL);
//		int size = d.size > 50 + offset ? 50+offset: d.size;
//		for (int i = offset; i < size ; i ++)
//			std::cout << rt[i] << " ";
//		std::cout << std::endl;
//
//		CheckError(ret);
    }
    static void print(std::vector<float> d){
//    	if (d.size() < 1) return;
//
//		for (int i = 0; i < d.size() ; i ++)
//			std::cout << d[i] << " ";
//		std::cout << std::endl;

    }
    static void CheckError (cl_int error)
	{
		if (error != CL_SUCCESS) {
			std::cerr << "OpenCL call failed with error " << error << std::endl;
			std::exit (1);
		}

	}


	static SystemCL& inst()
	{
	  static SystemCL INSTANCE;
	  return INSTANCE;
	}



};




#endif /* CURRENNT_LIB_SRC_SYSTEMCL_HPP_ */
