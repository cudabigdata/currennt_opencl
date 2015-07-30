// Define limit for numbers
#define NL_min     1.1754944e-038f
#define NL_max   3.4028235e+038f
#define NL_expLimit 88.722839f
#define NL_logInf   1e30
#define NL_logZero  -1e30


__kernel void memset_float(__global float* mem, float val) {
    mem[get_global_id(0)]=val; 
}

__kernel void mm_tranpose_AFn(int rowsA, int rowsB, int colsA, int colsB,
			                      int a_offset, int b_offset,
								__global float * a, __global float * b, 
								__global float * c, int offset){
	int idx = get_global_id(0);
	
	__global float *offColA = (a + a_offset) + (idx % colsA) * rowsA;
	__global float *offColB = (b + b_offset) + (idx / colsA) * rowsB;

	 float x = 0;
	 for (int i = 0; i < rowsA; ++i)
		 x += offColA[i] * offColB[i];
	 c[idx + offset] = x;
}

__kernel void mm_tranpose_Fn(int rowsA, int rowsB, int colsA, int colsB,
									int A_offset, int B_offset,
								__global float * a, __global float * b, 
								__global float * c, int offset){
	int idx = get_global_id(0);
	
	__global float *offRowA = (a + A_offset) + (idx % rowsA);
	__global float *offColB = (b + B_offset) + (idx / rowsA) * rowsB;

	float x = 0;
	for (int i = 0; i < colsA; ++i)
		x += offRowA[i * rowsA] * offColB[i];
	 c[idx + offset] = x;
}

__kernel void mm_tranpose_BFn(int rowsA, int rowsB, int colsA, int colsB,
		                                 int A_offset, int B_offset,
								__global float * a, __global float * b, 
								__global float * c, int offset){
	int idx = get_global_id(0);
	
	__global float *offRowA = (a + A_offset) + (idx % rowsA);
	__global float *offRowB = (b + B_offset) + (idx / rowsA);

	float x = 0;
	for (int i = 0; i < colsA; ++i) {
		x += *offRowA * *offRowB;
		offRowA += rowsA;
		offRowB += rowsB;
	}
	 c[idx + offset] = x;
}

__kernel void mma_tranpose_AFn(int rowsA, int rowsB, int colsA, int colsB,
		                           int A_offset, int B_offset,
								__global float * a, __global float * b, 
								__global float * c, int offset){
	int idx = get_global_id(0);
	
	__global float *offColA = (a + A_offset) + (idx % colsA) * rowsA;
	__global float *offColB = (b + B_offset) + (idx / colsA) * rowsB;

	float x = 0;
	for (int i = 0; i < rowsA; ++i)
		x += offColA[i] * offColB[i];
	
	 c[idx + offset] += x;
}

__kernel void mma_tranpose_BFn(int rowsA, int rowsB, int colsA, int colsB,
		                         int A_offset, int B_offset,
								__global float * a, __global float * b, 
								__global float * c, int offset){
	int idx = get_global_id(0);
	
	__global float *offRowA = (a + A_offset) + (idx % rowsA);
	__global float *offRowB = (b + B_offset) + (idx / rowsA);
	
	float x = 0;
	for (int i = 0; i < colsA; ++i) {
		x += *offRowA * *offRowB;
		offRowA += rowsA;
		offRowB += rowsB;
	}
	 c[idx + offset] += x;
}
__kernel void mma_tranpose_Fn(int rowsA, int rowsB, int colsA, int colsB,
		                        int A_offset, int B_offset,
								__global float * a, __global float * b, 
								__global float * c, int offset){
	int idx = get_global_id(0);
	
	__global float *offRowA = (a + A_offset) + (idx % rowsA);
	__global float *offColB = (b + B_offset) + (idx / rowsA) * rowsB;
	
	float x = 0;
	for (int i = 0; i < colsA; ++i)
	   x += offRowA[i * rowsA] * offColB[i];
	
	 c[idx + offset] += x;
}

__kernel void r_plus(__global float * a, __global float * b){
	int idx = get_global_id(0);
	a[idx] +=b[idx];
}

#define PATTYPE_NONE   0 ///< pattern does not belong to the sequence
#define PATTYPE_FIRST  1 ///< first pattern/timestep in the sequence
#define PATTYPE_NORMAL 2 ///< pattern/timestep with a sequence (not first/last)
#define PATTYPE_LAST   3 ///< last pattern/timestep in the sequence

__kernel void compute_weightedSSeFn(int layerSize, __global char * patTypes,
					__global float * targets, __global float * outputs,
					__local float* shared, __global float * out, int n)
{
	int index = get_local_id(0);
	shared[get_local_id(0)] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	while ( index < n){

		int patIdx = index / layerSize;
		if (patTypes[patIdx] != PATTYPE_NONE)
		{
			float target = targets[index * 2];
			float output = outputs[index];
			float weight = targets[index * 2 + 1];
		
			// calculate the error
			float diff = (output - target) * weight;
			shared[get_local_id(0)] += (diff * diff);
		}
		index += get_local_size(0);
		
	}
	index = get_local_id(0);
	
 	barrier(CLK_LOCAL_MEM_FENCE);
 	for (unsigned int s = get_local_size(0)/2; s>0; s>>=1) {
		if(index < s) {
			shared[index] += shared[index + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
 	}
 	
 	if (get_local_id(0) == 0){
 		out[0] = shared[0];
 	}
}

__kernel void compute_OutputErrorFn(int layerSize, __global char * patTypes,
		__global float * targets, __global float * outputs, __global float * outputError){
	int index = get_global_id(0);
	float target = targets[index * 2];
	float output = outputs[index];
	float weight = targets[index * 2 + 1];

	// calculate the pattern index
	int patIdx = index / layerSize;

	// check if the pattern is a dummy
	if (patTypes[patIdx] == PATTYPE_NONE)
		outputError[index] = 0;
	else
		outputError[index] = (output - target) * weight;
}
///////////////////////////////////



float boundRange(float x, float lowerLimit, float upperLimit)
{
    return (x < lowerLimit ? lowerLimit : (x > upperLimit ? upperLimit : x));
}


float Logistic_fn(float x)
{
    if (x < NL_expLimit) {
        if (x > -NL_expLimit)
            return (float)1.0 / ((float)1.0 + exp(-x));
        else
            return 0;
    }
    
    return 1;
}


float Maxmin1_fn(float x)
{
    return ((float)2.0 * Logistic_fn(x) - (float)1.0);
}

float Tanh_fn(float x)
{
    return Maxmin1_fn((float)2.0 * x);
}

float Identity_fn(float x)
{
     return x;
}

#define TANH 0
#define LOGISTIC 1
#define IDENTITY 2


__kernel void ffl_computeOutputFn(int layerSize, float bias, __global float * biasWeights , int biasOffset,
						__global float * aa, int typeFunction){
		// calculate indices
	    int outputIdx = get_global_id(0);
	    
		int blockIdx = outputIdx % layerSize;
		float a = aa[outputIdx];

		// add the bias
		a += bias * biasWeights[biasOffset + blockIdx];

		// apply the activation function
		float b;
		if (typeFunction ==  TANH)
			b = Tanh_fn(a);
		else if(typeFunction == LOGISTIC)
			b = Logistic_fn(a);
		else 
			b = Identity_fn(a);
		aa[outputIdx] = b;
}

float Tanh_deriv(float y)
{
      return (float)1.0 - (y * y);
}
float Logistic_deriv(float y)
{
     return y * ((float)1.0 - y);
}
float Identity_deriv(float y)
{
    return 1;
}
__kernel void ffl_computeDeltaFn(__global float * t0, __global float * t1, int typeAF){
	
	int index = get_global_id(0);
	float d;
	if ( typeAF == TANH )
		d = Tanh_deriv(t1[index]);
	else if (typeAF == LOGISTIC)
		d = Logistic_deriv(t1[index]);
	else
		d = Identity_deriv(t1[index]);
    
    t0[index] = d * t0[index];
	
}

__kernel void ffl_computeBiasWeightUpdateFn(int layerSize, int patternsCount, float bias,
		__global float * deltas, __global float * out, int offset){
	
	int biasWeightIdx = get_global_id(0);
	__global float *offDeltas = deltas + biasWeightIdx;

	float wu = 0;
	for (int i = 0; i < patternsCount; ++i) {
		wu += bias * *offDeltas;
		offDeltas += layerSize;
	}

	out[biasWeightIdx + offset] =  wu;
}



__kernel void spol_computeSseFn(int layerSize, __global char * patTypes, __global float * targets,
		         __global float * actualOutput, __global float * out, __local float * shared, int n){
	
	int index = get_local_id(0);
	shared[get_local_id(0)] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	while (index < n){
		float target = targets[index];
		float output = actualOutput[index];

	
		// check if we have to skip this value
		int patIdx = index / layerSize;
		if (patTypes[patIdx] !=PATTYPE_NONE)
		{
			// calculate the error
			float diff = target - output;
			shared[get_local_id(0)] += (diff * diff);
		}
		
		index += get_local_size(0);
	}
	
	index = get_local_id(0);
	
 	barrier(CLK_LOCAL_MEM_FENCE);
 	for (unsigned int s = get_local_size(0)/2; s>0; s>>=1) {
		if(index < s) {
			shared[index] += shared[index + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
 	}
 	
 	if (get_local_id(0) == 0){
 		out[0] = shared[0];
 	}
}

__kernel void spol_computeOutputErrorFn(int layerSize, __global char* patTypes, 
		    __global float * actualoutput, __global float * targets,
			__global float * outputErrors){
	
	int outputIdx = get_global_id(0);
	
	float actualOutput = actualoutput[outputIdx];
	float targetOutput = targets[outputIdx];


	// calculate the pattern index
	int patIdx = outputIdx / layerSize;

	// check if the pattern is a dummy
	if (patTypes[patIdx] == PATTYPE_NONE)
		outputErrors[outputIdx] = 0;
	else
		outputErrors[outputIdx] = actualOutput - targetOutput;

	
}

__kernel void bcl_countCorrectClassificationsFn(__global float * targets, __global float * outputs,
						__global char * patTypes, __global int * out, __local int * shared, int n){ 
	
	int index = get_local_id(0);
	shared[get_local_id(0)] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	while (index < n){
		float target  = targets[index];
		float output  = outputs[index];
		int    patType = patTypes[index];

		// determine target and estimated class
		bool tgtClass = (target > (float)0.5);
		bool estClass = (output > (float)0.5);

		// count correct classification
		shared[get_local_id(0)] += (patType != PATTYPE_NONE) && (tgtClass == estClass);
		
		index += get_local_size(0);
	}
	
	index = get_local_id(0);
	
 	barrier(CLK_LOCAL_MEM_FENCE);
 	for (unsigned int s = get_local_size(0)/2; s>0; s>>=1) {
		if(index < s) {
			shared[index] += shared[index + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
 	}
 	
 	if (get_local_id(0) == 0){
 		out[0] = shared[0];
 	}
}


__kernel void bcl_computeCrossEntropyErrorFn(__global char * patTypes, __global float * targets, 
					__global float * outputs, __global float * out, __local float * shared, int n){
	int index = get_local_id(0);
	shared[get_local_id(0)] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	while ( index < n){
		// check if we actually need to continue
		if (patTypes[index] != PATTYPE_NONE)
		{
			// calculate the cross entropy error
			float target = targets[index];
			float output = outputs[index];
		
			float act        = max(output, NL_min);
			float targetProb = (target > 0 ? act : 1-act);
			float error      = -log(targetProb);
		
			shared[get_local_id(0)]+= error;
		}
		index += get_local_size(0);
		
	}
	index = get_local_id(0);
	
 	barrier(CLK_LOCAL_MEM_FENCE);
 	for (unsigned int s = get_local_size(0)/2; s>0; s>>=1) {
		if(index < s) {
			shared[index] += shared[index + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
 	}
 	
 	if (get_local_id(0) == 0){
 		out[0] = shared[0];
 	}
}

__kernel void bcl_computeOutputErrorFn(__global char * patTypes, __global float * targets,
				__global float * erroroutputs,  __global float * actualoutput){
	
		// unpack the tuple
		int outputIdx =  get_global_id(0);

		// check if we actually need to continue
		if (patTypes[outputIdx] != PATTYPE_NONE)
		{
			// calculate the error
			float target = targets[outputIdx];
			float output = actualoutput[outputIdx];
	
			float act        = max(output, NL_min);
			float targetProb = (target > 0 ? act : 1-act);
			float error      = (target > 0 ? -(1/targetProb) : (1/targetProb));
	
			// store the error
			erroroutputs[outputIdx] = error;
		}
}

__kernel void cpol_computeCeFn(int layerSize, __global char * patTypes, __global float * targets,
		  __global float * outputs, __global float * out, __local float * shared, int n){
	int index = get_local_id(0);
	shared[get_local_id(0)] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	while (index < n){
		float target = targets[index];
		float output = outputs[index];
		
		// check if we have to skip this value
		int patIdx = index / layerSize;
		if (patTypes[index] != PATTYPE_NONE)
		{
			float ftarget = max(NL_min, target);
			output = max(NL_min, output);
			float div = target * log(ftarget / output);
			shared[get_local_id(0)] += div;
		}
		
		index += get_local_size(0);
	}
	
	index = get_local_id(0);
	
 	barrier(CLK_LOCAL_MEM_FENCE);
 	for (unsigned int s = get_local_size(0)/2; s>0; s>>=1) {
		if(index < s) {
			shared[index] += shared[index + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
 	}
 	
 	if (get_local_id(0) == 0){
 		out[0] = shared[0];
 	}
	
}

__kernel void cpol_computeOutputErrorFn(int layerSize, __global char * patTypes, __global float * actualOutputs,
		 __global float * targets, __global float * outputError){
	
		int    outputIdx    = get_global_id(0);
		float actualOutput = actualOutputs[outputIdx];
		float targetOutput = targets[outputIdx];
	
		
		// calculate the pattern index
		int patIdx = outputIdx / layerSize;
		
		// check if the pattern is a dummy
		if (patTypes[patIdx] == PATTYPE_NONE)
			outputError[outputIdx]= 0;
		else{
			actualOutput = max(NL_min, actualOutput);
			outputError[outputIdx] = boundRange(-targetOutput / actualOutput, -100, +100);
		}
		
	
}

__kernel void mcl_countCorrectClassificationsFn(int layerSize, __global float * outputs, 
		        __global int * targetClasses, __global int * out, __local int * shared, int n){
	
	int index = get_local_id(0);
	shared[get_local_id(0)] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	while ( index < n){
		int targetClass = targetClasses[index];

		// check for dummy
		if (targetClass != -1)
		{
			// determine the estimated target class
			__global float *offOutputs = outputs + index * layerSize;
			float maxProb = 0;
			int estClass   = 0;
		
			for (int i = 0; i < layerSize; ++i) {
				float outv = offOutputs[i];
				if (outv > maxProb) {
					maxProb  = outv;
					estClass = i;
				}
			}
		
			// check if the we correctly classified the timestep
			if (targetClass == estClass)
				shared[get_local_id(0)] += 1;
		}
		index += get_local_size(0);
		
	}
	index = get_local_id(0);
	
 	barrier(CLK_LOCAL_MEM_FENCE);
 	for (unsigned int s = get_local_size(0)/2; s>0; s>>=1) {
		if(index < s) {
			shared[index] += shared[index + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
 	}
 	
 	if (get_local_id(0) == 0){
 		out[0] = shared[0];
 	}
}
__kernel void mcl_computeCrossEntropyErrorFn(int layerSize, __global float * outputs,
		    __global int * targetClasses, __global float * out, __local float * shared, int n){
	int index = get_local_id(0);
	shared[get_local_id(0)] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	while ( index < n){
		int targetClass = targetClasses[index];

		// calculate the CEE
		if (targetClass != -1)
		{
			int outputIdx  = index * layerSize + targetClass;
			float targetProb = max(NL_min, outputs[outputIdx]);
			shared[get_local_id(0)] += log(targetProb);
		}
		index += get_local_size(0);
		
	}
	index = get_local_id(0);
	
 	barrier(CLK_LOCAL_MEM_FENCE);
 	for (unsigned int s = get_local_size(0)/2; s>0; s>>=1) {
		if(index < s) {
			shared[index] += shared[index + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
 	}
 	
 	if (get_local_id(0) == 0){
 		out[0] = shared[0];
 	}
}
__kernel void mcl_computeOutputErrorFn(int layerSize, __global float * outputs,
		       __global float * outputErrors, __global int * targetClasses){
	int patIdx = get_global_id(0);
	
	int targetClass = targetClasses[patIdx];
	
	// check if we need to continue
	if (targetClass == -1)
		return;
	
	// calculate indices
	int outputIdx = patIdx * layerSize + targetClass;
	
	// calculate the error
	float targetProb = max(NL_min, outputs[outputIdx]);
	float error = - (1/targetProb);
	
	// store the error
	outputErrors[outputIdx] = error;
}

__kernel void rpol_calculateError(__global float * m_rmses, __global float * out, 
						 __local float *  shared, int n){
	int index = get_local_id(0);
	shared[index] = 0;
 	barrier(CLK_LOCAL_MEM_FENCE);
	while ( index < n){
	
		shared[get_local_id(0)] += m_rmses[index];		
		index += get_local_size(0);	
	}
 	for (unsigned int s = get_local_size(0)/2; s>0; s>>=1) {
		if(index < s) {
			shared[index] += shared[index + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
 	}
 	
 	if (get_local_id(0) == 0){
 		out[0] = shared[0];
 	}
}

__kernel void rpol_computeRmseFn(int layerSize, __global float * actualOutputs,
		   __global float * targetOutputs, __global char * patTypes, __global float * m_rmses){
	
	int patIdx = get_global_id(0);
	if (patTypes[patIdx] == PATTYPE_NONE)
		m_rmses[patIdx] = 0;
	else{

		// offset arrays to the beginning of the pattern
		__global float *offActualOutputs = &actualOutputs[patIdx * layerSize];
		__global float *offTargetOutputs = &targetOutputs[patIdx * layerSize];
	
		// sum up the squared errors
		float sum = 0;
		for (int i = 0; i < layerSize; ++i) {
			float diff = offActualOutputs[i] - offTargetOutputs[i];
			sum += diff * diff;
		}
	
		// calculate the rmse
		m_rmses[patIdx] = sqrt(sum / layerSize);
	}
}

__kernel void rpol_computeOutputErrorFn(int layerSize, __global float * actualOuputs,
		__global float * targetOutputs, __global float * outputError, __global float * rmses){
	
	    int outputIdx =  get_global_id(0);
		float actualOutput = actualOuputs[outputIdx];
		float  targetOutput = targetOutputs[outputIdx];

		// calculate the pattern index
		int patIdx = outputIdx / layerSize;

		// get the RMSE for the current pattern
		float rmse = rmses[patIdx];

		// calculate the error
		outputError[outputIdx] = rmse * (actualOutput - targetOutput);

}
__kernel void sml_calculateOffsetFn(int layerSize, __global float * outputs, 
		__global char * patTypes, __global float * m_patTmp){
	
	int patIdx = get_global_id(0);
	if (patTypes[patIdx] == PATTYPE_NONE)
		m_patTmp[patIdx] =  NL_max;
	else
	{
		// search for the min and max output
		float maxx = NL_min;
		float minx = NL_max;
	
		__global float *offOutputs = &outputs[patIdx * layerSize];
	
		for (int i = 0; i < layerSize; ++i) {
			float x = offOutputs[i];
			minx = min(minx, x);
			maxx = max(maxx, x);
		}
	
		// calculate the offset
		m_patTmp[patIdx] = (float)0.5 * (minx + maxx);
	
	}
}


float safeExp(float x)
 {
     if (x <= NL_logZero)
         return 0;
     else if (x >= NL_expLimit)
         return NL_max;
     else
         return exp(x);
 }

__kernel void sml_calculateExpFn(int layerSize, __global float * offsets,
		__global float * outputs){
	int outputId = get_global_id(0);
	float output = outputs[outputId];
	
	// calculate the pattern index
	int patIdx = outputId / layerSize;

	// check if we can stop the calculation
	float offset = offsets[patIdx];
	if (offset != NL_max)
	{
		// calculate the exponent
		float x = safeExp(output - offset);
		// store the result
		outputs[outputId] = x;
	}
}

__kernel void sml_sumUpOutputsFn(int layerSize, __global float * outputs, 
		 __global float * m_patTmp){
    int patIdx = get_global_id(0);

	// check if the pattern belongs to a sequence
	if (m_patTmp[patIdx] != NL_max)
	{
		// sum up the outputs
		__global float *offOutputs = &outputs[patIdx * layerSize];
	
		float sum = 0;
		for (int i = 0; i < layerSize; ++i)
			sum += offOutputs[i];
	
		// store the result
		m_patTmp[patIdx] = sum;
	}
}

__kernel void sml_normalizeOutputsFn(int layerSize, __global float * normFacts,
		 __global float * outputs){
    int outputIdx = get_global_id(0);

	// calculate the pattern index
	int patIdx = outputIdx / layerSize;

	// check if we can stop the calculation
	float normFact = normFacts[patIdx];
	if (normFact != NL_max)
	{
	// calculate the normalized value
		float x = outputs[outputIdx] / normFact;
	
		// store the result
		outputs[outputIdx] = x;
	}
}

__kernel void sml_calculateErrorOffsetFn(int layerSize, __global float * outputs,
		__global float * outputErrors, __global char * patTypes, __global float * m_patTmp){
	int patIdx = get_global_id(0);
	
	if (patTypes[patIdx] == PATTYPE_NONE)
		m_patTmp[patIdx] =  NL_max;
	else {
		// calculate the offset
		__global float *offOutputs      = &outputs     [patIdx * layerSize];
		__global float *offOutputErrors = &outputErrors[patIdx * layerSize];
	
		float offset = 0;
		for (int i = 0; i < layerSize; ++i)
			offset += offOutputs[i] * offOutputErrors[i];
	
		m_patTmp[patIdx]=  offset;
	}
}
__kernel void sml_calculateErrorsFn(int layerSize, __global float * errorOffsets,
		__global float * errors, __global float * outputs){
	int outputIdx = get_global_id(0);

	// calculate the pattern index
	int patIdx = outputIdx / layerSize;

	// check if we can stop the calculation
	float offset = errorOffsets[patIdx];
	if (offset != NL_max)
	{
	
		// calculate the delta
		float error  = errors[outputIdx];
		float output = outputs[outputIdx];
	
		float x = output * (error - offset);
	
		// store the result
		errors[outputIdx] = x;
	}
}

__kernel void smpol_computeSseMaskFn(int layerSize, __global char * patTypes,
		__global float * targets, __global float * outputs, __global float * out,
		__local float * shared , int n){
	int index = get_local_id(0);
	shared[get_local_id(0)] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	while ( index < n){
		float target = targets[index * 2];
		float actualFilter = outputs[index];
		float filterInput = targets[index * 2 + 1];

		// check if we have to skip this value
		int patIdx = index / layerSize;
		if (patTypes[patIdx] != PATTYPE_NONE)
		{

		// calculate the error
			float diff = actualFilter * filterInput - target;
			shared[get_local_id(0)]+= (diff * diff);
		}
		
		index += get_local_size(0);
		
	}
	index = get_local_id(0);
	
 	barrier(CLK_LOCAL_MEM_FENCE);
 	for (unsigned int s = get_local_size(0)/2; s>0; s>>=1) {
		if(index < s) {
			shared[index] += shared[index + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
 	}
 	
 	if (get_local_id(0) == 0){
 		out[0] = shared[0];
 	}
}

__kernel void smpol_computeOutputErrorFn(int layerSize, __global char * patTypes,
		__global float * targets, __global float * outputs, __global float * outputErrors){
	
	int index = get_global_id(0);
	
	float target = targets[index * 2];
	float actualFilter = outputs[index];
	float filterInput = targets[index * 2 + 1];

	// calculate the pattern index
	int patIdx = index / layerSize;

	// check if the pattern is a dummy
	if (patTypes[patIdx] == PATTYPE_NONE)
		outputErrors[index] =  0;
	else
		outputErrors[index] =  (actualFilter * filterInput - target) * filterInput;

}



__kernel void ll_computeBlockOutputFn(int effLayerCLSize, int prevOutputDistance,
		float bias, __global float * Xweights,
		int niBias_offset,
		int igBias_offset,
		int fgBias_offset,
		int ogBias_offset,
		int igPeep_offset,
		int fgPeep_offset,
		int ogPeep_offset,
		__global char * patTypes, __global float * cellStates,
		__global float * niActs, __global float * igActs, __global float * fgActs,
		   __global float * ogActs, int firstCall, int checkPatType, 
		   __global float * outputs, int output_offset){
	
	__global float *niBiasWeights = &Xweights[niBias_offset] ;
	__global float *igBiasWeights = &Xweights[igBias_offset] ;
	__global float *fgBiasWeights = &Xweights[fgBias_offset];
	__global float *ogBiasWeights = &Xweights[ogBias_offset];

	__global float *igPeepWeights= &Xweights[igPeep_offset];
	__global float *fgPeepWeights= &Xweights[fgPeep_offset];
	__global float *ogPeepWeights= &Xweights[ogPeep_offset];
	int outputIdx = get_global_id(0) + output_offset;
	
//	outputs[outputIdx] = bias; //patTypes[outputIdx];
//	return;

	if (checkPatType == 1) {
		int patIdx = outputIdx / effLayerCLSize;
		if (patTypes[patIdx] == PATTYPE_NONE) {
			if (prevOutputDistance > 0)
				cellStates[outputIdx] = 0;
			
			outputs[outputIdx] = 0;
			return ;
		}
	}

	// calculate indices
	int blockIdx = outputIdx % effLayerCLSize;

	// load the niag activations
	float niAct = niActs[outputIdx];
	float igAct = igActs[outputIdx];
	float fgAct = fgActs[outputIdx];
	float ogAct = ogActs[outputIdx];

	// add bias activations
	niAct += bias * niBiasWeights[blockIdx];
	igAct += bias * igBiasWeights[blockIdx];
	fgAct += bias * fgBiasWeights[blockIdx];
	ogAct += bias * ogBiasWeights[blockIdx];

	// add activation from peephole weights
	if (!(firstCall == 1)) {
		float prevCellState = cellStates[outputIdx + prevOutputDistance];

		igAct += prevCellState * igPeepWeights[blockIdx];
		fgAct += prevCellState * fgPeepWeights[blockIdx];
	}

	// apply the activation functions
	niAct = Tanh_fn(niAct);
	igAct = Logistic_fn(igAct);
	fgAct = Logistic_fn(fgAct);

	// store the niag activations
	niActs[outputIdx] = niAct;
	igActs[outputIdx] = igAct;
	fgActs[outputIdx] = fgAct;

	// calculate the cell state and store the result
	float cellState = niAct * igAct;

	if (!(firstCall == 1))
		cellState += cellStates[outputIdx + prevOutputDistance] * fgAct;

	cellStates[outputIdx] = cellState;

	// calculate the output gate activation and store the result
	ogAct += cellState * ogPeepWeights[blockIdx];
	ogAct = Logistic_fn(ogAct);
	ogActs[outputIdx] = ogAct;

	// calculate the block output
	 outputs[outputIdx] = Tanh_fn(cellState) * ogAct;
}
__kernel void ll_resortOutputsFn(int layerSize, int effLayerCLSize, 
		__global float * fwOutputs, __global float * bwOutputs, __global float * output
		){
	int outputIdx = get_global_id(0);
	int patIdx = outputIdx / layerSize;
	int valIdx = outputIdx % layerSize;
	int offset = patIdx * effLayerCLSize + valIdx;

	// store the value
	if (valIdx < effLayerCLSize)
		output[outputIdx] =  fwOutputs[offset];
	else
		output[outputIdx] =  bwOutputs[offset - effLayerCLSize];
	
}

__kernel void ll_resortOutputErrorsFn(int layerSize, int effLayerCLSize, __global float * fwOutputErrors,
		 __global float * bwOutputErrors,
		 __global float * outputErrs){
	 int    outputIdx = get_global_id(0);
	 float outputErr = outputErrs[outputIdx];


	 // calculate indices
	 int patIdx = outputIdx / layerSize;
	 int valIdx = outputIdx % layerSize;
	 int offset = patIdx * effLayerCLSize + valIdx;

	 // store the value
	 if (valIdx < effLayerCLSize)
		 fwOutputErrors[offset] = outputErr;
	 else
		 bwOutputErrors[offset - effLayerCLSize] = outputErr;
}
		 

float limitedError(float error)
{
        return boundRange(error, -1.0, +1.0);
}

__kernel void ll_computeBlockErrorsFn(int effLayerCLSize, int prevOutputDistance,
		__global char * patTypes, int igPeepWeights_offset, 
		   int fgPeepWeights_offset, int ogPeepWeights_offset, __global const float* Xweight, 
		   __global const float * cellStates, __global const float * niActs,
		   __global const float * igActs, __global const float * fgActs, __global const float * ogActs,
		   __global float * cellStateErrors,
		   __global float * niDeltas, __global float * igDeltas,
		   __global float * fgDeltas, __global float * ogDeltas, 
		   __global float * outputErrs,int offset,  int firstCall, int lastCall, int checkPatType){
	
	 __global float *igPeepWeights = &Xweight[igPeepWeights_offset];
	 __global float *fgPeepWeights = &Xweight[fgPeepWeights_offset];
	 __global float *ogPeepWeights = &Xweight[ogPeepWeights_offset];
	 

	 int    outputIdx    = get_global_id(0) + offset;
	 float  outputErr    = outputErrs[outputIdx];


	 // check if we can skip the whole calculation because the pattern is a dummy
	 // in that case, we set all values of that pattern to zero
	 if (checkPatType == 1) {
		 int patIdx = outputIdx / effLayerCLSize;
		 if (patTypes[patIdx] == PATTYPE_NONE) {
			 niDeltas       [outputIdx] = 0;
			 igDeltas       [outputIdx] = 0;
			 fgDeltas       [outputIdx] = 0;
			 ogDeltas       [outputIdx] = 0;
			 cellStateErrors[outputIdx] = 0;
			 return;
		 }
	 }

	 // calculate indices
	 int blockIdx = outputIdx % effLayerCLSize;

	 // load the niag activations, the cell state and the output error
	 float niAct     = niActs      [outputIdx];
	 float igAct     = igActs      [outputIdx];
	 float ogAct     = ogActs      [outputIdx];
	 float cellState = cellStates  [outputIdx];

	 // calculate the output gate delta
	 float ogDelta = Logistic_deriv(ogAct) * Tanh_fn(cellState) * outputErr;

	 // calculate the cell state error
	 float ogPeepWeight = ogPeepWeights[blockIdx];
	 float cellStateErr = ogAct * Tanh_deriv(Tanh_fn(cellState)) * outputErr + ogPeepWeight * ogDelta;

	 if (!(firstCall == 1)) {
		 float nextFgAct        = fgActs         [outputIdx - prevOutputDistance];
		 float nextCellStateErr = cellStateErrors[outputIdx - prevOutputDistance];
		 float nextIgDelta      = igDeltas       [outputIdx - prevOutputDistance];
		 float nextFgDelta      = fgDeltas       [outputIdx - prevOutputDistance];

		 float igPeepWeight = igPeepWeights[blockIdx];
		 float fgPeepWeight = fgPeepWeights[blockIdx];

		 cellStateErr += nextFgAct * nextCellStateErr + igPeepWeight * nextIgDelta + fgPeepWeight * nextFgDelta;
	 }

	 // calculate the net input delta
	 float niDelta = igAct * Tanh_deriv(niAct) * cellStateErr;

	 // calculate the forget gate delta
	 float fgDelta = 0;

	 if (!(lastCall == 1)) {
		 float fgAct         = fgActs    [outputIdx];
		 float prevCellState = cellStates[outputIdx + prevOutputDistance];

		 fgDelta = Logistic_deriv(fgAct) * prevCellState * cellStateErr;
	 }

	 // calculate the input gate delta
	 float igDelta = Logistic_deriv(igAct) * niAct * cellStateErr;

	 // store the niag deltas and the cell state error
	 niDeltas       [outputIdx] = limitedError(niDelta);
	 igDeltas       [outputIdx] = limitedError(igDelta);
	 fgDeltas       [outputIdx] = limitedError(fgDelta);
	 ogDeltas       [outputIdx] = limitedError(ogDelta);
	 cellStateErrors[outputIdx] = cellStateErr;
	
}

__kernel void ll_computeWeightUpdateFn(int layerSize, int effLayerCLSize,
		int precLayerCLSize, int timestepDistance,
		int parallelSequences, int patternsCount,
		int biasWeightsOffset, int internalWeightsOffset,
		int peepholeWeightsOffset,
		float bias,
		__global const float * plOutputs,
		__global const float * fwOutputs,
		__global const float * bwOutputs,
		__global const float * fwCellStates,
		__global const float * bwCellStates,
		__global const float * fwNiDeltas,
		__global const float * bwNiDeltas,
		__global const float * fwIgDeltas,
		__global const float * bwIgDeltas,
		__global const float * fwFgDeltas,
		__global const float * bwFgDeltas,
		__global const float * fwOgDeltas,
		__global const float * bwOgDeltas,
		__global  float * Output
		
		){
	    int weightIdx = get_global_id(0);
	    
//	    Output[weightIdx] = plOutputs[weightIdx];
//	    return;
			   
			   
		int inwc = layerSize * precLayerCLSize;
		int biwc = layerSize;
		int itwc = layerSize * effLayerCLSize;
		int pewc = layerSize;

		int weightType = (int)(weightIdx >= 0                     + 1 * inwc) +
						 (int)(weightIdx >= 0                     + 2 * inwc) +
						 (int)(weightIdx >= 0                     + 3 * inwc) +
						 (int)(weightIdx >= 0                     + 4 * inwc) +
						 (int)(weightIdx >= biasWeightsOffset     + 1 * biwc) +
						 (int)(weightIdx >= biasWeightsOffset     + 2 * biwc) +
						 (int)(weightIdx >= biasWeightsOffset     + 3 * biwc) +
						 (int)(weightIdx >= biasWeightsOffset     + 4 * biwc) +
						 (int)(weightIdx >= internalWeightsOffset + 1 * itwc) +
						 (int)(weightIdx >= internalWeightsOffset + 2 * itwc) +
						 (int)(weightIdx >= internalWeightsOffset + 3 * itwc) +
						 (int)(weightIdx >= internalWeightsOffset + 4 * itwc) * 2 +
						 (int)(weightIdx >= peepholeWeightsOffset + 1 * pewc) +
						 (int)(weightIdx >= peepholeWeightsOffset + 2 * pewc);

		int weightTypeX = weightType & 0xC;
		int weightTypeY = weightType & 0x3;

		// calculate indices, offsets and increments
		__global const float *offOutputs;
		int           tgtBlockIdx;
		int           offOutputsInc;
		bool          skipFirstPattern = false;
		bool          skipLastPattern  = false;
		bool          isBwStateWeight;

		switch (weightTypeX) {
		// input weight
		case 0x0:
			{{
				// calculate indices
				int inputWeightIdx = weightIdx;
				int plBlockIdx     = inputWeightIdx % precLayerCLSize;
				int blockIdx       = (inputWeightIdx - weightTypeY * (biasWeightsOffset/4)) / precLayerCLSize;

				// check if we calculate backward state weights and adjust the block index
				isBwStateWeight = (blockIdx >= effLayerCLSize);
				if (isBwStateWeight)
					blockIdx -= effLayerCLSize;

				// set values for the loop below
				tgtBlockIdx   = blockIdx;
				offOutputs    = &plOutputs[plBlockIdx];
				offOutputsInc = precLayerCLSize;
			}}
			break;

		// bias weight
		case 0x4:
			{{
				// calculate indices
				int biasWeightIdx = weightIdx - biasWeightsOffset;
				int blockIdx      = biasWeightIdx - weightTypeY * layerSize;

				// check if we calculate backward state weights and adjust the block index
				isBwStateWeight = (blockIdx >= effLayerCLSize);
				if (isBwStateWeight)
					blockIdx -= effLayerCLSize;

				// set values for the loop below
				tgtBlockIdx   = blockIdx;
				offOutputs    = 0;
				offOutputsInc = 0;
			}}
			break;

		// internal weight
		case 0x8:
			{{
				// calculate indices
				int internalWeightIdx = weightIdx - internalWeightsOffset;
				int srcBlockIdx       = internalWeightIdx % effLayerCLSize;
				int blockIdx          = internalWeightIdx / effLayerCLSize - weightTypeY * layerSize;

				// check if we calculate backward state weights and adjust the block index
				isBwStateWeight = (blockIdx >= effLayerCLSize);
				if (isBwStateWeight)
					blockIdx -= effLayerCLSize;

				// set values for the loop below
				tgtBlockIdx   = blockIdx;
				offOutputs    = (isBwStateWeight ? &bwOutputs[srcBlockIdx] : &fwOutputs[srcBlockIdx]);
				offOutputsInc = effLayerCLSize;

				if (isBwStateWeight) {
					offOutputs += timestepDistance;
					skipLastPattern = true;
				}
				else {
					offOutputs -= timestepDistance;
					skipFirstPattern = true;
				}
			}}
			break;

		// peephole weight
		default:
			{{
				// calculate indices
				int peepholeWeightIdx = weightIdx - peepholeWeightsOffset;
				int blockIdx          = peepholeWeightIdx - (weightTypeY-1) * layerSize;

				// check if we calculate backward state weights and adjust the block index
				isBwStateWeight = (blockIdx >= effLayerCLSize);
				if (isBwStateWeight)
					blockIdx -= effLayerCLSize;

				// select the appropriate cell states and adjust the block index
				__global const float *cellStates = (isBwStateWeight ? bwCellStates : fwCellStates);

				// set the timeshift
				int timeShift;
				if (weightTypeY == 0x3) {
					timeShift = 0;
				}
				else {
					if (isBwStateWeight) {
						timeShift       = timestepDistance;
						skipLastPattern = true;
					}
					else {
						timeShift        = -timestepDistance;
						skipFirstPattern = true;
					}
				}

				// set values for the loop below
				tgtBlockIdx   = blockIdx;
				offOutputs    = &cellStates[blockIdx + timeShift];
				offOutputsInc = effLayerCLSize;
			}}
			break;
		}

		// determine the start of the delta values
		__global const float *niagDeltasLut[] = {
			fwNiDeltas,
			fwIgDeltas,
			fwFgDeltas,
			fwOgDeltas,
			bwNiDeltas,
			bwIgDeltas,
			bwFgDeltas,
			bwOgDeltas
		};

		// calculate the weight update over all patterns
		__global const float *offDeltas = &niagDeltasLut[weightTypeY + (isBwStateWeight ? 4 : 0)][tgtBlockIdx];

		if (skipFirstPattern) {
			offOutputs += parallelSequences * offOutputsInc;
			offDeltas  += parallelSequences * effLayerCLSize;
		}

		int numPatterns = patternsCount;
		if (skipFirstPattern || skipLastPattern)
			numPatterns -= parallelSequences;

		float wu = 0;
		for (int i = 0; i < numPatterns; ++i) {
			wu += (offOutputs ? *offOutputs : bias) * *offDeltas;

			offOutputs += offOutputsInc;
			offDeltas  += effLayerCLSize;
		}

		Output[weightIdx] =  wu;
	
}


__kernel void sdo_updateWeightFn(float learningRate, float momentum,
		 __global float * weights, __global float * weightUpdates,
		 __global float * weightDeltas, __global float * out){
	// calculate and store the weight delta
	int weightIdx = get_global_id(0);
	
	float delta = momentum * weightDeltas[weightIdx] - learningRate * weightUpdates[weightIdx];
	weightDeltas[weightIdx] = delta;

	// calculate the new weight
	float newWeight = weights[weightIdx] + delta;

	out[weightIdx] = newWeight;
}

