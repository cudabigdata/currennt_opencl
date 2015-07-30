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

#ifdef _MSC_VER
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif

#include "CePostOutputLayerCL.hpp"




//
//namespace internal {
//namespace {
//
//    struct ComputeCeFn
//    {
//        int layerSize;
//
//        const char *patTypes;
//
//        __host__ __device__ real_t operator() (const thrust::tuple<real_t, real_t, int> &values) const
//        {
//            // unpack the tuple
//            real_t target = values.get<0>();
//            real_t output = values.get<1>();
//            int outputIdx = values.get<2>();
//
//            // check if we have to skip this value
//            int patIdx = outputIdx / layerSize;
//            if (patTypes[patIdx] == PATTYPE_NONE)
//                return 0;
//
//            // calculate the error
//            // this is actually KL div.
//            // it does not have the additive term (entropy of targets)
//            // so it always goes to zero if the distributions are identical
//            // but the derivative is the same as CE ...
//            real_t ftarget = helpers::max(helpers::NumericLimits<real_t>::min(), target);
//            output = helpers::max(helpers::NumericLimits<real_t>::min(), output);
//            real_t div = target * log(ftarget / output);
//            return div;
//        }
//    };
//
//    struct ComputeOutputErrorFn
//    {
//        int layerSize;
//
//        const char *patTypes;
//
//        __host__ __device__ real_t operator() (const thrust::tuple<const real_t&, const real_t&, int> &t) const
//        {
//            // unpack the tuple
//            real_t actualOutput = t.get<0>();
//            real_t targetOutput = t.get<1>();
//            int    outputIdx    = t.get<2>();
//
//            // calculate the pattern index
//            int patIdx = outputIdx / layerSize;
//
//            // check if the pattern is a dummy
//            if (patTypes[patIdx] == PATTYPE_NONE)
//                return 0;
//
//            // calculate the error
//            actualOutput = helpers::max(helpers::NumericLimits<real_t>::min(), actualOutput);
//            real_t bp_error = helpers::boundRange(-targetOutput / actualOutput, -100, +100);
//
//            return bp_error;
//        }
//    };
//
//} // anonymous namespace
//} // namespace anonymous


namespace layers {

    
    CePostOutputLayerCL::CePostOutputLayerCL(const helpers::JsonValue &layerChild, LayerCL &precedingLayerCL)
        : PostOutputLayerCL(layerChild, precedingLayerCL, precedingLayerCL.size())
    {
    }


    CePostOutputLayerCL::~CePostOutputLayerCL()
    {
    }


    const std::string& CePostOutputLayerCL::type() const
    {
        static const std::string s("ce");
        return s;
    }


    real_t CePostOutputLayerCL::calculateError()
    {
        int layerSize = this->size();
        int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
        return SystemCL::cpol_computeCeFn(layerSize,this->patTypes(),this->_targets(),this->_actualOutputs(),n);
    }


    void CePostOutputLayerCL::computeForwardPass()
    {
    }


    void CePostOutputLayerCL::computeBackwardPass()
    {

    	int layerSize = this->size();

        int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

    	SystemCL::cpol_computeOutputErrorFn(layerSize, this->patTypes(),this->_actualOutputs(),this->_targets(),this->_outputErrors(), n);

    }



} // namespace layers
