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

#include "BinaryClassificationLayerCL.hpp"


#include <stdexcept>


//namespace internal {
//namespace {
//
//    struct ComputeCrossEntropyErrorFn
//    {
//        const char *patTypes;
//
//        __host__ __device__ real_t operator() (const thrust::tuple<real_t, real_t, int> &t) const
//        {
//            // unpack the tuple
//            int outputIdx = t.get<2>();
//
//            // check if we actually need to continue
//            if (patTypes[outputIdx] == PATTYPE_NONE)
//                return 0;
//
//            // calculate the cross entropy error
//            real_t target = t.get<0>();
//            real_t output = t.get<1>();
//
//            real_t act        = helpers::max(output, helpers::NumericLimits<real_t>::min());
//            real_t targetProb = (target > 0 ? act : 1-act);
//            real_t error      = -log(targetProb);
//
//            return error;
//        }
//    };
//
//    struct CountCorrectClassificationsFn
//    {
//        __host__ __device__ int operator() (const thrust::tuple<real_t, real_t, int> &t) const
//        {
//            // unpack the tuple
//            real_t target  = t.get<0>();
//            real_t output  = t.get<1>();
//            int    patType = t.get<2>();
//
//            // determine target and estimated class
//            bool tgtClass = (target > (real_t)0.5);
//            bool estClass = (output > (real_t)0.5);
//
//            // count correct classification
//            return (patType != PATTYPE_NONE) && (tgtClass == estClass);
//        }
//    };
//
//    struct ComputeOutputErrorFn
//    {
//        const char *patTypes;
//
//        __host__ __device__ void operator() (const thrust::tuple<real_t&, const real_t&, const real_t&, int> &t) const
//        {
//            // unpack the tuple
//            int outputIdx = t.get<3>();
//
//            // check if we actually need to continue
//            if (patTypes[outputIdx] == PATTYPE_NONE)
//                return;
//
//            // calculate the error
//            real_t target = t.get<1>();
//            real_t output = t.get<2>();
//
//            real_t act        = helpers::max(output, helpers::NumericLimits<real_t>::min());
//            real_t targetProb = (target > 0 ? act : 1-act);
//            real_t error      = (target > 0 ? -(1/targetProb) : (1/targetProb));
//
//            // store the error
//            t.get<0>() = error;
//        }
//    };
//
//} // anonymous namespace
//} // namespace anonymous


namespace layers {


    BinaryClassificationLayerCL::BinaryClassificationLayerCL(const helpers::JsonValue &layerChild, LayerCL &precedingLayerCL)
        : PostOutputLayerCL(layerChild, precedingLayerCL, precedingLayerCL.size())
    {
        if (this->size() != 1)
            throw std::runtime_error("The binary classification post output layer cannot be used for an output layer size != 1");
    }


    BinaryClassificationLayerCL::~BinaryClassificationLayerCL()
    {
    }


    int BinaryClassificationLayerCL::countCorrectClassifications()
    {

        int n = this->curMaxSeqLength() * this->parallelSequences();

        return SystemCL::bcl_countCorrectClassificationsFn(this->_targets(), this->_actualOutputs(),this->patTypes(), n);

    }
    

    const std::string& BinaryClassificationLayerCL::type() const
    {
        static std::string s("binary_classification");
        return s;
    }


    void BinaryClassificationLayerCL::loadSequences(const data_sets::DataSetFractionCL &fraction)
    {
        PostOutputLayerCL::loadSequences(fraction);

        SystemCL::copy_int_real( this->_targets(), fraction.targetClasses());
    }


    real_t BinaryClassificationLayerCL::calculateError()
    {

        int n = this->curMaxSeqLength() * this->parallelSequences();

        return SystemCL::bcl_computeCrossEntropyErrorFn(this->patTypes(),this->_targets(),this->_actualOutputs(),n);

    }


    void BinaryClassificationLayerCL::computeForwardPass()
    {
    }


    void BinaryClassificationLayerCL::computeBackwardPass()
    {

        int n = this->curMaxSeqLength() * this->parallelSequences();

    	SystemCL::bcl_computeOutputErrorFn(this->patTypes(),this->_outputErrors(),this->_targets(), this->_actualOutputs(), n);

    }


} // namespace layers
