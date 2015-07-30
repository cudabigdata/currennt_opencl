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

#include "FeedForwardLayerCL.hpp"

#include "../helpers/MatrixCL.hpp"


#include <typeinfo>
#include <stdexcept>

//namespace internal {
//namespace {
//
//    template <typename TActFn>
//    struct ComputeOutputFn
//    {
//        int    layerSize;
//        real_t bias;
//
//        const real_t *biasWeights;
//
//        __host__ __device__ real_t operator() (real_t a, const int &outputIdx) const
//        {
//            // calculate indices
//            int blockIdx = outputIdx % layerSize;
//
//            // add the bias
//            a += bias * biasWeights[blockIdx];
//
//            // apply the activation function
//            real_t b = TActFn::fn(a);
//
//            // store the activation
//            return b;
//        }
//    };
//
//    template <typename TActFn>
//    struct ComputeDeltaFn
//    {
//        // since calculating the derivatives is very cheap for our activation functions,
//        // we simple calculate the deltas of all timesteps, including dummies
//
//        __host__ __device__ void operator() (const thrust::tuple<real_t&, const real_t&> &t) const
//        {
//            real_t delta = TActFn::deriv(t.get<1>()) * t.get<0>();
//            t.get<0>() = delta;
//        }
//    };
//
//    struct ComputeBiasWeightUpdateFn
//    {
//        int    layerSize;
//        int    patternsCount;
//        real_t bias;
//
//        const real_t *deltas;
//
//        __host__ __device__ real_t operator() (const int &biasWeightIdx) const
//        {
//            const real_t *offDeltas = deltas + biasWeightIdx;
//
//            real_t wu = 0;
//            for (int i = 0; i < patternsCount; ++i) {
//                wu += bias * *offDeltas;
//                offDeltas += layerSize;
//            }
//
//            return wu;
//        }
//    };
//
//} // anonymous namespace
//} // namespace internal


namespace layers {


    FeedForwardLayerCL::FeedForwardLayerCL(
        const helpers::JsonValue &layerChild, 
        const helpers::JsonValue &weightsSection,
        LayerCL &precedingLayerCL, std::string act_fun)
        : TrainableLayerCL(layerChild, weightsSection, 1, 0, precedingLayerCL)
    {
    	activate_fun = act_fun;
    }


    FeedForwardLayerCL::~FeedForwardLayerCL()
    {
    }


    const std::string& FeedForwardLayerCL::type() const
    {
        static std::string s;

        if (s.empty()) {
            if (activate_fun == "TANH")
                s = "feedforward_tanh";
            else if (activate_fun == "LOGISTIC")
                s = "feedforward_logistic";
            else if (activate_fun == "IDENTITY")
                s = "feedforward_identity";
            else
                throw std::runtime_error("Unsupported activation function");
        }
        
        return s;
    }


    void FeedForwardLayerCL::computeForwardPass()
    {
        // collect outputs from preceding layer
        {{
            helpers::MatrixCL weightsMatrix  (&this->weights(),                  this->precedingLayerCL().size(), this->size());
            helpers::MatrixCL plOutputsMatrix(&this->precedingLayerCL().outputs(), this->precedingLayerCL().size(), this->curMaxSeqLength() * this->parallelSequences());
            helpers::MatrixCL outputsMatrix  (&this->_outputs(),                 this->size(),                  this->curMaxSeqLength() * this->parallelSequences());

            outputsMatrix.assignProduct(weightsMatrix, true, plOutputsMatrix, false);
        }}

        // calculate the outputs of the layer
        int layerSize   = this->size();
        real_t	bias		= this->bias();
        int biasOffset = this->size() * this->precedingLayerCL().size();
        int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

        int typeFunction  = 2;
        if (activate_fun == "TANH")
        	typeFunction = 0;
        else if (activate_fun == "LOGISTIC")
        	typeFunction = 1;
        else if (activate_fun == "IDENTITY")
        	typeFunction = 2;

        SystemCL::ffl_computeOutputFn(layerSize, bias, this->weights(), biasOffset, this->_outputs(), n,
        		typeFunction );


    }

    void FeedForwardLayerCL::computeBackwardPass()
    {

		int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

		int typeFunction  = 2;
		if (activate_fun == "TANH")
			typeFunction = 0;
		else if (activate_fun == "LOGISTIC")
			typeFunction = 1;
		else if (activate_fun == "IDENTITY")
			typeFunction = 2;
    	SystemCL::ffl_computeDeltaFn(this->outputErrors(), this->outputs(), n,typeFunction);

        // back-propagate the error to the preceding layer
        {{
            TrainableLayerCL *pl = dynamic_cast<TrainableLayerCL*>(&this->precedingLayerCL());
            if (pl) {
                helpers::MatrixCL weightsMatrix (&this->weights(),      pl->size(),   this->size());
                helpers::MatrixCL plErrorsMatrix(&pl->outputErrors(),   pl->size(),   this->curMaxSeqLength() * this->parallelSequences());
                helpers::MatrixCL deltasMatrix  (&this->outputErrors(), this->size(), this->curMaxSeqLength() * this->parallelSequences());

                plErrorsMatrix.assignProduct(weightsMatrix, false, deltasMatrix, false);
            }
        }}

        // compute the input weight updates
        {{
            helpers::MatrixCL weightUpdatesMatrix(&this->_weightUpdates(),           this->precedingLayerCL().size(), this->size());
            helpers::MatrixCL plOutputsMatrix    (&this->precedingLayerCL().outputs(), this->precedingLayerCL().size(), this->curMaxSeqLength() * this->parallelSequences());
            helpers::MatrixCL deltasMatrix       (&this->outputErrors(),             this->size(),                  this->curMaxSeqLength() * this->parallelSequences());

            weightUpdatesMatrix.assignProduct(plOutputsMatrix, false, deltasMatrix, true);
        }}

        // compute the bias weight updates

            int layerSize     = this->size();
            int patternsCount = this->curMaxSeqLength() * this->parallelSequences();
            int offset =  this->precedingLayerCL().size() * this->size();


        SystemCL::ffl_computeBiasWeightUpdateFn(layerSize, patternsCount, this->bias(), this->outputErrors()
        		          ,  this->weightUpdates(), offset, this->size() );

    }



} // namespace layers
