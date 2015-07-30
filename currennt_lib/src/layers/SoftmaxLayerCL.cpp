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

#include "SoftmaxLayerCL.hpp"





namespace layers {


    SoftmaxLayerCL::SoftmaxLayerCL(
        const helpers::JsonValue &layerChild, 
        const helpers::JsonValue &weightsSection,
        LayerCL &precedingLayerCL, std::string type)
        : FeedForwardLayerCL(layerChild, weightsSection, precedingLayerCL, type)
    {
         SystemCL::malloc_real(m_patTmp, this->patTypes().size);
    }


    SoftmaxLayerCL::~SoftmaxLayerCL()
    {
    }


    const std::string& SoftmaxLayerCL::type() const
    {
        static const std::string s = "softmax";
        return s;
    }


    void SoftmaxLayerCL::computeForwardPass()
    {
        // compute the forward pass of the feedforward layer
        FeedForwardLayerCL::computeForwardPass();

        // calculate the offset to center the activations for safer exponentiation
        {{

            int layerSize = this->size();

            int n = this->curMaxSeqLength() * this->parallelSequences();


            SystemCL::sml_calculateOffsetFn(layerSize, this->_outputs(), this->patTypes(),
            		m_patTmp, n);
        }}



        // calculate the exponent
        {{
            int layerSize = this->size();

            int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

            SystemCL::sml_calculateExpFn(layerSize, m_patTmp,  this->_outputs(),n );
        }}

        // sum up all outputs for each pattern
        {{

            int layerSize = this->size();
            int n = this->curMaxSeqLength() * this->parallelSequences();
            SystemCL::sml_sumUpOutputsFn(layerSize, this->_outputs(), m_patTmp, n);
        }}

        // normalize the outputs
        {{

            int layerSize = this->size();

           int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

            SystemCL::sml_normalizeOutputsFn(layerSize,m_patTmp,this->_outputs(),  n);
        }}
    }


    void SoftmaxLayerCL::computeBackwardPass()
    {
        // calculate the error offset for each pattern
        {{
            int layerSize    = this->size();
            int n = this->curMaxSeqLength() * this->parallelSequences();
            SystemCL::sml_calculateErrorOffsetFn(layerSize,this->_outputs(),this->outputErrors(),
            		this->patTypes(), m_patTmp, n);

        }}
        // calculate the errors
        {{
            int layerSize    = this->size();

            int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

            SystemCL::sml_calculateErrorsFn(layerSize, m_patTmp,this->outputErrors(),  this->_outputs(),  n);
        }}
        // compute the backward pass of the feedforward layer
        FeedForwardLayerCL::computeBackwardPass();
    }



} // namespace layers
