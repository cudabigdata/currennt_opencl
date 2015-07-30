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

#include "WeightedSsePostOutputLayerCL.hpp"



namespace layers {

    
    WeightedSsePostOutputLayerCL::WeightedSsePostOutputLayerCL(const helpers::JsonValue &layerChild, LayerCL &precedingLayerCL)
        : PostOutputLayerCL  (layerChild, precedingLayerCL, precedingLayerCL.size() * 2)
    {
    }


    WeightedSsePostOutputLayerCL::~WeightedSsePostOutputLayerCL()
    {
    }


    const std::string& WeightedSsePostOutputLayerCL::type() const
    {
        static const std::string s("weightedsse");
        return s;
    }


    real_t WeightedSsePostOutputLayerCL::calculateError()
    {

        int layerSize = this->size() / 2;

        int n = this->curMaxSeqLength() * this->parallelSequences() * this->size() / 2;

        real_t mse = (real_t)0.5 * SystemCL::weighedSSeFn(layerSize,this->patTypes(),
        		this->_targets(), this->_actualOutputs(),n);

        return mse;
    }


    void WeightedSsePostOutputLayerCL::computeForwardPass()
    {
    }


    void WeightedSsePostOutputLayerCL::computeBackwardPass()
    {
        // calculate the errors

        int layerSize = this->size() / 2;
        int n = this->curMaxSeqLength() * this->parallelSequences() * this->size() / 2;
        SystemCL::outputErrorFn(layerSize,this->patTypes(),this->_targets(),this->_actualOutputs(),
        		this->_outputErrors(),n);
    }




} // namespace layers
