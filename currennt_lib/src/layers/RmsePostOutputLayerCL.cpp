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

#include "RmsePostOutputLayerCL.hpp"


namespace layers {


    RmsePostOutputLayerCL::RmsePostOutputLayerCL(const helpers::JsonValue &layerChild, LayerCL &precedingLayerCL)
        : PostOutputLayerCL(layerChild, precedingLayerCL, precedingLayerCL.size())
    {
        // resize the vector for RMSEs
        SystemCL::malloc_real(m_rmses, (int)this->patTypes().size);
    }


    RmsePostOutputLayerCL::~RmsePostOutputLayerCL()
    {
    }


    const std::string& RmsePostOutputLayerCL::type() const
    {
        static const std::string s("rmse");
        return s;
    }


    real_t RmsePostOutputLayerCL::calculateError()
    {

        return SystemCL::rpol_calculateError(m_rmses, this->curMaxSeqLength() * this->parallelSequences());
    }


    void RmsePostOutputLayerCL::computeForwardPass()
    {
        // calculate the RMSE for each pattern

        int layerSize     = this->size();

        int n = this->curMaxSeqLength() * this->parallelSequences();

        SystemCL::rpol_computeRmseFn(layerSize, this->_actualOutputs(),
        		this->_targets(), this->patTypes(), m_rmses , n);
    }


    void RmsePostOutputLayerCL::computeBackwardPass()
    {

        int layerSize = this->size();

        int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();


        SystemCL::rpol_computeOutputErrorFn(layerSize, m_rmses,this->_actualOutputs(),
        		this->_targets(),  this->_outputErrors(),n);
    }



} // namespace layers
