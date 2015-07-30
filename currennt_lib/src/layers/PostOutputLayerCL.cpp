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

#include "PostOutputLayerCL.hpp"

#include <boost/lexical_cast.hpp>
#include <stdexcept>


namespace layers {


    d_mem_real& PostOutputLayerCL::_targets()
    {
        return this->outputs();
    }


    d_mem_real& PostOutputLayerCL::_actualOutputs()
    {
        return m_precedingLayerCL.outputs();
    }


    d_mem_real& PostOutputLayerCL::_outputErrors()
    {
        return m_precedingLayerCL.outputErrors();
    }


    PostOutputLayerCL::PostOutputLayerCL(
        const helpers::JsonValue &layerChild, 
        LayerCL &precedingLayerCL,
        int requiredSize,
        bool createOutputs)
        : LayerCL  (layerChild, precedingLayerCL.parallelSequences(), precedingLayerCL.maxSeqLength(), createOutputs)
        , m_precedingLayerCL(precedingLayerCL)
    {
        if (this->size() != requiredSize)
            throw std::runtime_error("Size mismatch: " + boost::lexical_cast<std::string>(this->size()) + " vs. " + boost::lexical_cast<std::string>(requiredSize));
    }


    PostOutputLayerCL::~PostOutputLayerCL()
    {
    }


    void PostOutputLayerCL::loadSequences(const data_sets::DataSetFractionCL &fraction)
    {
        if (fraction.outputPatternSize() != this->size()) {
            throw std::runtime_error(std::string("Output layer size of ") + boost::lexical_cast<std::string>(this->size())
            + " != data target pattern size of " + boost::lexical_cast<std::string>(fraction.outputPatternSize()));
        }

        LayerCL::loadSequences(fraction);

        if (!this->_outputs().size == 0)
        	 SystemCL::copy_real(this->_outputs(), fraction.outputs());
    }



} // namespace layers
