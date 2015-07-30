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

#include "InputLayerCL.hpp"

#include <boost/lexical_cast.hpp>
#include <stdexcept>


namespace layers {


    InputLayerCL::InputLayerCL(const helpers::JsonValue &layerChild, int parallelSequences, int maxSeqLength)
        : LayerCL(layerChild, parallelSequences, maxSeqLength)
    {
    }


    InputLayerCL::~InputLayerCL()
    {
    }


    const std::string& InputLayerCL::type() const
    {
        static const std::string s("input");
        return s;
    }


    void InputLayerCL::loadSequences(const data_sets::DataSetFractionCL &fraction)
    {
        if (fraction.inputPatternSize() != this->size()) {
            throw std::runtime_error(std::string("Input layer size of ") + boost::lexical_cast<std::string>(this->size())
            + " != data input pattern size of " + boost::lexical_cast<std::string>(fraction.inputPatternSize()));
        }

        LayerCL::loadSequences(fraction);

        SystemCL::copy_real(this->_outputs(), fraction.inputs());
    }


    void InputLayerCL::computeForwardPass()
    {
    }


    void InputLayerCL::computeBackwardPass()
    {
    }




} // namespace layers
