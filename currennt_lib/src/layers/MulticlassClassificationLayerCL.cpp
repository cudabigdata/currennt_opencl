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

#include "MulticlassClassificationLayerCL.hpp"

#include <stdexcept>
#include <cassert>


#define SKIP_MARKER helpers::NumericLimits<real_t>::max()


namespace layers {


    MulticlassClassificationLayerCL::MulticlassClassificationLayerCL(const helpers::JsonValue &layerChild, LayerCL &precedingLayerCLL)
        : PostOutputLayerCL(layerChild, precedingLayerCLL, precedingLayerCLL.size(), false)
    {
        if (this->size() == 1)
            throw std::runtime_error("The multiclass classification post output layer cannot be used for an output layer size of 1");

        // resize the pattern target classes vector
        SystemCL::malloc_int(m_patTargetClasses, this->patTypes().size);
    }


    MulticlassClassificationLayerCL::~MulticlassClassificationLayerCL()
    {
    }


    int MulticlassClassificationLayerCL::countCorrectClassifications()
    {

        int layerSize = this->size();

        int n = this->curMaxSeqLength() * this->parallelSequences();

        return SystemCL::mcl_countCorrectClassificationsFn(layerSize, this->_actualOutputs(), m_patTargetClasses, n);

    }
    

    const std::string& MulticlassClassificationLayerCL::type() const
    {
        static std::string s("multiclass_classification");
        return s;
    }


    void MulticlassClassificationLayerCL::loadSequences(const data_sets::DataSetFractionCL &fraction)
    {
        PostOutputLayerCL::loadSequences(fraction);

        SystemCL::copy_int(m_patTargetClasses, fraction.targetClasses());
    }

    real_t MulticlassClassificationLayerCL::calculateError()
    {
        // calculate the cross entropy error
        int layerSize = this->size();
        int n = this->curMaxSeqLength() * this->parallelSequences();
        real_t error = SystemCL::mcl_computeCrossEntropyErrorFn(layerSize,this->_actualOutputs(),
        		m_patTargetClasses, n);
        return -error;
    }


    void MulticlassClassificationLayerCL::computeForwardPass()
    {
    }


    void MulticlassClassificationLayerCL::computeBackwardPass()
    {
        int n = this->curMaxSeqLength() * this->parallelSequences();

        // set all errors to zero
        assert (n * this->size() <= this->_outputErrors().size);

        SystemCL::fill(this->_outputErrors(), n * this->size(), 0);

        int layerSize    = this->size();

        SystemCL::mcl_computeOutputErrorFn(layerSize, this->_actualOutputs(),this->_outputErrors(),m_patTargetClasses,n );
    }



} // namespace layers
