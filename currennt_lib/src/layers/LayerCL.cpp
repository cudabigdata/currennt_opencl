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
#   pragma warning (disable: 4244)
#endif

#include "LayerCL.hpp"
#include "../helpers/JsonClasses.hpp"

#include <stdexcept>


namespace layers {


    d_mem_real& LayerCL::_outputs()
    {
        return m_outputs;
    }


    LayerCL::LayerCL(const helpers::JsonValue &layerChild, int parallelSequences, int maxSeqLength, bool createOutputs)
        : m_name             (layerChild->HasMember("name") ? (*layerChild)["name"].GetString()  : "")
        , m_size             (layerChild->HasMember("size") ? (*layerChild)["size"].GetInt()     : 0)
        , m_parallelSequences(parallelSequences)
        , m_maxSeqLength     (maxSeqLength)
        , m_curMaxSeqLength  (0)
        , m_curMinSeqLength  (0)
        , m_curNumSeqs       (0)
    {
        // check if the name and size values exist
        if (!layerChild->HasMember("name"))
            throw std::runtime_error("Missing value 'name' in layer description");
        if (m_name.empty())
            throw std::runtime_error("Empty layer name in layer description");
        if (!layerChild->HasMember("size"))
            throw std::runtime_error(std::string("Missing value 'size' in layer '") + m_name + "'");

        // allocate space for the vectors
        if (createOutputs)
             SystemCL::malloc_real(m_outputs, m_parallelSequences * m_maxSeqLength * m_size);

             SystemCL::malloc_char(m_patTypes, m_parallelSequences * m_maxSeqLength);

        // resize the output errors vector
        SystemCL::malloc_real(m_outputErrors, this->_outputs().size, (real_t)0);
    }


    LayerCL::~LayerCL()
    {
    }


    const std::string& LayerCL::name() const
    {
        return m_name;
    }


    int LayerCL::size() const
    {
        return m_size;
    }


    int LayerCL::parallelSequences() const
    {
        return m_parallelSequences;
    }


    int LayerCL::maxSeqLength() const
    {
        return m_maxSeqLength;
    }


    int LayerCL::curMaxSeqLength() const
    {
        return m_curMaxSeqLength;
    }


    int LayerCL::curMinSeqLength() const
    {
        return m_curMinSeqLength;
    }


    int LayerCL::curNumSeqs() const
    {
        return m_curNumSeqs;
    }


    const  d_mem_char& LayerCL::patTypes() const
    {
        return m_patTypes;
    }


     d_mem_real& LayerCL::outputs()
    {
        return m_outputs;
    }


     d_mem_real& LayerCL::outputErrors()
    {
        return m_outputErrors;
    }


    void LayerCL::loadSequences(const data_sets::DataSetFractionCL &fraction)
    {
        m_curMaxSeqLength = fraction.maxSeqLength();
        m_curMinSeqLength = fraction.minSeqLength();
        m_curNumSeqs      = fraction.numSequences();
        SystemCL::copy_char(m_patTypes, fraction.patTypes());



    }
    

    void LayerCL::exportLayerCL(const helpers::JsonValue &layersArray, const helpers::JsonAllocator &allocator) const
    {
        if (!layersArray->IsArray())
            throw std::runtime_error("The JSON value is not an array");

        // create and fill the layer object
        rapidjson::Value layerObject(rapidjson::kObjectType);
        layerObject.AddMember("name", name().c_str(), allocator);
        layerObject.AddMember("type", type().c_str(), allocator);
        layerObject.AddMember("size", size(),         allocator);

        // add the layer object to the layers array
        layersArray->PushBack(layerObject, allocator);
    }



} // namespace layers
