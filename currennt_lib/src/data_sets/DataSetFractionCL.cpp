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

#include "DataSetFractionCL.hpp"


namespace data_sets {

	DataSetFractionCL::DataSetFractionCL()
    {
    }

	DataSetFractionCL::~DataSetFractionCL()
    {
    }

    int DataSetFractionCL::inputPatternSize() const
    {
        return m_inputPatternSize;
    }

    int DataSetFractionCL::outputPatternSize() const
    {
        return m_outputPatternSize;
    }

    int DataSetFractionCL::maxSeqLength() const
    {
        return m_maxSeqLength;
    }

    int DataSetFractionCL::minSeqLength() const
    {
        return m_minSeqLength;
    }

    int DataSetFractionCL::numSequences() const
    {
        return (int)m_seqInfo.size();
    }

    const DataSetFractionCL::seq_info_t& DataSetFractionCL::seqInfo(int seqIdx) const
    {
        return m_seqInfo[seqIdx];
    }

    const std::vector<char>& DataSetFractionCL::patTypes() const
    {
        return m_patTypes;
    }

    const std::vector<real_t>& DataSetFractionCL::inputs() const
    {
        return m_inputs;
    }

    const std::vector<real_t>& DataSetFractionCL::outputs() const
    {
        return m_outputs;
    }

    const std::vector<int>& DataSetFractionCL::targetClasses() const
    {
        return m_targetClasses;
    }

} // namespace data_sets
