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

#ifndef DATA_SETS_DATASETFRACTIONCL_HPP
#define DATA_SETS_DATASETFRACTIONCL_HPP

#include "../TypesCL.hpp"

#include <vector>
#include <string>


namespace data_sets {

    /******************************************************************************************//**
     * Contains a fraction of the data sequences in a DataSet that is small enough to be 
     * transferred completely to the GPU
     *********************************************************************************************/
    class DataSetFractionCL
    {
        friend class DataSetCL;

    public:
        struct seq_info_t {
            int         originalSeqIdx;
            int         length;
            std::string seqTag;
        };

    private:
        int m_inputPatternSize;
        int m_outputPatternSize;
        int m_maxSeqLength;
        int m_minSeqLength;

        std::vector<seq_info_t> m_seqInfo;

        std::vector<real_t>     m_inputs;
        std::vector<real_t>     m_outputs;
        std::vector<char>       m_patTypes;
        std::vector<int>        m_targetClasses;

    private:
        /**
         * Creates the instance
         */
        DataSetFractionCL();

    public:
        /**
         * Destructor
         */
        ~DataSetFractionCL();

        /**
         * Returns the size of each input pattern
         *
         * @return The size of each input pattern
         */
        int inputPatternSize() const;

        /**
         * Returns the size of each output pattern
         *
         * @return The size of each output pattern
         */
        int outputPatternSize() const;

        /**
         * Returns the length of the longest sequence
         *
         * @return The length of the longest sequence
         */
        int maxSeqLength() const;

        /**
         * Returns the length of the shortest sequence
         *
         * @return The length of the shortest sequence
         */
        int minSeqLength() const;

        /**
         * Returns the number of sequences in the fraction
         *
         * @return The number of sequences in the fraction
         */
        int numSequences() const;

        /**
         * Returns information about a sequence
         *
         * @param seqIdx The index of the sequence
         */
        const seq_info_t& seqInfo(int seqIdx) const;

        /**
         * Returns the pattern types vector
         *
         * @return The pattern types vector
         */
        const std::vector<char>& patTypes() const;

        /**
         * Returns the input patterns vector
         *
         * @return The input patterns vector
         */
        const std::vector<real_t>& inputs() const;

        /**
         * Returns the output patterns vector
         *
         * @return The output patterns vector
         */
        const std::vector<real_t>& outputs() const;

        /**
         * Returns the target classes vector
         *
         * @return The target classes vector
         */
        const std::vector<int>& targetClasses() const;
    };

} // namespace data_sets


#endif // DATA_SETS_DATASETFRACTIONCL_HPP
