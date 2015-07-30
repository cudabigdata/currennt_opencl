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

#ifndef LAYERS_LAYERCL_HPP
#define LAYERS_LAYERCL_HPP

#include "../TypesCL.hpp"
#include "../data_sets/DataSetFractionCL.hpp"
#include "../helpers/JsonClassesForward.hpp"
#include "../SystemCL.hpp"
#include <string>


namespace layers {

    /******************************************************************************************//**
     * Represents a layer in the neural network
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/

    class LayerCL
    {


    private:
        const std::string m_name;
        const int         m_size;
        const int         m_parallelSequences;
        const int         m_maxSeqLength;

        int               m_curMaxSeqLength;
        int               m_curMinSeqLength;
        int               m_curNumSeqs;
        d_mem_real        m_outputs;
        d_mem_real        m_outputErrors;
        d_mem_char        m_patTypes;

    protected:
        d_mem_real& _outputs();

    public:
        /**
         * Constructs the LayerCL
         *
         * @param layerChild        The layer child of the JSON configuration for this layer
         * @param parallelSequences The maximum number of sequences that shall be computed in parallel
         * @param maxSeqLength      The maximum length of a sequence
         * @param createOutputs     If false, then the outputs vector will be left empty
         */
        LayerCL(const helpers::JsonValue &layerChild, int parallelSequences, int maxSeqLength, bool createOutputs = true);

        /**
         * Destructs the LayerCL
         */
        virtual ~LayerCL();

        /**
         * Returns the name of the layer
         *
         * @return The name of the layer
         */
        const std::string& name() const;

        /**
         * Returns the number of blocks in the layer
         * 
         * @return The number of blocks in the layer
         */
        int size() const;

        /**
         * Returns the maximum number of sequences that can be computed in parallel
         *
         * @return The maximum number of sequences that can be computed in parallel
         */
        int parallelSequences() const;

        /**
         * Returns the maximum length of a sequence
         *
         * @return The maximum length of a sequence
         */
        int maxSeqLength() const;

        /**
         * Returns the maximum length of the currently loaded sequences
         *
         * @return The maximum length of the currently loaded sequences
         */
        int curMaxSeqLength() const;

        /**
         * Returns the minimum length of the currently loaded sequences
         *
         * @return The minimum length of the currently loaded sequences
         */
        int curMinSeqLength() const;

        /**
         * Returns the number sequences in the current data set fraction
         *
         * @return The number sequences in the current data set fraction
         */
        int curNumSeqs() const;

        /**
         * Calculates the output errors of the layer
         *
         * @return The output error
         */
        d_mem_real& outputErrors();

        /**
         * Returns the pattern types vector
         * 
         * @return The pattern types vector
         */
        const d_mem_char& patTypes() const;

        /**
         * Returns a string describing the layer type
         *
         * @return A string describing the layer type
         */
        virtual const std::string& type() const =0;

        /**
         * Returns the outputs of the layer
         *
         * @return The outputs
         */
        virtual d_mem_real& outputs();

        /**
         * Loads sequences from a data set
         *
         * @param fraction The fraction of the data set to load
         */
        virtual void loadSequences(const data_sets::DataSetFractionCL &fraction);

        /**
         * Computes the forward pass
         */
        virtual void computeForwardPass() =0;

        /**
         * Computes the backward pass, including the weight updates
         */
        virtual void computeBackwardPass() =0;

        /**
         * Stores the description of the layer in a JSON object
         *
         * @param layersArray The array of layers in the document
         * @param allocator   The allocator to use
         */
        virtual void exportLayerCL(const helpers::JsonValue &layersArray, const helpers::JsonAllocator &allocator) const;
    };

} // namespace layers


#endif // LAYERS_LAYERCL_HPP
