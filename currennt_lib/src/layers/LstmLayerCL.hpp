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

#ifndef LAYERS_LSTMLAYERCL_HPP
#define LAYERS_LSTMLAYERCL_HPP

#include "TrainableLayerCL.hpp"
#include "../helpers/MatrixCL.hpp"


namespace layers {

    /******************************************************************************************//**
     * Represents a fully connected layer which uses LSTM cells with forget gates, peephole
     * connections and one cell per block
     *
     * weights; with P = precedingLayerCL().size() and L = size():
     *    ~ weights from preceding layer:
     *        - [0 .. PL-1]:    net input
     *        - [PL .. 2PL-1]:  input gate
     *        - [2PL .. 3PL-1]: forget gate
     *        - [3PL .. 4PL-1]: output gate
     *    ~ bias weights:
     *        - [4PL + 0  .. 4PL + L-1]:  net input
     *        - [4PL + L  .. 4PL + 2L-1]: input gate
     *        - [4PL + 2L .. 4PL + 3L-1]: forget gate
     *        - [4PL + 3L .. 4PL + 4L-1]: output gate
     *    ~ internal weights (from other cells in the same layer):
     *        - [4(P+1)L + 0   .. 4(P+1)L + LL-1]:  net input
     *        - [4(P+1)L + LL  .. 4(P+1)L + 2LL-1]: input gate
     *        - [4(P+1)L + 2LL .. 4(P+1)L + 3LL-1]: forget gate
     *        - [4(P+1)L + 3LL .. 4(P+1)L + 4LL-1]: output gate
     *    ~ peephole weights (from cell state to all gates in the same cell):
     *        - [4(P+1+L)L + 0   .. 4(P+1+L)L + L-1]:  input gate
     *        - [4(P+1+L)L + LL  .. 4(P+1+L)L + 2L-1]: forget gate
     *        - [4(P+1+L)L + 2LL .. 4(P+1+L)L + 3L-1]: output gate
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/

    class LstmLayerCL : public TrainableLayerCL
    {


        struct weight_matrices_t {
            helpers::MatrixCL niInput;
            helpers::MatrixCL igInput;
            helpers::MatrixCL fgInput;
            helpers::MatrixCL ogInput;
            helpers::MatrixCL niInternal;
            helpers::MatrixCL igInternal;
            helpers::MatrixCL fgInternal;
            helpers::MatrixCL ogInternal;
        };

        struct timestep_matrices_t {
            helpers::MatrixCL tmpOutputs;
            helpers::MatrixCL tmpOutputErrors;
            helpers::MatrixCL niActs;
            helpers::MatrixCL igActs;
            helpers::MatrixCL fgActs;
            helpers::MatrixCL ogActs;
            helpers::MatrixCL niDeltas;
            helpers::MatrixCL igDeltas;
            helpers::MatrixCL fgDeltas;
            helpers::MatrixCL ogDeltas;
        };

        struct forward_backward_info_t {
            d_mem_real tmpOutputs;
            d_mem_real tmpOutputErrors;
            d_mem_real cellStates;
            d_mem_real cellStateErrors;
            d_mem_real niActs;
            d_mem_real igActs;
            d_mem_real fgActs;
            d_mem_real ogActs;
            d_mem_real niDeltas;
            d_mem_real igDeltas;
            d_mem_real fgDeltas;
            d_mem_real ogDeltas;

            helpers::MatrixCL niActsMatrixCL;
            helpers::MatrixCL igActsMatrixCL;
            helpers::MatrixCL fgActsMatrixCL;
            helpers::MatrixCL ogActsMatrixCL;
            helpers::MatrixCL niDeltasMatrixCL;
            helpers::MatrixCL igDeltasMatrixCL;
            helpers::MatrixCL fgDeltasMatrixCL;
            helpers::MatrixCL ogDeltasMatrixCL;

            weight_matrices_t                weightMatrices;
            weight_matrices_t                weightUpdateMatrices;
            std::vector<timestep_matrices_t> timestepMatrices;
        };

    public:
        const bool m_isBidirectional;
        


        int _rawNiBiasWeights_Offset;
        int _rawIgBiasWeights_Offset;
        int _rawFgBiasWeights_Offset;
        int _rawOgBiasWeights_Offset;
        int _rawIgPeepholeWeights_Offset;
        int _rawFgPeepholeWeights_Offset;
        int _rawOgPeepholeWeights_Offset;

        forward_backward_info_t m_fw;
        forward_backward_info_t m_bw;

        helpers::MatrixCL m_precLayerCLOutputsMatrixCL;

    public:
        /**
         * Constructs the LayerCL
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param weightsSection The weights section of the JSON configuration
         * @param precedingLayerCL The layer preceding this one
         * @param bidirectional  Wether the layer should be bidirectional or unidirectional
         */
        LstmLayerCL(
            const helpers::JsonValue &layerChild, 
            const helpers::JsonValue &weightsSection,
            LayerCL           &precedingLayerCL,
            bool                      bidirectional = false
            );

        /**
         * Destructs the LayerCL
         */
        virtual ~LstmLayerCL();

        /**
         * @see LayerCL::type()
         */
        virtual const std::string& type() const;

        /**
         * Returns true if the layer is bidirectional
         *
         * @return True if the layer is bidirectional
         */
        bool isBidirectional() const;

        /**
         * Returns the cell states
         *
         * @return The cell states
         */
        const d_mem_real& cellStates() const;

        /**
         * Returns the cell state errors
         *
         * @return The cell state errors
         */
        const d_mem_real& cellStateErrors() const;

        /**
         * Returns the net input activations
         *
         * @return The net input activations
         */
        const d_mem_real& netInputActs() const;

        /**
         * Returns the net input activation deltas
         *
         * @return The net input activation deltas
         */
        const d_mem_real& netInputDeltas() const;

        /**
         * Returns the input gate activations
         *
         * @return The input gate activations
         */
        const d_mem_real& inputGateActs() const;

        /**
         * Returns the input gate deltas
         *
         * @return The input gate deltas
         */
        const d_mem_real& inputGateDeltas() const;

        /**
         * Returns the forget gate activations
         *
         * @return The forget gate activations
         */
        const d_mem_real& forgetGateActs() const;

        /**
         * Returns the forget gate deltas
         *
         * @return The forget gate deltas
         */
        const d_mem_real& forgetGateDeltas() const;

        /**
         * Returns the output gate activations
         *
         * @return The output gate activations
         */
        const d_mem_real& outputGateActs() const;

        /**
         * Returns the output gate deltas
         *
         * @return The output gate deltas
         */
        const d_mem_real& outputGateDeltas() const;

        /**
         * @see LayerCL::loadSequences
         */
        virtual void loadSequences(const data_sets::DataSetFractionCL &fraction);

        /**
         * @see LayerCL::computeForwardPass()
         */
        virtual void computeForwardPass();

         /**
         * @see LayerCL::computeBackwardPass()
         */
        virtual void computeBackwardPass();
    };

} // namespace layers


#endif // LAYERS_LSTMLAYER_HPP
