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

#ifndef LAYERS_POSTOUTPUTLAYERCL_HPP
#define LAYERS_POSTOUTPUTLAYERCL_HPP

#include "TrainableLayerCL.hpp"


namespace layers {

    /******************************************************************************************//**
     * This layer is used as the very last layer which store the target outputs and calculates the
     * error between the target outputs and the actual outputs
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/

    class PostOutputLayerCL : public LayerCL
    {

    private:
        LayerCL &m_precedingLayerCL;

    protected:
        d_mem_real& _targets();
        d_mem_real& _actualOutputs();
        d_mem_real& _outputErrors();

    public:
        /**
         * Constructs the LayerCL
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param precedingLayerCL The layer preceding this one
         * @param createOutputs  If false, then the outputs vector will be left empty
         */
        PostOutputLayerCL(
            const helpers::JsonValue &layerChild, 
            LayerCL  &precedingLayerCL,
            int requiredSize,
            bool                      createOutputs = true
            );

        /**
         * Destructs the LayerCL
         */
        virtual ~PostOutputLayerCL();

        /**
         * @see LayerCL::loadSequences()
         */
        virtual void loadSequences(const data_sets::DataSetFractionCL &fraction);

        /**
         * Computes the error with respect to the target outputs
         *
         * @return The error 
         */
        virtual real_t calculateError() =0;

    };

} // namespace layers


#endif // LAYERS_POSTOUTPUTLAYERCL_HPP
