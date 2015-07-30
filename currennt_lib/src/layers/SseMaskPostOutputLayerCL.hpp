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


#ifndef LAYERS_WFPOSTOUTPUTLAYERCL_HPP
#define LAYERS_WFPOSTOUTPUTLAYERCL_HPP

#include "PostOutputLayerCL.hpp"


namespace layers {

    /******************************************************************************************//**
     * This layer is used as the very last layer which store the target outputs and calculates the
     * error between the target outputs and the actual outputs
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/

    class SseMaskPostOutputLayerCL : public PostOutputLayerCL
    {
    public:
        /**
         * Constructs the LayerCL
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param precedingLayerCL The layer preceding this one
         */
        SseMaskPostOutputLayerCL(
            const helpers::JsonValue &layerChild, 
            LayerCL &precedingLayerCL
            );

        /**
         * Destructs the LayerCL
         */
        virtual ~SseMaskPostOutputLayerCL();

        /**
         * @see LayerCL::type()
         */
        virtual const std::string& type() const;

        /**
         * @see PostOutputLayerCL::calculateError()
         */
        virtual real_t calculateError();

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


#endif // LAYERS_WFPOSTOUTPUTLAYERCL_HPP
