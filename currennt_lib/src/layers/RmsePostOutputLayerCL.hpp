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

#ifndef LAYERS_RMSEPOSTOUTPUTLAYERCL_HPP
#define LAYERS_RMSEPOSTOUTPUTLAYERCL_HPP

#include "PostOutputLayerCL.hpp"


namespace layers {

    /******************************************************************************************//**
     * Implements the Root Mean Squared Error (RMSE) objective function
     *
     * RMSE = sqrt((sum(x_i-z_i)^2)/N)
     * RMSE deriv = RMSE * (x_i-z_i)
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/

    class RmsePostOutputLayerCL : public PostOutputLayerCL
    {

    private:
        d_mem_real m_rmses; // contains the RMSE for each pattern

    public:
        /**
         * Constructs the LayerCL
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param precedingLayerCL The layer preceding this one
         */
        RmsePostOutputLayerCL(
            const helpers::JsonValue &layerChild, 
            LayerCL &precedingLayerCL
            );

        /**
         * Destructs the LayerCL
         */
        virtual ~RmsePostOutputLayerCL();

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


#endif // LAYERS_RMSEPOSTOUTPUTLAYER_HPP
