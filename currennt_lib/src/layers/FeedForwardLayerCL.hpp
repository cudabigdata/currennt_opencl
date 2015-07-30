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

#ifndef LAYERS_FEEDFORWARDLAYERCL_HPP
#define LAYERS_FEEDFORWARDLAYERCL_HPP

#include "TrainableLayerCL.hpp"


namespace layers {

    /******************************************************************************************//**
     * Represents a feed forward layer in the neural network
     *
     * @param TDevice The computation device (Cpu or Gpu)
     * @param TActFn  The activation function to use
     *********************************************************************************************/

    class FeedForwardLayerCL : public TrainableLayerCL
    {
    private:
    	std::string activate_fun;
    public:
        /**
         * Constructs the LayerCL
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param weightsSection The weights section of the JSON configuration
         * @param precedingLayerCL The layer preceding this one
         */
        FeedForwardLayerCL(
            const helpers::JsonValue &layerChild, 
            const helpers::JsonValue &weightsSection,
            LayerCL           &precedingLayerCL,
			std::string a_fn

            );

        /**
         * Destructs the LayerCL
         */
        virtual ~FeedForwardLayerCL();

        /**
         * @see LayerCL::type()
         */
        virtual const std::string& type() const;

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


#endif // LAYERS_FEEDFORWARDLAYERCL_HPP
