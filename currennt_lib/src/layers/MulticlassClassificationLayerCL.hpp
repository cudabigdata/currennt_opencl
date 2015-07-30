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

#ifndef LAYERS_MULTICLASSCLASSIFICATIONLAYERCL_HPP
#define LAYERS_MULTICLASSCLASSIFICATIONLAYERCL_HPP

#include "PostOutputLayerCL.hpp"


namespace layers {

    /******************************************************************************************//**
     * Post output layer for multiclass classification
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/

    class MulticlassClassificationLayerCL : public PostOutputLayerCL
    {


    private:
        d_mem_int m_patTargetClasses;

    public:
        /**
         * Constructs the LayerCL
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param precedingLayerCL The layer preceding this one
         */
        MulticlassClassificationLayerCL(
            const helpers::JsonValue &layerChild, 
            LayerCL  &precedingLayerCL
            );

        /**
         * Destructs the LayerCL
         */
        virtual ~MulticlassClassificationLayerCL();

        /**
         * Counts correct classifications
         *
         * @return Number of correct classifications
         */
        int countCorrectClassifications();

        /**
         * @see LayerCL::type()
         */
        virtual const std::string& type() const;

        /**
         * @see LayerCL::loadSequences()
         */
        virtual void loadSequences(const data_sets::DataSetFractionCL &fraction);

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


#endif // LAYERS_MULTICLASSCLASSIFICATIONLAYERCL_HPP
