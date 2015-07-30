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
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif

#include "SteepestDescentOptimizerCL.hpp"
#include "../layers/TrainableLayerCL.hpp"
#include "../layers/LstmLayerCL.hpp"
#include "../rapidjson/document.h"



namespace optimizers {


    void SteepestDescentOptimizerCL::_updateWeights()
    {
        for (size_t i = 1; i < this->_neuralNetwork().layers().size()-1; ++i) {
        	layers::TrainableLayerCL *layer = dynamic_cast<layers::TrainableLayerCL*>(this->_neuralNetwork().layers()[i].get());
            if (!layer)
                continue;

            float learningRate = m_learningRate;
            if (layer->learningRate() >= 0.0)
                learningRate = layer->learningRate();
       //     std::cout << "layer " << layer->name() << ": learning rate " << learningRate << std::endl;


            int n = (int)layer->weights().size;


            SystemCL::sdo_updateWeightFn(m_momentum, learningRate,layer->weights(),
            		this->_curWeightUpdates()[i],m_weightDeltas[i],  layer->weights(), n);

        }
    }


    SteepestDescentOptimizerCL::SteepestDescentOptimizerCL(
        NeuralNetworkCL &neuralNetwork, data_sets::DataSetCL &trainingSet, data_sets::DataSetCL &validationSet,
        data_sets::DataSetCL &testSet, int maxEpochs, int maxEpochsNoBest, int validateEvery, int testEvery,
        real_t learningRate, real_t momentum)
        : OptimizerCL(neuralNetwork, trainingSet, validationSet, testSet, maxEpochs, maxEpochsNoBest, validateEvery, testEvery)
        , m_learningRate    (learningRate)
        , m_learningRateFirst(learningRate)
        , m_momentum        (momentum)
    {
        // intialize the weight deltas vectors with zeros
    	int size = this->_curWeightUpdates().size();
    	m_weightDeltas.resize(size);
        for (int i = 0 ; i < this->_curWeightUpdates().size(); i++){
        	d_mem_real * dm = new d_mem_real();
        	SystemCL::assign_real(*dm, this->_curWeightUpdates()[i]);
        	m_weightDeltas[i]= (*dm);
        }


        for (size_t i = 0; i < m_weightDeltas.size(); ++i){

        	SystemCL::fill(m_weightDeltas[i], 0);
        }


    }


    SteepestDescentOptimizerCL::~SteepestDescentOptimizerCL()
    {
    }


    void SteepestDescentOptimizerCL::exportState(const helpers::JsonDocument &jsonDoc) const
    {
        OptimizerCL::exportState(jsonDoc);

        OptimizerCL::_exportWeights(jsonDoc, "steepest_descent_optimizer_weight_deltas", m_weightDeltas);
    }


    void SteepestDescentOptimizerCL::importState(const helpers::JsonDocument &jsonDoc)
    {
        OptimizerCL::importState(jsonDoc);

        OptimizerCL::_importWeights(jsonDoc, "steepest_descent_optimizer_weight_deltas", &m_weightDeltas);
    }


    void SteepestDescentOptimizerCL::setLearningRateFirst(real_t learningRateFirst)
    {
        m_learningRateFirst = learningRateFirst;
    }


    // explicit template instantiations


} // namespace optimizers
