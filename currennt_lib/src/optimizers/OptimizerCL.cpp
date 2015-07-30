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

#include "OptimizerCL.hpp"
#include "../layers/TrainableLayerCL.hpp"
#include "../layers/BinaryClassificationLayerCL.hpp"
#include "../layers/MulticlassClassificationLayerCL.hpp"
#include "../Configuration.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../layers/LstmLayerCL.hpp"
#include <limits>



namespace optimizers {


    real_t OptimizerCL::_processDataSetCL(data_sets::DataSetCL &ds, bool calcWeightUpdates, real_t *classError)
    {
        // process all data set fractions
        real_t error = 0;
        *classError = (real_t) ds.totalTimesteps();

        boost::shared_ptr<data_sets::DataSetFractionCL> frac;
        bool firstFraction = true;
        while ((frac = ds.getNextFraction())) {
            // compute forward pass and calculate the error
            m_neuralNetwork.loadSequences(*frac);
            m_neuralNetwork.computeForwardPass();
            error += m_neuralNetwork.calculateError();

            if (dynamic_cast<layers::BinaryClassificationLayerCL*>(&m_neuralNetwork.postOutputLayerCL()))
                *classError -= (real_t)static_cast<layers::BinaryClassificationLayerCL&>(m_neuralNetwork.postOutputLayerCL()).countCorrectClassifications();
            if (dynamic_cast<layers::MulticlassClassificationLayerCL*>(&m_neuralNetwork.postOutputLayerCL()))
                *classError -= (real_t)static_cast<layers::MulticlassClassificationLayerCL&>(m_neuralNetwork.postOutputLayerCL()).countCorrectClassifications();
            
            if (calcWeightUpdates) {
                // weight noise:
            	int size = m_neuralNetwork.layers().size();
                std::vector<std::vector<real_t> > origWeights(size);
                if (Configuration::instance().weightNoiseSigma() > 0) {
                    for (size_t i = 1; i < m_neuralNetwork.layers().size()-1; ++i) {
                        layers::TrainableLayerCL *layer = dynamic_cast<layers::TrainableLayerCL*>(m_neuralNetwork.layers()[i].get());
                        if (layer) {
                            origWeights[i] = SystemCL::copy_real(layer->weights());
                            layer->injectWeightNoise(Configuration::instance().weightNoiseSigma());
                        }
                    }
                }
                // compute the backward pass and accumulate the weight updates
                m_neuralNetwork.computeBackwardPass();

                for (size_t i = 1; i < m_neuralNetwork.layers().size()-1; ++i) {
                    layers::TrainableLayerCL *layer = dynamic_cast<layers::TrainableLayerCL*>(m_neuralNetwork.layers()[i].get());
                    if (!layer)
                        continue;

                    if (!firstFraction && !Configuration::instance().hybridOnlineBatch())
                    	SystemCL::r_plus(m_curWeightUpdates[i].mem,layer->weightUpdates().mem, layer->weightUpdates().size);
                      //  thrust::transform(layer->weightUpdates().begin(), layer->weightUpdates().end(), m_curWeightUpdates[i].begin(), m_curWeightUpdates[i].begin(), thrust::plus<real_t>());
                    else
                    	SystemCL::copy_real(m_curWeightUpdates[i],layer->weightUpdates());


                    // restore old weights before update in case of weight noise
                    if (Configuration::instance().weightNoiseSigma() > 0.0)
                    	 SystemCL::copy_real(layer->weights(), origWeights[i]);
                }

                // update weights for hybrid online/batch learning
                if (Configuration::instance().hybridOnlineBatch())
                    _updateWeights();
            }

            firstFraction = false;
        }

        // update weights for batch learning
        if (calcWeightUpdates && !Configuration::instance().hybridOnlineBatch())
            _updateWeights();

        // normalize the errors
        error /= ds.totalSequences();
        *classError /= (real_t)ds.totalTimesteps();

        return error;
    }


    void OptimizerCL::_exportWeights(const helpers::JsonDocument &jsonDoc, const char *arrayName, const std::vector<d_mem_real> &weights)
    {
        rapidjson::Value weightsArray(rapidjson::kArrayType);
        weightsArray.Reserve((rapidjson::SizeType)weights.size(), jsonDoc->GetAllocator());

        for (size_t i = 0; i < weights.size(); ++i) {
            rapidjson::Value v(rapidjson::kArrayType);
            std::vector<real_t> w =  SystemCL::copy_real( weights[i]);
            v.Reserve((rapidjson::SizeType)w.size(), jsonDoc->GetAllocator());
            for (size_t j = 0; j < w.size(); ++j)
                v.PushBack(w[j], jsonDoc->GetAllocator());
            weightsArray.PushBack(v, jsonDoc->GetAllocator());
        }

        jsonDoc->AddMember(arrayName, weightsArray, jsonDoc->GetAllocator());
    }


    void OptimizerCL::_importWeights(const helpers::JsonDocument &jsonDoc, const char *arrayName, std::vector<d_mem_real> *weights)
    {
        if (!jsonDoc->HasMember(arrayName) || !(*jsonDoc)[arrayName].IsArray())
            throw std::runtime_error(std::string("Array '") + arrayName + "' is missing or has the wrong type");

        if ((*jsonDoc)[arrayName].Size() != (rapidjson::SizeType)weights->size())
            throw std::runtime_error(std::string("Array '") + arrayName + "' has a wrong size");

        int i = 0;
        for (rapidjson::Value::ConstValueIterator it = (*jsonDoc)[arrayName].Begin(); it != (*jsonDoc)[arrayName].End(); ++it) {
            if (!it->IsArray())
                throw std::runtime_error(std::string("Object in '") + arrayName + "' is not an array");
            if (it->Size() != (rapidjson::SizeType)(*weights)[i].size)
                throw std::runtime_error(std::string("Subarray in '") + arrayName + "' has a wrong size");

            std::vector<real_t> w;
            w.reserve(it->Size());
            for (rapidjson::Value::ConstValueIterator it2 = it->Begin(); it2 != it->End(); ++it2)
                w.push_back((real_t)it2->GetDouble());

            SystemCL::copy_real( (*weights)[i], w);

            ++i;
        }
    }


    void OptimizerCL::_storeWeights()
    {
        for (size_t i = 1; i < m_neuralNetwork.layers().size() - 1; ++i) {
            layers::TrainableLayerCL *layer = dynamic_cast<layers::TrainableLayerCL*>(m_neuralNetwork.layers()[i].get());
            if (layer) {
            	SystemCL::copy_real( m_bestWeights[i],layer->weights());
            	SystemCL::print(m_bestWeights[i]);
            }
            	//thrust::copy(layer->weights().begin(), layer->weights().end(), m_bestWeights[i].begin());
        }
    }


    void OptimizerCL::_restoreWeights()
    {
        for (size_t i = 1; i < m_neuralNetwork.layers().size() - 1; ++i) {
        	layers::TrainableLayerCL *layer = dynamic_cast<layers::TrainableLayerCL*>(m_neuralNetwork.layers()[i].get());
            if (layer)
            	SystemCL::copy_real(layer->weights(), m_bestWeights[i]);
            //	thrust::copy(m_bestWeights[i].begin(), m_bestWeights[i].end(), layer->weights().begin());
        }
    }


    NeuralNetworkCL& OptimizerCL::_neuralNetwork()
    {
        return m_neuralNetwork;
    }


    const std::vector< d_mem_real>& OptimizerCL::_curWeightUpdates() const
    {
        return m_curWeightUpdates;
    }


    OptimizerCL::OptimizerCL(NeuralNetworkCL &neuralNetwork, data_sets::DataSetCL &trainingSet,
                                   data_sets::DataSetCL &validationSet, data_sets::DataSetCL &testSet,
                                   int maxEpochs, int maxEpochsNoBest, int validateEvery, int testEvery)
        : m_neuralNetwork             (neuralNetwork)
        , m_trainingSet               (trainingSet)
        , m_validationSet             (validationSet)
        , m_testSet                   (testSet)
        , m_maxEpochs                 (maxEpochs)
        , m_maxEpochsNoBest           (maxEpochsNoBest)
        , m_validateEvery             (validateEvery)
        , m_testEvery                 (testEvery)
        , m_finished                  (false)
        , m_curEpoch                  (0)
        , m_epochsSinceLowestError    (0)
        , m_lowestValidationError     (std::numeric_limits<real_t>::max())
        , m_curTrainingError          (std::numeric_limits<real_t>::max())
        , m_curValidationError        (std::numeric_limits<real_t>::max())
        , m_curTestError              (std::numeric_limits<real_t>::max())
        , m_curValidationClassError   (0)
        , m_curTrainingClassError     (0)
        , m_curTestClassError         (0)
    {
        // initialize the best weights vectors
        m_bestWeights.resize(m_neuralNetwork.layers().size());
        for (size_t i = 1; i < m_neuralNetwork.layers().size()-1; ++i) {
        	layers::TrainableLayerCL *layer = dynamic_cast<layers::TrainableLayerCL*>(m_neuralNetwork.layers()[i].get());
            if (layer)
                SystemCL::assign_real(m_bestWeights[i] ,layer->weights());



        }

        // initialize the current weight updates vectors
        m_curWeightUpdates = m_bestWeights;


    }


    OptimizerCL::~OptimizerCL()
    {
    }


    bool OptimizerCL::finished() const
    {
        return m_finished;
    }


    int OptimizerCL::currentEpoch() const
    {
        return m_curEpoch;
    }


    real_t OptimizerCL::lowestValidationError() const
    {
        return m_lowestValidationError;
    }


    int OptimizerCL::epochsSinceLowestValidationError() const
    {
        return m_epochsSinceLowestError;
    }


    real_t OptimizerCL::curTrainingError() const
    {
        return m_curTrainingError;
    }


    real_t OptimizerCL::curValidationError() const
    {
        return m_curValidationError;
    }


    real_t OptimizerCL::curTestError() const
    {
        return m_curTestError;
    }


    real_t OptimizerCL::curTrainingClassError() const
    {
        return m_curTrainingClassError;
    }


    real_t OptimizerCL::curValidationClassError() const
    {
        return m_curValidationClassError;
    }


    real_t OptimizerCL::curTestClassError() const
    {
        return m_curTestClassError;
    }


    bool OptimizerCL::train()
    {
        if (!m_finished) {
            ++m_curEpoch;
            // train one epoch and update the weights
            m_curTrainingError = _processDataSetCL(m_trainingSet, true, &m_curTrainingClassError);

            // calculate the validation error and store the weights if we a new lowest error
            if (!m_validationSet.empty() && m_curEpoch % m_validateEvery == 0) {
                m_curValidationError = _processDataSetCL(m_validationSet, false, &m_curValidationClassError);
                
                if (m_curValidationError < m_lowestValidationError) {
                    m_lowestValidationError  = m_curValidationError;
                    m_epochsSinceLowestError = 0;

                    _storeWeights();
                }
                else {
                    m_epochsSinceLowestError += m_validateEvery;
                }
            }
            else if (m_validationSet.empty()) {
                m_epochsSinceLowestError = 0;
                _storeWeights();
            }

            // calculate the test error
            if (!m_testSet.empty() && m_curEpoch % m_testEvery == 0)
                m_curTestError = _processDataSetCL(m_testSet, false, &m_curTestClassError);

            // check if we did not get a new lowest error for some training epochs 
            // or if we reached the maximum number of training epochs
            if (m_epochsSinceLowestError >= m_maxEpochsNoBest || (m_maxEpochs >= 0 && m_curEpoch >= m_maxEpochs)) {
                _restoreWeights();
                m_finished = true;
            }
        }

        return m_finished;
    }


    void OptimizerCL::exportState(const helpers::JsonDocument &jsonDoc) const
    {
        jsonDoc->AddMember("optimizer_finished",                   m_finished,                jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_epoch",                  m_curEpoch,                jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_epochs_since_lowest_error",  m_epochsSinceLowestError,  jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_lowest_validation_error",    m_lowestValidationError,   jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_training_error",         m_curTrainingError,        jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_validation_error",       m_curValidationError,      jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_test_error",             m_curTestError,            jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_training_class_error",   m_curTrainingClassError,   jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_validation_class_error", m_curValidationClassError, jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_test_class_error",       m_curTestClassError,       jsonDoc->GetAllocator());

        _exportWeights(jsonDoc, "optimizer_best_weights", m_bestWeights);
    }


    void OptimizerCL::importState(const helpers::JsonDocument &jsonDoc)
    {
        m_finished                = helpers::checkedJsonGet<bool  >(*jsonDoc, "optimizer_finished");
        m_curEpoch                = helpers::checkedJsonGet<int   >(*jsonDoc, "optimizer_cur_epoch");
        m_epochsSinceLowestError  = helpers::checkedJsonGet<int   >(*jsonDoc, "optimizer_epochs_since_lowest_error");
        m_lowestValidationError   = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_lowest_validation_error");
        m_curTrainingError        = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_training_error");
        m_curValidationError      = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_validation_error");
        m_curTestError            = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_test_error");
        m_curTrainingClassError   = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_training_class_error");
        m_curValidationClassError = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_validation_class_error");
        m_curTestClassError       = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_test_class_error");

        _importWeights(jsonDoc, "optimizer_best_weights", &m_bestWeights);
    }




} // namespace optimizers
