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
#   pragma warning (disable: 4244)
#endif

#include "TrainableLayerCL.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../Configuration.hpp"

#include <stdexcept>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>




namespace layers {


     d_mem_real& TrainableLayerCL::_weightUpdates()
    {
        return m_weightUpdates;
    }


    TrainableLayerCL::TrainableLayerCL(const helpers::JsonValue &layerChild, const helpers::JsonValue &weightsSection,
                                            int inputWeightsPerBlock, int internalWeightsPerBlock, LayerCL &precedingLayerCL)
        : LayerCL           (layerChild, precedingLayerCL.parallelSequences(), precedingLayerCL.maxSeqLength())
        , m_precedingLayerCL         (precedingLayerCL)
        , m_inputWeightsPerBlock   (inputWeightsPerBlock)
        , m_internalWeightsPerBlock(internalWeightsPerBlock)
        , m_bias                   (layerChild->HasMember("bias") ? static_cast<real_t>((*layerChild)["bias"].GetDouble()) : 0)
        , m_learningRate           (layerChild->HasMember("learningRate") ? static_cast<real_t>((*layerChild)["learningRate"].GetDouble()) : -1)
    {
        //std::cout << "Creating layer " << this->name() << std::endl;
        // check if the bias value exists
        if (!layerChild->HasMember("bias"))
            throw std::runtime_error(std::string("Missing value 'bias' in layer '") + this->name() + "'");

        // extract the weights if they are given in the network file
        std::vector<real_t> weights;

        if (weightsSection.isValid() && weightsSection->HasMember(this->name().c_str())) {
            if (!weightsSection->HasMember(this->name().c_str()))
                throw std::runtime_error(std::string("Missing weights section for layer '") + this->name() + "'");
            const rapidjson::Value &weightsChild = (*weightsSection)[this->name().c_str()];
            if (!weightsChild.IsObject())
                throw std::runtime_error(std::string("Weights section for layer '") + this->name() + "' is not an object");

            if (!weightsChild.HasMember("input") || !weightsChild["input"].IsArray())
                throw std::runtime_error(std::string("Missing array 'weights/") + this->name() + "/input'");
            if (!weightsChild.HasMember("bias") || !weightsChild["bias"].IsArray())
                throw std::runtime_error(std::string("Missing array 'weights/") + this->name() + "/bias'");
            if (!weightsChild.HasMember("internal") || !weightsChild["internal"].IsArray())
                throw std::runtime_error(std::string("Missing array 'weights/") + this->name() + "/internal'");
        
            const rapidjson::Value &inputWeightsChild    = weightsChild["input"];
            const rapidjson::Value &biasWeightsChild     = weightsChild["bias"];
            const rapidjson::Value &internalWeightsChild = weightsChild["internal"];

            if (inputWeightsChild.Size() != this->size() * inputWeightsPerBlock * m_precedingLayerCL.size())
                throw std::runtime_error(std::string("Invalid number of input weights for layer '") + this->name() + "'");
            if (biasWeightsChild.Size() != this->size() * inputWeightsPerBlock)
                throw std::runtime_error(std::string("Invalid number of bias weights for layer '") + this->name() + "'");
            if (internalWeightsChild.Size() != this->size() * internalWeightsPerBlock)
                throw std::runtime_error(std::string("Invalid number of internal weights for layer '") + this->name() + "'");

            weights.reserve(inputWeightsChild.Size() + biasWeightsChild.Size() + internalWeightsChild.Size());

            for (rapidjson::Value::ConstValueIterator it = inputWeightsChild.Begin(); it != inputWeightsChild.End(); ++it)
                weights.push_back(static_cast<real_t>(it->GetDouble()));
            for (rapidjson::Value::ConstValueIterator it = biasWeightsChild.Begin(); it != biasWeightsChild.End(); ++it)
                weights.push_back(static_cast<real_t>(it->GetDouble()));
            for (rapidjson::Value::ConstValueIterator it = internalWeightsChild.Begin(); it != internalWeightsChild.End(); ++it)
                weights.push_back(static_cast<real_t>(it->GetDouble()));
        }
        // create random weights if no weights are given in the network file
        else {
            weights.resize(this->size() * (inputWeightsPerBlock * (m_precedingLayerCL.size() + 1) + internalWeightsPerBlock));

            const Configuration &config = Configuration::instance();

            static boost::mt19937 *gen = NULL;
            if (!gen) {
                gen = new boost::mt19937;
                gen->seed(config.randomSeed());
            }
            
            if (config.weightsDistributionType() == Configuration::DISTRIBUTION_UNIFORM) {
                real_t range = config.weightsDistributionUniformMax() - config.weightsDistributionUniformMin();
                boost::random::uniform_real_distribution<real_t> dist(0, range);
                for (size_t i = 0; i < weights.size(); ++i)
                    weights[i] = dist(*gen) + config.weightsDistributionUniformMin();
            }
            else {
                boost::random::normal_distribution<real_t> dist(config.weightsDistributionNormalMean(), config.weightsDistributionNormalSigma());
                for (size_t i = 0; i < weights.size(); ++i)
                    weights[i] = dist(*gen);
            }
        }

        SystemCL::copy_real(m_weights , weights);
        SystemCL::copy_real(m_weightUpdates, weights);;


        // resize the output errors vector
        //m_outputErrors = std::vector<real_t>(this->_outputs().size(), (real_t)0);
    }


    TrainableLayerCL::~TrainableLayerCL()
    {
    }


    LayerCL& TrainableLayerCL::precedingLayerCL()
    {
        return m_precedingLayerCL;
    }


    const LayerCL& TrainableLayerCL::precedingLayerCL() const
    {
        return m_precedingLayerCL;
    }


    real_t TrainableLayerCL::bias() const
    {
        return m_bias;
    }


    real_t TrainableLayerCL::learningRate() const
    {
        return m_learningRate;
    }

/*
    typename d_mem_real& TrainableLayerCL::outputErrors()
    {
        return m_outputErrors;
    }*/


     d_mem_real& TrainableLayerCL::weights()
    {
        return m_weights;
    }


    const  d_mem_real& TrainableLayerCL::weights() const
    {
        return m_weights;
    }


    const  d_mem_real& TrainableLayerCL::weightUpdates() const
    {
        return m_weightUpdates;
    }


    void TrainableLayerCL::injectWeightNoise(real_t sigma)
    {
        // generate vector of weight noise on the host
        // note: RNG is sequential, so we can't parallelize ...
        static boost::mt19937 *gen = NULL;
        if (!gen) {
            gen = new boost::mt19937;
            gen->seed(Configuration::instance().randomSeed());
        }
        boost::normal_distribution<real_t> dist(0.0f, sigma);
        std::vector<real_t> weightNoise(weights().size);
        for (int i = 0; i < weightNoise.size(); ++i) {
            weightNoise[i] = dist(*gen);
        }

        // copy weight noise to device
        d_mem_real weightNoiseD;
        SystemCL::malloc_real(weightNoiseD, weightNoise.size());

        SystemCL::copy_real(weightNoiseD, weightNoise);


        // weights = weights + weightNoiseD
        // add weight noise to device vector of weights
        SystemCL::r_plus(weights().mem, weightNoiseD.mem, weights().size);

    }


    void TrainableLayerCL::exportWeights(const helpers::JsonValue &weightsObject, const helpers::JsonAllocator &allocator) const
    {
        if (!weightsObject->IsObject())
            throw std::runtime_error("The JSON value is not an object");

        // do nothing if we don't have any weights
        if (m_weights.size == 0)
            return;

        // create and fill the weight arrays
        rapidjson::Value inputWeightsArray(rapidjson::kArrayType);
        int inputWeightsCount = this->size() * m_inputWeightsPerBlock * m_precedingLayerCL.size();
        inputWeightsArray.Reserve(inputWeightsCount, allocator);
        std::vector<real_t> h_weights = SystemCL::copy_real(m_weights);
        for (int i = 0; i < inputWeightsCount; ++i)
            inputWeightsArray.PushBack(h_weights[i], allocator);

        rapidjson::Value biasWeightsArray(rapidjson::kArrayType);
        int biasWeightsCount = this->size() * m_inputWeightsPerBlock;
        biasWeightsArray.Reserve(biasWeightsCount, allocator);
        for (int i = 0; i < biasWeightsCount; ++i)
            biasWeightsArray.PushBack(h_weights[inputWeightsCount + i], allocator);

        rapidjson::Value internalWeightsArray(rapidjson::kArrayType);
        int internalWeightsCount = this->size() * m_internalWeightsPerBlock;
        internalWeightsArray.Reserve(internalWeightsCount, allocator);
        for (int i = 0; i < internalWeightsCount; ++i)
            internalWeightsArray.PushBack(h_weights[inputWeightsCount + biasWeightsCount + i], allocator);

        // create and fill the weights subsection
        rapidjson::Value weightsSection(rapidjson::kObjectType);
        weightsSection.AddMember("input",    inputWeightsArray,    allocator);
        weightsSection.AddMember("bias",     biasWeightsArray,     allocator);
        weightsSection.AddMember("internal", internalWeightsArray, allocator);

        // add the weights section tot he weights object
        weightsObject->AddMember(this->name().c_str(), weightsSection, allocator);
    }


    void TrainableLayerCL::exportLayerCL(const helpers::JsonValue &layersArray, const helpers::JsonAllocator &allocator) const
    {
        LayerCL::exportLayerCL(layersArray, allocator);
        (*layersArray)[layersArray->Size() - 1].AddMember("bias", m_bias, allocator);
    }




} // namespace layers
