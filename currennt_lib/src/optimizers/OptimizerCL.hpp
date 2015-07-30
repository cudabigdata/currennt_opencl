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

#ifndef OPTIMIZERS_OPTIMIZERCL_HPP
#define OPTIMIZERS_OPTIMIZERCL_HPP

#include "../NeuralNetworkCL.hpp"
#include "../data_sets/DataSetCL.hpp"


namespace optimizers {

    /******************************************************************************************//**
     * Base class for weight optimizers
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/

    class OptimizerCL
    {


    public:
        NeuralNetworkCL &m_neuralNetwork;
        data_sets::DataSetCL     &m_trainingSet;
        data_sets::DataSetCL     &m_validationSet;
        data_sets::DataSetCL     &m_testSet;

        const int m_maxEpochs;
        const int m_maxEpochsNoBest;
        const int m_validateEvery;
        const int m_testEvery;

        bool   m_finished;
        int	   m_curEpoch;
        int	   m_epochsSinceLowestError;
        real_t m_lowestValidationError;
        real_t m_curTrainingError;
        real_t m_curValidationError;
        real_t m_curTestError;
        real_t m_curValidationClassError;
        real_t m_curTrainingClassError;
        real_t m_curTestClassError;

        std::vector<d_mem_real> m_curWeightUpdates;
        std::vector<d_mem_real> m_bestWeights;

    private:
        real_t _processDataSetCL(data_sets::DataSetCL &ds, bool calcWeightUpdates, real_t *classError);
        void _storeWeights();
        void _restoreWeights();

    public:
        static void _exportWeights(const helpers::JsonDocument &jsonDoc, const char *arrayName, const std::vector<d_mem_real> &weights);
        static void _importWeights(const helpers::JsonDocument &jsonDoc, const char *arrayName, std::vector<d_mem_real> *weights);
        NeuralNetworkCL& _neuralNetwork();
        const std::vector<d_mem_real>& _curWeightUpdates() const;
        virtual void _updateWeights() =0;

    public:
        /**
         * Constructs the optimizer
         *
         * @param neuralNetwork   The neural network to operate on
         * @param trainingSet     The set of training sequences
         * @param validationSet   The set of validation sequences
         * @param testSet         The set of test sequences
         * @param maxEpochs       The maximum total number of epochs to train
         * @param maxEpochsNoBest The number of epochs in which no new lowest error could be
         *                        achieved before training is stopped
         * @param validateEvery   After how many epochs the validation error shall be calculated
         * @param testEvery       After how many epochs the test error shall be calculated
         */
        OptimizerCL(
            NeuralNetworkCL &neuralNetwork,
            data_sets::DataSetCL     &trainingSet,
            data_sets::DataSetCL     &validationSet,
            data_sets::DataSetCL     &testSet,
            int maxEpochs, 
            int maxEpochsNoBest,
            int validateEvery,
            int testEvery
            );

        /**
         * Destructs the optimizer
         */
        virtual ~OptimizerCL();

        /**
         * Check if the training is finished
         *
         * @return True if the training is finished
         */
        bool finished() const;

        /**
         * Returns the current training epoch
         *
         * @return The current training epoch
         */
        int currentEpoch() const;

        /**
         * Returns the lowest error on the validation set
         *
         * @return The lowest error on the validation set
         */
        real_t lowestValidationError() const;

        /**
         * Returns the number of training epochs since the lowest error on the validation set
         *
         * @return The number of training epochs since the lowest error on the validation set
         */
        int epochsSinceLowestValidationError() const;

        /**
         * Returns the current training set error
         *
         * @return The current training set error
         */
        real_t curTrainingError() const;

        /**
         * Returns the current validation set error
         *
         * @return The current validation set error
         */
        real_t curValidationError() const;

        /**
         * Returns the current test set error
         *
         * @return The current test set error
         */
        real_t curTestError() const;

        /**
         * Returns the current training set classification error
         *
         * @return The current training set classification error
         */
        real_t curTrainingClassError() const;

        /**
         * Returns the current validation set classification error
         *
         * @return The current validation set classification error
         */
        real_t curValidationClassError() const;

        /**
         * Returns the current test set classification error
         *
         * @return The current test set classification error
         */
        real_t curTestClassError() const;

        /**
         * Optimizes the weights
         *
         * If either the maximum number of training epochs from the process configuration has been
         * reached or no new lowest error has been achieved since the last x epochs, the function
         * returnes true and the network will not be trained any further.
         *
         * @return True if the training is finished
         */
        bool train();

        /**
         * Writes the current state to a JSON tree
         *
         * @param jsonDoc The JSON document
         */
        virtual void exportState(const helpers::JsonDocument &jsonDoc) const;

        /**
         * Restores the state from a JSON tree
         *
         * @param jsonDoc The JSON document
         */
        virtual void importState(const helpers::JsonDocument &jsonDoc);
    };

} // namespace optimizers


#endif // OPTIMIZERS_OPTIMIZER_HPP
