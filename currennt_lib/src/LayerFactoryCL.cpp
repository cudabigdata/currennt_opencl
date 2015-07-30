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

#include "LayerFactoryCL.hpp"

#include "layers/InputLayerCL.hpp"
#include "layers/FeedForwardLayerCL.hpp"
#include "layers/SoftmaxLayerCL.hpp"
#include "layers/LstmLayerCL.hpp"
#include "layers/SsePostOutputLayerCL.hpp"
#include "layers/RmsePostOutputLayerCL.hpp"
#include "layers/CePostOutputLayerCL.hpp"
#include "layers/SseMaskPostOutputLayerCL.hpp"
#include "layers/WeightedSsePostOutputLayerCL.hpp"
#include "layers/BinaryClassificationLayerCL.hpp"
#include "layers/MulticlassClassificationLayerCL.hpp"


#include <stdexcept>



layers::LayerCL* LayerFactoryCL::createLayerCL(
		const std::string &layerType, const helpers::JsonValue &layerChild,
        const helpers::JsonValue &weightsSection, int parallelSequences, 
        int maxSeqLength, layers::LayerCL *precedingLayerCL)
{
    using namespace layers;
//    using namespace activation_functions;

    if (layerType == "input")
    	return new InputLayerCL(layerChild, parallelSequences, maxSeqLength);
    else if (layerType == "feedforward_tanh")
    	return new FeedForwardLayerCL(layerChild, weightsSection, *precedingLayerCL,"TANH");
    else if (layerType == "feedforward_logistic")
    	return new FeedForwardLayerCL(layerChild, weightsSection, *precedingLayerCL, "LOGISTIC");
    else if (layerType == "feedforward_identity")
    	return new FeedForwardLayerCL(layerChild, weightsSection, *precedingLayerCL, "IDENTITY");
    else if (layerType == "softmax")
    	return new SoftmaxLayerCL(layerChild, weightsSection, *precedingLayerCL,"IDENTITY");
    else if (layerType == "lstm")
    	return new LstmLayerCL(layerChild, weightsSection, *precedingLayerCL, false);
    else if (layerType == "blstm")
    	return new LstmLayerCL(layerChild, weightsSection, *precedingLayerCL, true);
    else if (layerType == "sse" || layerType == "weightedsse" || layerType == "rmse" || layerType == "ce" || layerType == "wf" || layerType == "binary_classification" || layerType == "multiclass_classification") {

        if (layerType == "sse")
    	    return new SsePostOutputLayerCL(layerChild, *precedingLayerCL);
        else if (layerType == "weightedsse")
    	    return new WeightedSsePostOutputLayerCL(layerChild, *precedingLayerCL);
        else if (layerType == "rmse")
            return new RmsePostOutputLayerCL(layerChild, *precedingLayerCL);
        else if (layerType == "ce")
            return new CePostOutputLayerCL(layerChild, *precedingLayerCL);
        if (layerType == "sse_mask" || layerType == "wf") // wf provided for compat. with dev. version
    	    return new SseMaskPostOutputLayerCL(layerChild, *precedingLayerCL);
        else if (layerType == "binary_classification")
    	    return new BinaryClassificationLayerCL(layerChild, *precedingLayerCL);
        else // if (layerType == "multiclass_classification")
    	    return new MulticlassClassificationLayerCL(layerChild, *precedingLayerCL);
    }
    else
        throw std::runtime_error(std::string("Unknown layer type '") + layerType + "'");
}



