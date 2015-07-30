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

#include "LstmLayerCL.hpp"

#include "../helpers/MatrixCL.hpp"
#include "../SystemCL.hpp"
#include  <stdexcept>





namespace layers {

    
    LstmLayerCL::LstmLayerCL(const helpers::JsonValue &layerChild,
                                  const helpers::JsonValue &weightsSection,
                                  LayerCL &precedingLayerCL,
                                  bool bidirectional)
        : TrainableLayerCL(layerChild, weightsSection, 4, (bidirectional ? 2 : 4) * helpers::safeJsonGetInt(layerChild, "size") + 3, precedingLayerCL)
        , m_isBidirectional      (bidirectional)
    {
        if (m_isBidirectional && this->size() % 2 != 0)
            throw std::runtime_error("Cannot create a bidirectional layer with an odd layer size");

        // set raw pointers
        int ls  = this->size();
        int pls = this->precedingLayerCL().size();


        _rawNiBiasWeights_Offset     = 4 * ls * pls + 0 * ls;
        _rawIgBiasWeights_Offset     = 4 * ls * pls + 1 * ls;
        _rawFgBiasWeights_Offset     = 4 * ls * pls + 2 * ls;
        _rawOgBiasWeights_Offset     = 4 * ls * pls + 3 * ls;
        _rawIgPeepholeWeights_Offset = 4 * ls * pls + 4 * ls + 4 * ls * ls / (m_isBidirectional ? 2 : 1) + 0 * ls;
        _rawFgPeepholeWeights_Offset = 4 * ls * pls + 4 * ls + 4 * ls * ls / (m_isBidirectional ? 2 : 1) + 1 * ls;
        _rawOgPeepholeWeights_Offset = 4 * ls * pls + 4 * ls + 4 * ls * ls / (m_isBidirectional ? 2 : 1) + 2 * ls;

        // create the forward and backward info structs
        forward_backward_info_t* fwbwArr[] = { &m_fw, &m_bw };
        for (int fwbwArrIdx = 0; fwbwArrIdx < (m_isBidirectional ? 2 : 1); ++fwbwArrIdx) {
            forward_backward_info_t *fwbw = fwbwArr[fwbwArrIdx];

            // calculate sizes
            int pls = this->precedingLayerCL().size();
            int ls  = this->size();
            int els = this->size() / (m_isBidirectional ? 2 : 1);

            // cell states, niags, deltas, ...
            std::vector<real_t> tmp(this->outputs().size / (m_isBidirectional ? 2 : 1), 0);

            if (m_isBidirectional) {
                SystemCL::copy_real(fwbw->tmpOutputs, tmp);
                SystemCL::copy_real(fwbw->tmpOutputErrors,tmp);
            }
            else {
            	d_mem_real temp1 = this->_outputs();
            	d_mem_real temp2 = this->outputErrors();
            	this->_outputs() = fwbw->tmpOutputs ;
            	this->outputErrors() =  fwbw->tmpOutputErrors;
                fwbw->tmpOutputs     = temp1;
                fwbw->tmpOutputErrors= temp2;
            }

            SystemCL::copy_real( fwbw->cellStates , tmp);
            SystemCL::copy_real(fwbw->cellStateErrors, tmp);
            SystemCL::copy_real(fwbw->niActs, tmp);
            SystemCL::copy_real(fwbw->igActs, tmp);
            SystemCL::copy_real(fwbw->fgActs, tmp);
            SystemCL::copy_real(fwbw->ogActs, tmp);
            SystemCL::copy_real(fwbw->niDeltas, tmp);
            SystemCL::copy_real(fwbw->igDeltas, tmp);;
            SystemCL::copy_real(fwbw->fgDeltas, tmp);
            SystemCL::copy_real(fwbw->ogDeltas, tmp);
            // weight matrices
            weight_matrices_t* wmArr [] = { &fwbw->weightMatrices, &fwbw->weightUpdateMatrices };
            d_mem_real*       wtsArr[] = { &this->weights(),      &this->_weightUpdates() };
            for (int wmArrIdx = 0; wmArrIdx < 2; ++wmArrIdx) {
                weight_matrices_t *wm  = wmArr [wmArrIdx];
                d_mem_real       *wts = wtsArr[wmArrIdx];

                int numInputWeights      = ls * pls;
                int numInternalWeights   = ls * els;
                int inputWeightsStart    = ((fwbwArrIdx == 1) ? (numInputWeights    / 2) : 0);
                int internalWeightsStart = ((fwbwArrIdx == 1) ? (numInternalWeights / 2) : 0) + 4 * (ls * (pls + 1));

                wm->niInput = helpers::MatrixCL(wts, pls, els, inputWeightsStart + 0 * numInputWeights);

                wm->igInput = helpers::MatrixCL(wts, pls, els, inputWeightsStart + 1 * numInputWeights);
                wm->fgInput = helpers::MatrixCL(wts, pls, els, inputWeightsStart + 2 * numInputWeights);
                wm->ogInput = helpers::MatrixCL(wts, pls, els, inputWeightsStart + 3 * numInputWeights);

                wm->niInternal = helpers::MatrixCL(wts, els, els, internalWeightsStart + 0 * numInternalWeights);
                wm->igInternal = helpers::MatrixCL(wts, els, els, internalWeightsStart + 1 * numInternalWeights);
                wm->fgInternal = helpers::MatrixCL(wts, els, els, internalWeightsStart + 2 * numInternalWeights);
                wm->ogInternal = helpers::MatrixCL(wts, els, els, internalWeightsStart + 3 * numInternalWeights);
            }

            // matrices for each timestep
            for (int timestep = 0; timestep < this->maxSeqLength(); ++timestep) {
                int rows   = this->size() / (m_isBidirectional ? 2 : 1);
                int cols   = this->parallelSequences();
                int offset = timestep * rows * cols;

                timestep_matrices_t tm;
                tm.tmpOutputs      = helpers::MatrixCL(&fwbw->tmpOutputs,      rows, cols, offset);
                tm.tmpOutputErrors = helpers::MatrixCL(&fwbw->tmpOutputErrors, rows, cols, offset);
                tm.niActs          = helpers::MatrixCL(&fwbw->niActs,          rows, cols, offset);
                tm.igActs          = helpers::MatrixCL(&fwbw->igActs,          rows, cols, offset);
                tm.fgActs          = helpers::MatrixCL(&fwbw->fgActs,          rows, cols, offset);
                tm.ogActs          = helpers::MatrixCL(&fwbw->ogActs,          rows, cols, offset);
                tm.niDeltas        = helpers::MatrixCL(&fwbw->niDeltas,        rows, cols, offset);
                tm.igDeltas        = helpers::MatrixCL(&fwbw->igDeltas,        rows, cols, offset);
                tm.fgDeltas        = helpers::MatrixCL(&fwbw->fgDeltas,        rows, cols, offset);
                tm.ogDeltas        = helpers::MatrixCL(&fwbw->ogDeltas,        rows, cols, offset);

                fwbw->timestepMatrices.push_back(tm);
            }
        }

        if (!m_isBidirectional) {
        	d_mem_real temp1 = this->_outputs();
        	d_mem_real temp2 = this->outputErrors();
        	this->_outputs() =  m_fw.tmpOutputs ;
        	this->outputErrors() =   m_fw.tmpOutputErrors;
        	m_fw.tmpOutputs      = temp1;
        	m_fw.tmpOutputErrors = temp2;
        }
    }


    LstmLayerCL::~LstmLayerCL()
    {
    }


    const std::string& LstmLayerCL::type() const
    {
        static const std::string su("lstm");
        static const std::string sb("blstm");
        return (m_isBidirectional ? sb : su);
    }


    bool LstmLayerCL::isBidirectional() const
    {
        return m_isBidirectional;
    }


    const d_mem_real& LstmLayerCL::cellStates() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.cellStates;
    }


    const d_mem_real& LstmLayerCL::cellStateErrors() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.cellStateErrors;
    }


    const d_mem_real& LstmLayerCL::netInputActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.niActs;
    }


    const d_mem_real& LstmLayerCL::netInputDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.niDeltas;
    }


    const d_mem_real& LstmLayerCL::inputGateActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.igActs;
    }


    const d_mem_real& LstmLayerCL::inputGateDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.igDeltas;
    }


    const d_mem_real& LstmLayerCL::forgetGateActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.fgActs;
    }


    const d_mem_real& LstmLayerCL::forgetGateDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.fgDeltas;
    }


    const d_mem_real& LstmLayerCL::outputGateActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.ogActs;
    }


    const d_mem_real& LstmLayerCL::outputGateDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.ogDeltas;
    }


    void LstmLayerCL::loadSequences(const data_sets::DataSetFractionCL &fraction)
    {

        TrainableLayerCL::loadSequences(fraction);

        m_precLayerCLOutputsMatrixCL = helpers::MatrixCL(&this->precedingLayerCL().outputs(), this->precedingLayerCL().size(), this->curMaxSeqLength() * this->parallelSequences());

        // update the niag matrices
        forward_backward_info_t* fwbwArr[] = { &m_fw, &m_bw };
        for (int fwbwArrIdx = 0; fwbwArrIdx < (m_isBidirectional ? 2 : 1); ++fwbwArrIdx) {
            forward_backward_info_t *fwbw = fwbwArr[fwbwArrIdx];

            int rows = this->size() / (m_isBidirectional ? 2 : 1);
            int cols = this->curMaxSeqLength() * this->parallelSequences();

            fwbw->niActsMatrixCL = helpers::MatrixCL(&fwbw->niActs, rows, cols);


            fwbw->igActsMatrixCL = helpers::MatrixCL(&fwbw->igActs, rows, cols);
            fwbw->fgActsMatrixCL = helpers::MatrixCL(&fwbw->fgActs, rows, cols);
            fwbw->ogActsMatrixCL = helpers::MatrixCL(&fwbw->ogActs, rows, cols);

            fwbw->niDeltasMatrixCL = helpers::MatrixCL(&fwbw->niDeltas, rows, cols);
            fwbw->igDeltasMatrixCL = helpers::MatrixCL(&fwbw->igDeltas, rows, cols);
            fwbw->fgDeltasMatrixCL = helpers::MatrixCL(&fwbw->fgDeltas, rows, cols);
            fwbw->ogDeltasMatrixCL = helpers::MatrixCL(&fwbw->ogDeltas, rows, cols);
        }
    }


    void LstmLayerCL::computeForwardPass()
    {
        // for unidirectional LSTM, we can write the outputs directly in the layer output vector
        if (!m_isBidirectional) {
        	d_mem_real tmp = this->_outputs();
        	 this->_outputs() = m_fw.tmpOutputs;
        	 m_fw.tmpOutputs = tmp;
        }

        // sum up the activations from the preceding layer
        {{
            // forward states
            m_fw.niActsMatrixCL.assignProduct(m_fw.weightMatrices.niInput, true, m_precLayerCLOutputsMatrixCL, false);
            m_fw.igActsMatrixCL.assignProduct(m_fw.weightMatrices.igInput, true, m_precLayerCLOutputsMatrixCL, false);
            m_fw.fgActsMatrixCL.assignProduct(m_fw.weightMatrices.fgInput, true, m_precLayerCLOutputsMatrixCL, false);
            m_fw.ogActsMatrixCL.assignProduct(m_fw.weightMatrices.ogInput, true, m_precLayerCLOutputsMatrixCL, false);


            // backward states
            if (m_isBidirectional) {
                m_bw.niActsMatrixCL.assignProduct(m_bw.weightMatrices.niInput, true, m_precLayerCLOutputsMatrixCL, false);
                m_bw.igActsMatrixCL.assignProduct(m_bw.weightMatrices.igInput, true, m_precLayerCLOutputsMatrixCL, false);
                m_bw.fgActsMatrixCL.assignProduct(m_bw.weightMatrices.fgInput, true, m_precLayerCLOutputsMatrixCL, false);
                m_bw.ogActsMatrixCL.assignProduct(m_bw.weightMatrices.ogInput, true, m_precLayerCLOutputsMatrixCL, false);
            }

        }}

        // compute the block outputs
        {{
            int els = this->size() / (m_isBidirectional ? 2 : 1);
            int n   = this->parallelSequences() * els;

;
            int effLayerCLSize       = els;
            int prevOutputDistance = -n;
            float bias               = this->bias();

            int niBiasWeights      = _rawNiBiasWeights_Offset;
            int igBiasWeights      = _rawIgBiasWeights_Offset;
            int fgBiasWeights      = _rawFgBiasWeights_Offset;
            int ogBiasWeights      = _rawOgBiasWeights_Offset;
            int igPeepWeights      = _rawIgPeepholeWeights_Offset;
            int fgPeepWeights      = _rawFgPeepholeWeights_Offset;
            int ogPeepWeights      = _rawOgPeepholeWeights_Offset;

            for (int timestep = 0; timestep < this->curMaxSeqLength(); ++timestep) {
                // collect outputs from previous timestep
                if (timestep != 0) {
                    m_fw.timestepMatrices[timestep].niActs.addProduct(m_fw.weightMatrices.niInternal, true, m_fw.timestepMatrices[timestep-1].tmpOutputs, false);
                    m_fw.timestepMatrices[timestep].igActs.addProduct(m_fw.weightMatrices.igInternal, true, m_fw.timestepMatrices[timestep-1].tmpOutputs, false);
                    m_fw.timestepMatrices[timestep].fgActs.addProduct(m_fw.weightMatrices.fgInternal, true, m_fw.timestepMatrices[timestep-1].tmpOutputs, false);
                    m_fw.timestepMatrices[timestep].ogActs.addProduct(m_fw.weightMatrices.ogInternal, true, m_fw.timestepMatrices[timestep-1].tmpOutputs, false);
                }

                int firstCall = !timestep? 1 : 0;
                int checkPatType = timestep >= this->curMinSeqLength()? 1:  0;
                int offset = n * timestep;
                // compute outputs


                SystemCL::ll_computeBlockOutputFn(effLayerCLSize,prevOutputDistance,
                		bias,this->patTypes(),
						 this->weights(),
						niBiasWeights,
						igBiasWeights,
						fgBiasWeights,
						ogBiasWeights,
						igPeepWeights,
						fgPeepWeights,
						ogPeepWeights,
						m_fw.cellStates,m_fw.niActs, m_fw.igActs,
						m_fw.fgActs,m_fw.ogActs, firstCall, checkPatType,
						 m_fw.tmpOutputs, offset, n);
 //              SystemCL::print(m_fw.tmpOutputs, offset);


            }

            // backward states
            if (m_isBidirectional) {
                prevOutputDistance = +n;
                niBiasWeights     += els;
                igBiasWeights     += els;
                fgBiasWeights     += els;
                ogBiasWeights     += els;
                igPeepWeights     += els;
                fgPeepWeights     += els;
                ogPeepWeights     += els;



                for (int timestep = this->curMaxSeqLength()-1; timestep >= 0; --timestep) {
                    // collect outputs from previous timestep
                    if (timestep != this->curMaxSeqLength()-1) {
                        m_bw.timestepMatrices[timestep].niActs.addProduct(m_bw.weightMatrices.niInternal, true, m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
                        m_bw.timestepMatrices[timestep].igActs.addProduct(m_bw.weightMatrices.igInternal, true, m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
                        m_bw.timestepMatrices[timestep].fgActs.addProduct(m_bw.weightMatrices.fgInternal, true, m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
                        m_bw.timestepMatrices[timestep].ogActs.addProduct(m_bw.weightMatrices.ogInternal, true, m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
                    }


                    int firstCall = timestep == this->curMaxSeqLength()-1? 1:0;
                    int checkPatType = timestep >= this->curMinSeqLength() ? 1:0;
                    int offset = n * timestep;
                    SystemCL::ll_computeBlockOutputFn(effLayerCLSize,prevOutputDistance,
                    		bias,this->patTypes(),
							 this->weights(),
    						niBiasWeights,
    						igBiasWeights,
    						fgBiasWeights,
    						ogBiasWeights,
    						igPeepWeights,
    						fgPeepWeights,
    						ogPeepWeights,
							m_bw.cellStates,m_bw.niActs, m_bw.igActs,
							m_bw.fgActs,m_bw.ogActs, firstCall, checkPatType,
							m_bw.tmpOutputs, offset, n);
                }
            }
        }}

        // resort outputs
        if (m_isBidirectional) {

        	int layerSize    = this->size();
            int effLayerCLSize = this->size() / 2;

        	int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
        	SystemCL::ll_resortOutputsFn(layerSize, effLayerCLSize,m_fw.tmpOutputs, m_bw.tmpOutputs, this->_outputs(), n);
        }
        else {
        	d_mem_real temp = m_fw.tmpOutputs;
        	m_fw.tmpOutputs = this->_outputs();
        	this->_outputs() = temp;
        }

    }


    void LstmLayerCL::computeBackwardPass()
    {
        // for unidirectional LSTM, we can write the output errors directly in the layer output errors vector
        if (m_isBidirectional) {
            int layerSize      = this->size();
            int effLayerCLSize   = this->size() / 2;
            int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();


        	SystemCL::ll_resortOutputErrorsFn(layerSize, effLayerCLSize,m_fw.tmpOutputErrors, m_bw.tmpOutputErrors,
        			this->outputErrors(), n);
        }
        else {
        	d_mem_real temp = this->outputs();
        	this->outputs() = m_fw.tmpOutputs;
        	m_fw.tmpOutputs = temp;

        	temp = this->outputErrors();
        	this->outputErrors() =m_fw.tmpOutputErrors;
        	m_fw.tmpOutputErrors = temp;
        }

        // calculate the block errors
        {{
            int els = this->size() / (m_isBidirectional ? 2 : 1);
            int n   = this->parallelSequences() * els;

            // forward states
            int effLayerCLSize       = els;
            int prevOutputDistance = -n;
            int igPeepWeights_Offset      = _rawIgPeepholeWeights_Offset;
            int fgPeepWeights_Offset      = _rawFgPeepholeWeights_Offset;
            int ogPeepWeights_Offset      = _rawOgPeepholeWeights_Offset;


            for (int timestep = this->curMaxSeqLength()-1; timestep >= 0; --timestep) {
                // collect errors from previous timestep
                if (timestep != this->curMaxSeqLength()-1) {
                    m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_fw.weightMatrices.niInternal, false, m_fw.timestepMatrices[timestep+1].niDeltas, false);
                    m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_fw.weightMatrices.igInternal, false, m_fw.timestepMatrices[timestep+1].igDeltas, false);
                    m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_fw.weightMatrices.fgInternal, false, m_fw.timestepMatrices[timestep+1].fgDeltas, false);
                    m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_fw.weightMatrices.ogInternal, false, m_fw.timestepMatrices[timestep+1].ogDeltas, false);
                }
                int offset = n*timestep;
                int firstCall = (timestep == this->curMaxSeqLength()-1)? 1 : 0;
				int lastCall = !timestep ? 1 : 0;
				int checkPatType = (timestep >= this->curMinSeqLength())? 1: 0;

                SystemCL::ll_computeBlockErrorsFn(effLayerCLSize, prevOutputDistance, this->patTypes(),  this->weights(),
                		igPeepWeights_Offset ,fgPeepWeights_Offset,ogPeepWeights_Offset,
						m_fw.cellStates,
						m_fw.niActs,
						m_fw.igActs,
						m_fw.fgActs,
						m_fw.ogActs,
						m_fw.cellStateErrors,
						m_fw.niDeltas,
						m_fw.igDeltas,
						m_fw.fgDeltas,
						m_fw.ogDeltas,
						m_fw.tmpOutputErrors, offset, firstCall, lastCall, checkPatType, n
						);

            }

            // backward states
            if (m_isBidirectional) {
                prevOutputDistance = +n;
                igPeepWeights_Offset     += els;
                fgPeepWeights_Offset     += els;
                ogPeepWeights_Offset     += els;

                for (int timestep = 0; timestep < this->curMaxSeqLength(); ++timestep) {
                    // collect errors from previous timestep
                    if (timestep != 0) {
                        m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_bw.weightMatrices.niInternal, false, m_bw.timestepMatrices[timestep-1].niDeltas, false);
                        m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_bw.weightMatrices.igInternal, false, m_bw.timestepMatrices[timestep-1].igDeltas, false);
                        m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_bw.weightMatrices.fgInternal, false, m_bw.timestepMatrices[timestep-1].fgDeltas, false);
                        m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_bw.weightMatrices.ogInternal, false, m_bw.timestepMatrices[timestep-1].ogDeltas, false);
                    }


                    int offset = n*timestep;
                    int firstCall = !timestep? 1 : 0;
    				int lastCall = (timestep == this->curMaxSeqLength()-1)? 1 : 0;
    				int checkPatType = (timestep >= this->curMinSeqLength())? 1: 0;
                    SystemCL::ll_computeBlockErrorsFn(effLayerCLSize, prevOutputDistance, this->patTypes(),  this->weights(),
                    		igPeepWeights_Offset ,fgPeepWeights_Offset,ogPeepWeights_Offset,
							m_bw.cellStates,
							m_bw.niActs,
							m_bw.igActs,
							m_bw.fgActs,
							m_bw.ogActs,
							m_bw.cellStateErrors,
							m_bw.niDeltas,
							m_bw.igDeltas,
							m_bw.fgDeltas,
							m_bw.ogDeltas,
							m_bw.tmpOutputErrors, offset, firstCall, lastCall, checkPatType, n
    						);
                }
            }
        }}
        // back-propagate the error to the preceding layer
        {{
            TrainableLayerCL *pl = dynamic_cast<TrainableLayerCL*>(&this->precedingLayerCL());
            if (pl) {
                helpers::MatrixCL plErrorsMatrix(&pl->outputErrors(), pl->size(), this->curMaxSeqLength() * this->parallelSequences());

                // forward states
                plErrorsMatrix.assignProduct(m_fw.weightMatrices.niInput, false, m_fw.niDeltasMatrixCL, false);
                plErrorsMatrix.addProduct   (m_fw.weightMatrices.igInput, false, m_fw.igDeltasMatrixCL, false);
                plErrorsMatrix.addProduct   (m_fw.weightMatrices.fgInput, false, m_fw.fgDeltasMatrixCL, false);
                plErrorsMatrix.addProduct   (m_fw.weightMatrices.ogInput, false, m_fw.ogDeltasMatrixCL, false);

                // backward states
                if (m_isBidirectional) {
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.niInput, false, m_bw.niDeltasMatrixCL, false);
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.igInput, false, m_bw.igDeltasMatrixCL, false);
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.fgInput, false, m_bw.fgDeltasMatrixCL, false);
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.ogInput, false, m_bw.ogDeltasMatrixCL, false);
                }
            }
        }}


        // compute the weight updates
        {{
            int layerSize             = this->size();
            int effLayerCLSize        = this->size() / (m_isBidirectional ? 2 : 1);
            int precLayerCLSize       = this->precedingLayerCL().size();
            int timestepDistance      = this->parallelSequences() * this->size() / (m_isBidirectional ? 2 : 1);
            int parallelSequences     = this->parallelSequences();
            int patternsCount         = this->curMaxSeqLength() * this->parallelSequences();
            int biasWeightsOffset     = this->size() * this->precedingLayerCL().size() * 4;
            int internalWeightsOffset = biasWeightsOffset + this->size() * 4;
            int peepholeWeightsOffset = internalWeightsOffset + this->size() * effLayerCLSize * 4;
            float bias                  = this->bias();
            int n = (int)this->weightUpdates().size;

        	SystemCL::ll_computeWeightUpdateFn(layerSize,effLayerCLSize, precLayerCLSize,timestepDistance,
        			parallelSequences, patternsCount, biasWeightsOffset, internalWeightsOffset, peepholeWeightsOffset,
					bias,
					this->precedingLayerCL().outputs(),m_fw.tmpOutputs, m_bw.tmpOutputs,
					m_fw.cellStates, m_bw.cellStates,
					m_fw.niDeltas, m_bw.niDeltas,
					m_fw.igDeltas, m_bw.igDeltas,
					m_fw.fgDeltas, m_bw.fgDeltas,
					m_fw.ogDeltas, m_bw.ogDeltas,
					  this->_weightUpdates(), n);

        }}

        // re-swap the output errors and the tmp output errors of the forward pass
        if (!m_isBidirectional) {
        	d_mem_real temp = m_fw.tmpOutputErrors;
        	m_fw.tmpOutputErrors = this->outputErrors();
        	this->outputErrors() = temp;

        	temp = m_fw.tmpOutputs;
        	m_fw.tmpOutputs = this->_outputs();
        	this->_outputs() = temp;
        }
    }



} // namespace layers
