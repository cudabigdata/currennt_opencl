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

#include "MatrixCL.hpp"
#include "../SystemCL.hpp"


#include <stdexcept>




namespace helpers {


    MatrixCL::MatrixCL()
        : m_dataVector      (NULL)
        , m_dataVectorOffset(0)
        , m_data            (NULL)
        , m_rows            (0)
        , m_cols            (0)
    {
    }


    MatrixCL::MatrixCL(d_mem_real *data, int rows, int cols, int dataOffset)
        : m_dataVector      (data)
        , m_dataVectorOffset(dataOffset)
        , m_data            ((*data).mem )
        , m_rows            (rows)
        , m_cols            (cols)
    {
        if (rows * cols > data->size - dataOffset)
            throw std::runtime_error("MatrixCL exceeds available space in vector");
    }


    MatrixCL::~MatrixCL()
    {
    }


    void MatrixCL::assignProduct(const MatrixCL &a, bool transposeA, const MatrixCL &b, bool transposeB)
    {
        if (transposeA && !transposeB) {
            if (m_rows != a.m_cols || m_cols != b.m_cols || a.m_rows != b.m_rows)
                throw std::runtime_error("Invalid matrix dimensions");

            SystemCL::mm_tranpose_AFn(a.m_rows,b.m_rows, a.m_cols, b.m_cols,
            		             a.m_dataVectorOffset, b.m_dataVectorOffset, a.m_data,b.m_data,
            								m_rows, m_cols, m_data, m_dataVectorOffset);
        }
        else if (!transposeA && !transposeB) {
            if (m_rows != a.m_rows || m_cols != b.m_cols || a.m_cols != b.m_rows)
                throw std::runtime_error("Invalid matrix dimensions");

            SystemCL::mm_tranpose_Fn(a.m_rows,b.m_rows, a.m_cols, b.m_cols,
            		           a.m_dataVectorOffset, b.m_dataVectorOffset,a.m_data,b.m_data,
                       								m_rows, m_cols, m_data, m_dataVectorOffset);

        }
        else if (transposeA && transposeB) {
            throw std::runtime_error("Not implemented");
        }
        else /* if (!transposeA && transposeB) */ {
            if (m_rows != a.m_rows || m_cols != b.m_rows || a.m_cols != b.m_cols)
                throw std::runtime_error("Invalid matrix dimensions");

            SystemCL::mm_tranpose_BFn(a.m_rows,b.m_rows, a.m_cols, b.m_cols,
            		                    a.m_dataVectorOffset, b.m_dataVectorOffset,a.m_data,b.m_data,
                                         m_rows, m_cols, m_data, m_dataVectorOffset);
        }
    }


    void MatrixCL::addProduct(const MatrixCL &a, bool transposeA, const MatrixCL &b, bool transposeB)
    {
        if (transposeA && !transposeB) {
            if (m_rows != a.m_cols || m_cols != b.m_cols || a.m_rows != b.m_rows)
                throw std::runtime_error("Invalid matrix dimensions");

            SystemCL::mma_tranpose_AFn(a.m_rows,b.m_rows, a.m_cols, b.m_cols,
            		        a.m_dataVectorOffset, b.m_dataVectorOffset,a.m_data,b.m_data,
            								m_rows, m_cols, m_data, m_dataVectorOffset);

        }
        else if (!transposeA && !transposeB) {
            if (m_rows != a.m_rows || m_cols != b.m_cols || a.m_cols != b.m_rows)
                throw std::runtime_error("Invalid matrix dimensions");


            SystemCL::mma_tranpose_Fn(a.m_rows,b.m_rows, a.m_cols, b.m_cols,
            		             a.m_dataVectorOffset, b.m_dataVectorOffset,a.m_data,b.m_data,
                       								m_rows, m_cols, m_data, m_dataVectorOffset);
        }
        else if (transposeA && transposeB) {
            throw std::runtime_error("Not implemented");
        }
        else /* if (!transposeA && transposeB) */ {
            if (m_rows != a.m_rows || m_cols != b.m_rows || a.m_cols != b.m_cols)
                throw std::runtime_error("Invalid matrix dimensions");

            SystemCL::mma_tranpose_BFn(a.m_rows,b.m_rows, a.m_cols, b.m_cols,
            		  a.m_dataVectorOffset, b.m_dataVectorOffset,a.m_data,b.m_data,
                       								m_rows, m_cols, m_data, m_dataVectorOffset);
        }
    }




} // namespace helpers
