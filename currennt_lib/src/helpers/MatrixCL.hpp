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

#ifndef HELPERS_MATRIXCL_HPP
#define HELPERS_MATRIXCL_HPP

#include "../TypesCL.hpp"
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <stdio.h>
#include <vector>
#include <iostream>
#include "../SystemCL.hpp"
namespace helpers {


    class MatrixCL
    {


    private:
        d_mem_real *m_dataVector;
        int          m_dataVectorOffset;
        cl_mem       m_data;
        int          m_rows;
        int          m_cols;

    public:
        MatrixCL();
        MatrixCL(d_mem_real *data, int rows, int cols, int dataOffset = 0);
        ~MatrixCL();

        void assignProduct(const MatrixCL &a, bool transposeA, const MatrixCL &b, bool transposeB);
        void addProduct   (const MatrixCL &a, bool transposeA, const MatrixCL &b, bool transposeB);

        void TAM_print(){
//        	if (m_dataVector->size >0){
//       		std::vector<real_t> host = SystemCL::copy_real(*m_dataVector);
//       		int size = host.size() > 50 ? 50 : host.size();
//        		for (int i = 0 ; i < size; i++){
//        			std::cout << host[i + m_dataVectorOffset] << " ";
//        		}
//        		std::cout << std::endl;
//        	}
        }


    };

} // namespace helpers

#endif // HELPERS_MATRIXCL_HPP
