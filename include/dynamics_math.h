#pragma once 
#include <iostream>
#include <armadillo>

namespace dm
{
    // Locator matrix
    arma::dmat locator_matrix(arma::ivec locator_vector, 
        uint32_t n_cols);

    // Skew-symmetric matrix 
    arma::dmat s(arma::dvec r);
} 
