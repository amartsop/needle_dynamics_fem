#include "dynamics_math.h"

// Locator matrix
arma::dmat dm::locator_matrix(arma::ivec locator_vector, 
    uint32_t n_cols)
{
    arma::dmat l_mat = arma::zeros<arma::dmat>(locator_vector.n_rows, n_cols);
    
    for(uint32_t i = 0; i < locator_vector.n_rows; i++)    
    {
        l_mat(i, locator_vector(i) - 1) = 1;
    }
    return l_mat;
}

// Skew symmetric matrix
arma::dmat dm::s(arma::dvec r)
{
    arma::dmat skew = {{0.0, -r(2), r(1)}, {r(2), 0.0, -r(0)}, {-r(1), r(0), 0.0}};
    return skew;
}