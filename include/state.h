#pragma once 
#include <armadillo>

namespace state
{
    inline uint nf(arma::dvec q)
    {
        return (q.n_rows - 6);
    }

    // Reference frame position (roa_F_F)
    inline arma::dvec roa(arma::dvec q, arma::dvec q_dot)
    {
        return {q(0), q(1), q(2)};
    }

    // Reference frame orientation (euler angles theta)
    inline arma::dvec theta(arma::dvec q, arma::dvec q_dot)
    {
        return {q(3), q(4), q(5)};
    }

    // Elastic coordinates
    inline arma::dvec qf(arma::dvec q, arma::dvec q_dot)
    {
        return q(arma::span(6, q.n_rows - 1));
    }

    // Reference frame linear velocity
    inline arma::dvec roa_dot(arma::dvec q, arma::dvec q_dot)
    {
        return {q_dot(0), q_dot(1), q_dot(2)};
    }

    // Reference frame orientation velocity (euler angles derivative theta_dot)
    inline arma::dvec theta_dot(arma::dvec q, arma::dvec q_dot)
    {
        return {q_dot(3), q_dot(4), q_dot(5)};
    }

    inline arma::dvec qf_dot(arma::dvec q, arma::dvec q_dot)
    {
        return q_dot(arma::span(6, q.n_rows - 1));
    }
}