arma::dmat mff = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
arma::dmat kff = {{2.0, -1.0, 0.0}, {-1.0, 2.0, -1.0}, {0.0, -1.0, 2.0}};

arma::dmat alpha = arma::solve(mff, kff);
    
arma::cx_vec eigval_u = arma::eig_gen(alpha);
arma::dvec eig_val = arma::sort(arma::real(eigval_u));

// Iterations
uint s = 100;

// Coordinate transformation
arma::dmat p_mat = arma::zeros(eig_val.n_rows, eig_val.n_rows);

// Mu matrix 
arma::dvec mu_mat = arma::zeros(eig_val.n_rows, 1);

for (uint i = 0; i < eig_val.n_rows; i++) 
{
    if (i == 0) { mu_mat(i) = arma::as_scalar(eig_val(i)) - 0.1; }
    else
    {
        mu_mat(i) = arma::as_scalar(eig_val(i)) -
            abs(arma::as_scalar(eig_val(i - 1) - eig_val(i))) / 4.0;
    }
}

// Lambda_hat vector 
arma::dvec lambda_hat = arma::zeros(eig_val.n_rows, 1);

for (uint i = 0; i < eig_val.n_rows; i++) 
{
    arma::dmat k_hat = kff - mu_mat(i) * mff;
    lambda_hat = eig_val - mu_mat(i) * arma::ones(eig_val.n_rows, 1);

    // Starting vector
    arma::dvec  u = arma::ones(eig_val.n_rows, 1);
    double lambda_up = 0.0;

    // Dynamic matrix 
    arma::dmat dyn_mat = arma::solve(k_hat, mff);
    uint counter = 0; double tol = 1e-4;
        
    while(counter < s)
    {
        // v vector update
        arma::dvec v_up = dyn_mat * u;

        double nom = arma::as_scalar(v_up.t() * k_hat * v_up);
        double denom = arma::as_scalar(v_up.t() * mff * v_up);
        lambda_up = nom / denom;

        // Update eigenvector estimation
        u = v_up / sqrt(denom);

        // Update counter
        counter++; 
    }
    p_mat.col(i) = u;
}
