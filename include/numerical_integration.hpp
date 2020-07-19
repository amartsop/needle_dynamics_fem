#pragma once 

#include <iostream>
#include <vector>
#include <armadillo>


template <class T>
class NumericalIntegration
{
public:
    NumericalIntegration(T *model, double integration_step, uint type);

    arma::dvec solve(arma::dvec state, double t);

private:

    arma::dvec runge_kutta_2(arma::dvec state, double t);
    arma::dvec runge_kutta_4(arma::dvec state, double t);
    arma::dvec runge_kutta_5(arma::dvec state, double t);

    arma::dvec adams_bashforth_2(arma::dvec state, double t);
    arma::dvec adams_bashforth_4(arma::dvec state, double t);
    
    arma::dvec predictor_corrector_2(arma::dvec state, double t);
    arma::dvec predictor_corrector_4(arma::dvec state, double t);

    
    // Model handle
    T *m_model;

    // Integration step 
    double m_integration_step;

    // Integration type 
    uint m_type;

    // State vector 
    arma::dvec m_y;

    // Iterations counter 
    uint64_t m_iterations;

    // Previous evaluation holder 
    arma::dvec m_previous_evaluations[10];
};


template <class T>
NumericalIntegration<T>::NumericalIntegration(T *model, double integration_step,
    uint type)
{
    // Integration type 
    m_type = type;

    // Model to be integrated  
    m_model = model;
    
    // Integration step 
    m_integration_step = integration_step;

    // State vector 
    uint state_vec_size = m_model->get_model_size();
    m_y = arma::zeros<arma::dvec>(state_vec_size);

    // Iterations 
    m_iterations = 0;
}


template <class T>
arma::dvec NumericalIntegration<T>::solve(arma::dvec state, double t)
{
    switch (m_type)
    {
        case 2:
            m_y = runge_kutta_2(state, t);
            break;
        case 4:
            m_y = runge_kutta_4(state, t);
            break;
        case 5:
            m_y = runge_kutta_5(state, t);
            break;
        case 6:
            m_y = adams_bashforth_2(state, t);
            break;
        case 7:
            m_y = adams_bashforth_4(state, t);
            break;
        case 8:
            m_y = predictor_corrector_2(state, t);
            break;
        case 9:
            m_y = predictor_corrector_4(state, t);
            break;
        default:
            runge_kutta_4(state, t);
            break;
    }

    // Update iterations
    m_iterations += 1;

    return m_y;
}


//************************* Predictor-Corrector ****************************//
template <class T>
arma::dvec NumericalIntegration<T>::predictor_corrector_2(arma::dvec state,
    double t)
{
    // Predictor
    arma::dvec x_predict = adams_bashforth_2(state, t);
    double t_predict = t + m_integration_step;
    arma::dvec f_predicted = m_model->calculate(x_predict, t_predict);

    // Current 
    arma::dvec f_current = m_model->calculate(state, t);
    
    // Corrector
    return (state + (m_integration_step / 2.0) * (f_current + f_predicted));
}
 
template <class T>
arma::dvec NumericalIntegration<T>::predictor_corrector_4(arma::dvec state,
    double t)
{
    arma::dvec x_predict = adams_bashforth_4(state, t);
    double t_predict = t + m_integration_step;
    arma::dvec f_predicted = m_model->calculate(x_predict, t_predict);

    //Current 
    arma::dvec f_current = m_model->calculate(state, t);

    
    if(m_iterations < 3) 
    {
        return x_predict;
    }
    else
    {
        // Previous
        arma::dvec f_prev_2 = m_previous_evaluations[0];
        arma::dvec f_prev_1 = m_previous_evaluations[1];
    
        // Corrector
        return (state + (m_integration_step / 24.0) * (9.0 * f_predicted +
            19.0 * f_current  - 5.0 * f_prev_1 + f_prev_2));
    }
    
}

//************************* Adams-Bushforth Methods ****************************//
template <class T>
arma::dvec NumericalIntegration<T>::adams_bashforth_2(arma::dvec state, double t)
{

    if (m_iterations == 0)
    {
        m_previous_evaluations[0] = m_model->calculate(state, t);

        // Calculation of state 1
        return runge_kutta_2(state, t);
    }
    else
    {
        arma::dvec f_prev = m_previous_evaluations[0];
        arma::dvec f_current = m_model->calculate(state, t);
        m_previous_evaluations[0] = f_current;

        return state + (m_integration_step / 2.0) * (3 * f_current - f_prev);
    }
}

template <class T>
arma::dvec NumericalIntegration<T>::adams_bashforth_4(arma::dvec state, double t)
{
    if (m_iterations < 3)
    {
        m_previous_evaluations[m_iterations] = m_model->calculate(state, t);

        // Calculation of state 1
        return runge_kutta_4(state, t);
    }
    else
    {
        arma::dvec f_prev_3 = m_previous_evaluations[0];
        arma::dvec f_prev_2 = m_previous_evaluations[1];
        arma::dvec f_prev_1 = m_previous_evaluations[2];
        arma::dvec f_current = m_model->calculate(state, t);
        
        m_previous_evaluations[0] = f_prev_2;
        m_previous_evaluations[1] = f_prev_1;
        m_previous_evaluations[2] = f_current;

        return state + (m_integration_step / 24.0) * (55.0 * f_current - 
            59.0 * f_prev_1 + 37.0 * f_prev_2 - 9.0 * f_prev_3);
    }
}

//************************* Runge-Kutta Methods ****************************//
template <class T>
arma::dvec NumericalIntegration<T>::runge_kutta_2(arma::dvec state, double t)
{   
    double a = 2.0 / 3.0;
    double r = 1.0 / (2.0 * a);

    arma::dvec k1 = m_model->calculate(state, t);
    arma::dvec k2 = m_model->calculate(state + a * (m_integration_step) * k1,
        t + a * (m_integration_step));

    return state + (m_integration_step) * ( (1.0 - r) * k1 + r * k2 );
}

template <class T>
arma::dvec NumericalIntegration<T>::runge_kutta_4(arma::dvec state, double t)
{   
    arma::dvec k1 = m_model->calculate(state, t);

    arma::dvec k2 = m_model->calculate(state + (m_integration_step / 2.0) * k1, 
        t + (m_integration_step / 2.0));
    
    arma::dvec k3 = m_model->calculate(state + (m_integration_step / 2.0) * k2, 
        t + (m_integration_step / 2.0));

    arma::dvec k4 = m_model->calculate(state + m_integration_step * k3, 
        t + m_integration_step);

    return state + (1.0 / 6.0) * m_integration_step * (k1 + 2.0 * k2 + 
        2.0 * k3 + k4);
}

template <class T>
arma::dvec NumericalIntegration<T>::runge_kutta_5(arma::dvec state, double t)
{   
    arma::dvec k1 = m_model->calculate(state, t);

    arma::dvec k2 = m_model->calculate(state + (m_integration_step / 4.0) * k1, 
        t + (m_integration_step / 4.0));

    arma::dvec k3 = m_model->calculate(state + (m_integration_step / 8.0) * k1
        + (m_integration_step / 8.0) * k2, t + (m_integration_step / 4.0));
    
    arma::dvec k4 = m_model->calculate(state - (m_integration_step / 2.0) * k2
        + (m_integration_step / 1.0) * k3, t + (m_integration_step / 2.0));

    arma::dvec k5 = m_model->calculate(state + 
        3.0 * (m_integration_step / 16.0) * k1 + 
        9.0 * (m_integration_step / 16.0) * k4, t + 
        3.0 * (m_integration_step / 4.0));
    
    arma::dvec k6 = m_model->calculate(state - 
        3.0 * (m_integration_step / 7.0) * k1 + 
        2.0 * (m_integration_step / 7.0) * k2 + 
        12.0 * (m_integration_step / 7.0) * k3 -
        12.0 * (m_integration_step / 7.0) * k4 + 
        8.0 * (m_integration_step / 7.0) * k5, t + m_integration_step);
    

    return state + (1.0 / 90.0) * m_integration_step * (7.0 * k1 + 32.0 * k3 + 
        12.0 * k4 + 32.0 * k5 + 7.0 * k6);
}