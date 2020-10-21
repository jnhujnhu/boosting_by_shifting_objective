#include "grad_desc_dense.hpp"
#include "utils.hpp"
#include "regularizer.hpp"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string.h>
#include <sys/time.h>
#include <stdlib.h>
#include <chrono>

extern size_t MAX_DIM;
grad_desc_dense::outputs grad_desc_dense::SAGA(double* X, double* Y, size_t N
    , blackbox* model, size_t iteration_no, double step_size) {
    // Random Generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(0, N - 1);
    std::vector<double>* losses = new std::vector<double>;
    std::vector<double>* times = new std::vector<double>;
    struct timeval tp;
    long int start_ms = 0;
    int regular = model->get_regularizer();
    double* lambda = model->get_params();

    // Store results
    losses->push_back(model->zero_oracle_dense(X, Y, N));
    // Extra Pass for Create Gradient Table
    losses->push_back((*losses)[0]);
    times->push_back(0);
    gettimeofday(&tp, NULL);
    start_ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

    double* new_weights = new double[MAX_DIM];
    double* grad_core_table = new double[N];
    double* aver_grad = new double[MAX_DIM];
    copy_vec(new_weights, model->get_model());
    memset(aver_grad, 0, MAX_DIM * sizeof(double));
    // Init Gradient Core Table
    for(size_t i = 0; i < N; i ++) {
        grad_core_table[i] = model->first_component_oracle_core_dense(X, Y, N, i);
        for(size_t j = 0; j < MAX_DIM; j ++)
            aver_grad[j] += grad_core_table[i] * X[i * MAX_DIM + j] / N;
    }
    // First pass initialization
    gettimeofday(&tp, NULL);
    times->push_back(tp.tv_sec * 1000 + tp.tv_usec / 1000 - start_ms);
    for(size_t i = 0; i < iteration_no; i ++) {
        int rand_samp = distribution(generator);
        double core = model->first_component_oracle_core_dense(X, Y, N, rand_samp, new_weights);
        double past_grad_core = grad_core_table[rand_samp];
        grad_core_table[rand_samp] = core;
        for(size_t j = 0; j < MAX_DIM; j ++) {
            // Update Weight
            new_weights[j] -= step_size * ((core - past_grad_core)* X[rand_samp * MAX_DIM + j]
                            + aver_grad[j]);
            // Update Gradient Table Average
            aver_grad[j] -= (past_grad_core - core) * X[rand_samp * MAX_DIM + j] / N;
            regularizer::proximal_operator(regular, new_weights[j], step_size, lambda);
        }
        // Store results
        if(!((i + 1) % (3 * N))) {
            losses->push_back(model->zero_oracle_dense(X, Y, N, new_weights));
            gettimeofday(&tp, NULL);
            times->push_back(tp.tv_sec * 1000 + tp.tv_usec / 1000 - start_ms);
        }
    }
    model->update_model(new_weights);
    delete[] new_weights;
    delete[] grad_core_table;
    delete[] aver_grad;
    return grad_desc_dense::outputs(losses, times);
}

grad_desc_dense::outputs grad_desc_dense::Uni_Acc(double* X, double* Y, size_t N
    , blackbox* model, size_t iteration_no, double L, double mu, int Mode) {
    std::vector<double>* losses = new std::vector<double>;
    int regular = model->get_regularizer();
    double kappa = L / mu;
    double tauy = 1.0 / (sqrt(kappa) + 1.0);
    double alpha = sqrt(L * mu) - mu;
    double taux = 1.0 / sqrt(kappa);
    if (Mode == 1)
        taux = (2.0 * sqrt(kappa) - 1.0) / kappa;
    double* lambda = model->get_params();

    losses->push_back(model->zero_oracle_dense(X, Y, N));
    double* x = new double[MAX_DIM];
    double* z = new double[MAX_DIM];
    copy_vec(x, model->get_model());
    copy_vec(z, model->get_model());

    for(int i = 0; i < iteration_no; i ++) {
        double* y = new double[MAX_DIM];
        for(int j = 0; j < MAX_DIM; j ++)
            y[j] = tauy * z[j] + (1.0 - tauy) * x[j];

        double* grad = new double[MAX_DIM];
        memset(grad, 0, MAX_DIM * sizeof(double));
        for(int j = 0; j < N; j ++) {
            double core = model->first_component_oracle_core_dense(X, Y, N, j, y);
            for(int k = 0; k < MAX_DIM; k ++)
                grad[k] += (X[j * MAX_DIM + k] * core) / (double) N;
        }

        for(int j = 0; j < MAX_DIM; j ++) {
            grad[j] += mu * y[j];
            z[j] = alpha / (alpha + mu) * z[j] + mu / (alpha + mu) * y[j] - 1.0 / (alpha + mu) * grad[j];
            x[j] = taux * z[j] + (1.0 - taux) * x[j];
        }
        if(Mode == 0)
            losses->push_back(model->zero_oracle_dense(X, Y, N, x));
        else
            losses->push_back(model->zero_oracle_dense(X, Y, N, z));
        delete[] grad;
        delete[] y;
    }
    model->update_model(x);
    delete[] x;
    delete[] z;
    std::vector<double>* times = new std::vector<double>;
    return grad_desc_dense::outputs(losses, times);
}

grad_desc_dense::outputs grad_desc_dense::G_TM(double* X, double* Y, size_t N
    , blackbox* model, size_t iteration_no, double L, double mu) {
    std::vector<double>* losses = new std::vector<double>;
    int regular = model->get_regularizer();
    double kappa = L / mu;
    double tauz = (sqrt(kappa) - 1.0) /(L * (sqrt(kappa) + 1.0));
    double alpha = sqrt(L * mu) - mu;
    double taux = (2.0 * sqrt(kappa) - 1.0) / kappa;
    double* lambda = model->get_params();

    losses->push_back(model->zero_oracle_dense(X, Y, N));
    double* p_grad = new double[MAX_DIM];
    double* z = new double[MAX_DIM];
    double* y = new double[MAX_DIM];
    copy_vec(z, model->get_model());
    copy_vec(y, model->get_model());
    memset(p_grad, 0, MAX_DIM * sizeof(double));
    for(int j = 0; j < N; j ++) {
        double core = model->first_component_oracle_core_dense(X, Y, N, j);
        for(int k = 0; k < MAX_DIM; k ++)
            p_grad[k] += (X[j * MAX_DIM + k] * core) / (double) N;
    }

    for(int k = 0; k < MAX_DIM; k ++)
        p_grad[k] += mu * y[k];

    for(int i = 0; i < iteration_no; i ++) {
        for(int j = 0; j < MAX_DIM; j ++)
            y[j] = taux * z[j] + (1.0 - taux) * y[j] + tauz * (mu * (y[j] - z[j]) - p_grad[j]);

        memset(p_grad, 0, MAX_DIM * sizeof(double));
        for(int j = 0; j < N; j ++) {
            double core = model->first_component_oracle_core_dense(X, Y, N, j, y);
            for(int k = 0; k < MAX_DIM; k ++)
                p_grad[k] += (X[j * MAX_DIM + k] * core) / (double) N;
        }

        for(int j = 0; j < MAX_DIM; j ++) {
            p_grad[j] += mu * y[j];
            z[j] = alpha / (alpha + mu) * z[j] + mu / (alpha + mu) * y[j] - 1.0 / (alpha + mu) * p_grad[j];
        }
        losses->push_back(model->zero_oracle_dense(X, Y, N, z));
    }
    model->update_model(z);
    delete[] p_grad;
    delete[] z;
    delete[] y;
    std::vector<double>* times = new std::vector<double>;
    return grad_desc_dense::outputs(losses, times);
}

grad_desc_dense::outputs grad_desc_dense::Katyusha(double* X, double* Y, size_t N
    , blackbox* model, size_t iteration_no, double L, double mu, double tau_1) {
    // Random Generator
    std::vector<double>* losses = new std::vector<double>;
    std::vector<double>* times = new std::vector<double>;
    struct timeval tp;
    long int start_ms = 0;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(0, N - 1);
    size_t m = 2.0 * N;
    size_t total_iterations = 0;
    double tau_2 = 0.5;
    if(1 - tau_1 - tau_2 < 0) tau_2 = 1 - tau_1;
    double alpha = 1.0 / (tau_1 * 3.0 * L);
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    double step_size_y = 1.0 / (3.0 * L);
    double compos_factor = 1.0 + alpha * mu;
    double compos_base = (pow((double)compos_factor, (double)m) - 1.0) / (alpha * mu);
    double* compos_pow = new double[m + 1];
    for(size_t i = 0; i <= m; i ++)
        compos_pow[i] = pow((double)compos_factor, (double)i);
    // double* compos_pow, compos_base;
    double* y = new double[MAX_DIM];
    double* z = new double[MAX_DIM];
    double* x = new double[MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    // init vectors
    copy_vec(y, model->get_model());
    copy_vec(z, model->get_model());
    copy_vec(x, model->get_model());
    // Init Weight Evaluate
    losses->push_back(model->zero_oracle_dense(X, Y, N));
    times->push_back(0);
    gettimeofday(&tp, NULL);
    start_ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
    
    // OUTTER LOOP
    for(size_t i = 0; i < iteration_no; i ++) {
        double* full_grad_core = new double[N];
        double* outter_weights = (model->get_model());
        double* aver_weights = new double[MAX_DIM];
        memset(full_grad, 0, MAX_DIM * sizeof(double));
        memset(aver_weights, 0, MAX_DIM * sizeof(double));

        // Full Gradient
        for(size_t j = 0; j < N; j ++) {
            full_grad_core[j] = model->first_component_oracle_core_dense(X, Y, N, j);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                full_grad[k] += X[j * MAX_DIM + k] * full_grad_core[j] / (double) N;
            }
        }
        // 0th Inner Iteration
        for(size_t k = 0; k < MAX_DIM; k ++)
            x[k] = tau_1 * z[k] + tau_2 * outter_weights[k]
                             + (1 - tau_1 - tau_2) * y[k];
        // INNER LOOP
        for(size_t j = 0; j < m; j ++) {
            int rand_samp = distribution(generator);
            double inner_core = model->first_component_oracle_core_dense(X, Y, N, rand_samp, x);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                double val = X[rand_samp * MAX_DIM + k];
                double katyusha_grad = full_grad[k] + val * (inner_core - full_grad_core[rand_samp]);
                double prev_z = z[k];
                z[k] -= alpha * katyusha_grad;
                regularizer::proximal_operator(regular, z[k], alpha, lambda);

                //// For Katyusha With Update Option I //////
                y[k] = x[k] - step_size_y * katyusha_grad;
                regularizer::proximal_operator(regular, y[k], step_size_y, lambda);
                ////// For Katyusha With Update Option II //////
                // y[k] = x[k] + tau_1 * (z[k] - prev_z);

                aver_weights[k] += compos_pow[j] / compos_base * y[k];

                // (j + 1)th Inner Iteration
                if(j < m - 1)
                    x[k] = tau_1 * z[k] + tau_2 * outter_weights[k]
                                     + (1 - tau_1 - tau_2) * y[k];
            }
            total_iterations ++;
        }
        model->update_model(aver_weights);
        delete[] aver_weights;
        delete[] full_grad_core;
        // Store results
        losses->push_back(model->zero_oracle_dense(X, Y, N));
        gettimeofday(&tp, NULL);
        times->push_back(tp.tv_sec * 1000 + tp.tv_usec / 1000 - start_ms);
        
    }
    delete[] y;
    delete[] z;
    delete[] x;
    delete[] full_grad;
    delete[] compos_pow;
    return grad_desc_dense::outputs(losses, times);
}

// ONLY with L2 Regularizer
grad_desc_dense::outputs grad_desc_dense::BS_SVRG(double* X, double* Y, size_t N
    , blackbox* model, size_t iteration_no, double L, double mu, double alpha
    , double choice) {
    // Random Generator
    std::vector<double>* losses = new std::vector<double>;
    std::vector<double>* times = new std::vector<double>;
    struct timeval tp;
    gettimeofday(&tp, NULL);
    long int start_ms = 0;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(0, N - 1);
    size_t m = 2.0 * N;
    size_t total_iterations = 0;
    double c = 2.0 + sqrt(3);
    double taux = (1 - mu / (c * L)) * (alpha + mu) / (alpha + L);
    double tauz = taux / mu - alpha * (1 - taux) / (mu * (L - mu));

    // Using numerical choice
    if(choice == -1.0)
        taux = (alpha + mu) / (alpha + L);

    double prob_factor = (1.0 + mu / alpha) * (1.0 + mu / alpha);
    double* prob_pow = new double[m + 1];
    for(size_t i = 0; i <= m; i ++)
        prob_pow[i] = pow((double)prob_factor, (double)i);

    // For random update
    unsigned seed2 = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen2(seed2);
    std::discrete_distribution<int> disc_dist(prob_pow, prob_pow + m);

    double* x = new double[MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    // init vector
    copy_vec(x, model->get_model());
    // Init Weight Evaluate
    losses->push_back(model->zero_oracle_dense(X, Y, N));
    times->push_back(0);
    gettimeofday(&tp, NULL);
    start_ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

    // OUTTER LOOP
    for(size_t i = 0; i < iteration_no; i ++) {
        double* full_grad_core = new double[N];
        double* outter_x = (model->get_model());
        double* aver_x = new double[MAX_DIM];
        memset(full_grad, 0, MAX_DIM * sizeof(double));
        memset(aver_x, 0, MAX_DIM * sizeof(double));

        // Full Gradient
        for(size_t j = 0; j < N; j ++) {
            full_grad_core[j] = model->first_component_oracle_core_dense(X, Y, N, j);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                full_grad[k] += X[j * MAX_DIM + k] * full_grad_core[j] / (double) N;
            }
        }

        int store_index = disc_dist(gen2);
        // INNER LOOP
        for(size_t j = 0; j < m; j ++) {
            int rand_samp = distribution(generator);
            double* y = new double[MAX_DIM];
            for(size_t k = 0; k < MAX_DIM; k ++) 
                y[k] = taux * x[k] + (1 - taux) * outter_x[k] 
                     + tauz * (mu * (outter_x[k] - x[k]) - full_grad[k] - mu * outter_x[k]);
            double inner_core = model->first_component_oracle_core_dense(X, Y
                , N, rand_samp, y);

            if(j == store_index) {
                for(size_t k = 0; k < MAX_DIM; k ++)
                    aver_x[k] = y[k];
            }
            for(size_t k = 0; k < MAX_DIM; k ++) {
                double val = X[rand_samp * MAX_DIM + k], fac = alpha + mu;
                double vr_grad = full_grad[k] + val * (inner_core - full_grad_core[rand_samp]) + mu * y[k];
                x[k] = alpha / fac *  x[k] + mu / fac * y[k] - 1.0 / fac * vr_grad;
            }
            total_iterations ++;
            delete[] y;
        }
        model->update_model(aver_x);
        delete[] aver_x;
        delete[] full_grad_core;
        // Store results
        losses->push_back(model->zero_oracle_dense(X, Y, N, x));
        gettimeofday(&tp, NULL);
        times->push_back(tp.tv_sec * 1000 + tp.tv_usec / 1000 - start_ms);
    }
    delete[] x;
    delete[] full_grad;
    delete[] prob_pow;
    return grad_desc_dense::outputs(losses, times);
}

