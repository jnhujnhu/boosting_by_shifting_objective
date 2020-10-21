#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex>
#include "regularizer.hpp"
#include "utils.hpp"

extern size_t MAX_DIM;

double regularizer::zero_oracle(int _regular, double* lambda, double* weight) {
    assert (weight != NULL);
    switch(_regular) {
        case regularizer::L1: {
            return lambda[1] * comp_l1_norm(weight);
            break;
        }
        case regularizer::L2: {
            double l2_norm = comp_l2_norm(weight);
            return lambda[0] * 0.5 * l2_norm * l2_norm;
            break;
        }
        case regularizer::ELASTIC_NET: {
            double l2_norm = comp_l2_norm(weight);
            double l2_part = lambda[0] * 0.5 * l2_norm * l2_norm;
            double l1_part = lambda[1] * comp_l1_norm(weight);
            return l1_part + l2_part;
            break;
        }
        default:
            return 0;
    }
}

void regularizer::first_oracle(int _regular, double* _pR, double* lambda, double* weight) {
    assert (weight != NULL);
    memset(_pR, 0, MAX_DIM * sizeof(double));
    switch(_regular) {
        case regularizer::L1: {
            // Subderivative
            for(size_t i = 0; i < MAX_DIM; i ++) {
                if(weight[i] >= 0)
                    _pR[i] = lambda[1];
                else
                    _pR[i] = -lambda[1];
            }
            break;
        }
        case regularizer::L2: {
            for(size_t i = 0; i < MAX_DIM; i ++) {
                _pR[i] = lambda[0] * weight[i];
            }
            break;
        }
        case regularizer::ELASTIC_NET: {
            // Subderivative
            for(size_t i = 0; i < MAX_DIM; i ++) {
                if(weight[i] >= 0)
                    _pR[i] = lambda[1] + lambda[0] * weight[i];
                else
                    _pR[i] = -lambda[1] + lambda[0] * weight[i];
            }
            break;
        }
        default:
            break;
    }
}

double regularizer::L1_proximal_loop(double& _prox, double param, size_t times, double additional_constant,
        bool is_averaged) {
    double lazy_average = 0.0;
    for(size_t i = 0; i < times; i ++) {
        _prox += additional_constant;
        if(_prox > param)
            _prox -= param;
        else if(_prox < -param)
            _prox += param;
        else
            _prox = 0;
        if(is_averaged)
            lazy_average += _prox;
    }
    return lazy_average;
}

double regularizer::L1_single_step(double& X, double P, double C, bool is_averaged) {
    double lazy_average = 0.0;
    X += C;
    if(X > P)
        X -= P;
    else if(X < -P)
        X += P;
    else
        X = 0;
    if(is_averaged)
        lazy_average = X;
    return lazy_average;
}

double regularizer::EN_proximal_loop(double& X, double P, double Q, size_t times, double C, bool is_averaged) {
    double lazy_average = 0;
    for(size_t i = 0; i < times; i ++) {
        X += C;
        if(X > P)
            X = Q * (X - P);
        else if(X < -P)
            X = Q * (X + P);
        else
            X = 0;
        if(is_averaged)
            lazy_average += X;
    }
    return lazy_average;
}

double regularizer::EN_single_step(double& X, double P, double Q, double C, bool is_averaged) {
    double lazy_average = 0.0;
    X += C;
    if(X > P)
        X = Q * (X - P);
    else if(X < -P)
        X = Q * (X + P);
    else
        X = 0;
    if(is_averaged)
        lazy_average = X;
    return lazy_average;
}

double regularizer::proximal_operator(int _regular, double& _prox, double step_size
    , double* lambda) {
    switch(_regular) {
        case regularizer::L1: {
            double param = step_size * lambda[1];
            if(_prox > param)
                _prox -= param;
            else if(_prox < -param)
                _prox += param;
            else
                _prox = 0;
            return _prox;
        }
        case regularizer::L2: {
            _prox = _prox / (1 + step_size * lambda[0]);
            return _prox;
        }
        case regularizer::ELASTIC_NET: {
            double param_1 = step_size * lambda[1];
            double param_2 = 1.0 / (1.0 + step_size * lambda[0]);
            if(_prox > param_1)
                _prox = param_2 * (_prox - param_1);
            else if(_prox < - param_1)
                _prox = param_2 * (_prox + param_1);
            else
                _prox = 0;
            return _prox;
            break;
        }
        default:
            return 0.0;
            break;
    }
}

// Lazy(Lagged) Update
double regularizer::proximal_operator(int _regular, double& _prox, double step_size
    , double* lambda, size_t times, bool is_averaged, double C) {
    double lazy_average = 0.0;
    switch(_regular) {
        case regularizer::L1: {
            // return L1_proximal_loop(_prox, step_size * lambda[1], times, C, is_averaged);
            // New DnC Method
            double P = step_size * lambda[1];
            double X = _prox;
            size_t K = times;
            if(C >= P || C <= -P) {
                bool flag = false;
                // Symmetric Case
                if(C < -P) {
                    flag = true;
                    C = -C;
                    X = -_prox;
                }
                while(X < P - C && K > 0) {
                    double thres = ceil((-P - C - X) / (P + C));
                    if(K <= thres) {
                        if(is_averaged)
                            lazy_average = K * X + (P + C) * (1 + K) * K / 2.0;
                        _prox = X + K * (P + C);
                        // Symmetric Case
                        if(flag) {
                            _prox = -_prox;
                            lazy_average = -lazy_average;
                        }
                        return lazy_average;
                    }
                    else if(thres > 0.0){
                        if(is_averaged)
                            lazy_average += thres * X + (P + C) * (1 + thres) * thres / 2.0;
                        X += thres * (P + C);
                        K -= thres;
                    }
                    lazy_average += L1_single_step(X, P, C, is_averaged);
                    K --;
                }
                if(K == 0) {
                    _prox = X;
                    // Symmetric Case
                    if(flag) {
                        _prox = -_prox;
                        lazy_average = -lazy_average;
                    }
                    return lazy_average;
                }
                _prox = X + K * (C - P);
                if(is_averaged)
                    lazy_average += K * X + (C - P) * (1 + K) * K / 2.0;
                // Symmetric Case
                if(flag) {
                    lazy_average = -lazy_average;
                    _prox = -_prox;
                }
                return lazy_average;
            }
            else {
                double thres_1 = max(ceil((P - C - X) / (C - P)), 0.0);
                double thres_2 = max(ceil((-P - C - X) / (C + P)), 0.0);
                if(thres_2 == 0 && thres_1 == 0) {
                    _prox = 0;
                    return 0;
                }
                else if(K > thres_1 && K > thres_2) {
                    _prox = 0;
                    if(thres_1 != 0.0 && is_averaged)
                        lazy_average = thres_1 * X + (C - P) * (1 + thres_1) * thres_1 / 2.0;
                    else if(is_averaged)
                        lazy_average = thres_2 * X + (P + C) * (1 + thres_2) * thres_2 / 2.0;
                }
                else {
                    if(X > 0) {
                        if(is_averaged)
                            lazy_average = K * X + (C - P) * (1 + K) * K / 2.0;
                        _prox = X + K * (C - P);
                    }
                    else {
                        if(is_averaged)
                            lazy_average = K * X + (P + C) * (1 + K) * K / 2.0;
                        _prox = X + K * (P + C);
                    }
                }
                return lazy_average;
            }
            break;
        }
        case regularizer::L2: {
            if(times == 1) {
                _prox = (_prox + C) / (1 + step_size * lambda[0]);
                return _prox;
            }
            if(lambda[0] != 0.0) {
                double param_1 = step_size * lambda[0];
                double param_2 = pow((double) 1.0 / (1 + param_1), (double) times);
                double param_3 = C / param_1;
                if(is_averaged)
                    lazy_average = (_prox - param_3) * (1 - param_2) / param_1 + param_3 * times;
                _prox = _prox * param_2 + param_3 * (1 - param_2);
                return lazy_average;
            }
            else {
                if(is_averaged)
                    lazy_average = times * _prox + C * (1 + times) * times / 2.0;
                _prox = _prox + times * C;
                return lazy_average;
            }
            break;
        }
        case regularizer::ELASTIC_NET: {
            // New DnC Method
            double P = step_size * lambda[1];
            double Q = 1.0 / (1.0 + step_size * lambda[0]);
            double X = _prox;
            size_t K = times;
            if(C >= P || C <= -P) {
                bool flag = false;
                // Symmetric Case
                if(C < -P) {
                    flag = true;
                    C = -C;
                    X = -_prox;
                }
                double ratio_PCQ = (P + C) * Q / (1 - Q);
                double ratio_NPCQ = (P - C) * Q / (1 - Q);
                while(X < P - C && K > 0) {
                    double thres = ceil(log((P + C + ratio_PCQ) / (ratio_PCQ - X)) / log(Q));
                    if(K <= thres) {
                        double pow_QK = pow((double) Q, (double) K);
                        if(is_averaged)
                            lazy_average = equal_ratio(Q, pow_QK, K) * (X - ratio_PCQ)
                                         + ratio_PCQ * K;
                        _prox = pow_QK * X + ratio_PCQ * (1 - pow_QK);
                        // Symmetric Case
                        if(flag) {
                            _prox = -_prox;
                            lazy_average = -lazy_average;
                        }
                        return lazy_average;
                    }
                    else if(thres > 0.0){
                        double pow_Qtrs = pow((double) Q, (double) thres);
                        if(is_averaged)
                            lazy_average = equal_ratio(Q, pow_Qtrs, thres) * (X - ratio_PCQ)
                                         + ratio_PCQ * thres;
                        X = pow_Qtrs * X + ratio_PCQ * (1 - pow_Qtrs);
                        K -= thres;
                    }
                    lazy_average += EN_single_step(X, P, Q, C, is_averaged);
                    K --;
                }
                if(K == 0) {
                    _prox = X;
                    // Symmetric Case
                    if(flag) {
                        _prox = -_prox;
                        lazy_average = -lazy_average;
                    }
                    return lazy_average;
                }
                double pow_QK = pow((double) Q, (double) K);
                _prox = pow_QK * X - ratio_NPCQ * (1 - pow_QK);
                if(is_averaged)
                    lazy_average += equal_ratio(Q, pow_QK, K) * (X + ratio_NPCQ)
                                         - ratio_NPCQ * K;
                // Symmetric Case
                if(flag) {
                    lazy_average = -lazy_average;
                    _prox = -_prox;
                }
                return lazy_average;
            }
            else {
                double ratio_PCQ = (P + C) * Q / (1 - Q);
                double ratio_NPCQ = (P - C) * Q / (1 - Q);
                double thres_1 = max(ceil(log((P - C + ratio_NPCQ) / (ratio_NPCQ + X)) / log(Q)), 0.0); // P-C
                double thres_2 = max(ceil(log((P + C + ratio_PCQ) / (ratio_PCQ - X)) / log(Q)), 0.0); // -P-C
                if(thres_2 == 0 && thres_1 == 0) {
                    _prox = 0;
                    return 0;
                }
                else if(K > thres_1 && K > thres_2) {
                    _prox = 0;
                    double pow_Qtrs1 = pow((double) Q, (double) thres_1);
                    double pow_Qtrs2 = pow((double) Q, (double) thres_2);
                    if(thres_1 != 0.0 && is_averaged)
                        lazy_average = equal_ratio(Q, pow_Qtrs1, thres_1) * (X + ratio_NPCQ)
                                         - ratio_NPCQ * thres_1;
                    else if(is_averaged)
                        lazy_average = equal_ratio(Q, pow_Qtrs2, thres_2) * (X - ratio_PCQ)
                                         + ratio_PCQ * thres_2;
                }
                else {
                    double pow_QK = pow((double) Q, (double) K);
                    if(X > 0) {
                        if(is_averaged)
                            lazy_average = equal_ratio(Q, pow_QK, K) * (X + ratio_NPCQ)
                                         - ratio_NPCQ * K;
                        _prox = pow_QK * X - ratio_NPCQ * (1 - pow_QK);
                    }
                    else {
                        if(is_averaged)
                            lazy_average = equal_ratio(Q, pow_QK, K) * (X - ratio_PCQ)
                                         + ratio_PCQ * K;
                        _prox = pow_QK * X + ratio_PCQ * (1 - pow_QK);
                    }
                }
                return lazy_average;
            }
            break;
        }
        default:
            return 0.0;
            break;
    }
}

double regularizer::Naive_Momentum_L2_lazy_update(double& x, size_t k, double A, double B, double a, double& x0, double& x1) {
    double aver = 0;
    for(size_t i = 0; i < k; i ++) {
        x0 = x1;
        x1 = x;
        x = A * (x1 + a) + B * (x0 + a) - a;
        aver += x1;
    }
    return aver;
}

double regularizer::Momentum_L2_lazy_update(double& x, size_t k, double A, double B, double a, double& x0, double& x1) {
    if(k <= 4)
        return Naive_Momentum_L2_lazy_update(x, k, A, B, a, x0, x1);
    x0 = x1;
    x1 = x;
    std::complex<double> y0(x0 + a);
    std::complex<double> y1(x1 + a);
    std::complex<double> root_term(A * A + 4 * B);
    std::complex<double> root1 = (std::complex<double>(A) + sqrt(root_term))
        / std::complex<double>(2.0);
    std::complex<double> root2 = (std::complex<double>(A) - sqrt(root_term))
        / std::complex<double>(2.0);
    std::complex<double> s1 = (y0 * root2 - y1) / (root2 - root1);
    std::complex<double> s2 = (y1 - y0 * root1) / (root2 - root1);

    std::complex<double> rk1 = pow(root1, k - 1), rk2 = pow(root2, k - 1);
    double aver = std::real(s1 * root1 * root1 / (std::complex<double>(1) - root1)
            * (std::complex<double>(1) - rk1)
             + s2 * root2 * root2 / (std::complex<double>(1) - root2)
            * (std::complex<double>(1) - rk2)) - (k - 1) * a + x1;
    x0 = std::real(s1 * rk1 + s2 * rk2);
    x1 = std::real(s1 * rk1 * root1 + s2 * rk2 * root2);
    x =  A * x1 + B * x0 - a;
    x0 -= a;
    x1 -= a;
    return aver;
}
