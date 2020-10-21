#ifndef GRADDESCDENSE_H
#define GRADDESCDENSE_H
#include "blackbox.hpp"

namespace grad_desc_dense {
    class outputs {
    public:
        outputs(std::vector<double>* _losses, std::vector<double>* _times):
            losses(_losses), times(_times) {}
        std::vector<double>* losses;
        std::vector<double>* times;
    };

    // Deterministic Methods
    // For Uni_Acc, Mode = 0 is NAG; Mode = 1 is TM.
    outputs Uni_Acc(double* X, double* Y, size_t N, blackbox* model, size_t iteration_no
        , double L, double mu, int Mode = 0);
    outputs G_TM(double* X, double* Y, size_t N, blackbox* model, size_t iteration_no
        , double L, double mu);

    // Stochastic Methods
    outputs SAGA(double* X, double* Y, size_t N, blackbox* model, size_t iteration_no
        , double step_size);
    outputs Katyusha(double* X, double* Y, size_t N, blackbox* model
        , size_t iteration_no, double L, double mu, double tau_1);
    // (choice = -1) = Using numerical choice
    outputs BS_SVRG(double* X, double* Y, size_t N, blackbox* model
        , size_t iteration_no, double L, double mu, double alpha, double choice);
}

#endif
