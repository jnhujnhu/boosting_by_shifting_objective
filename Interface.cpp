#include <iostream>
#include "mex.h"
#include "grad_desc_dense.hpp"
#include "regularizer.hpp"
#include "logistic.hpp"
#include "utils.hpp"
#include <string.h>

size_t MAX_DIM;
const size_t MAX_PARAM_STR_LEN = 15;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    try {
        double *X = mxGetPr(prhs[0]);
        double *Y = mxGetPr(prhs[1]);
        MAX_DIM = mxGetM(prhs[0]);
        size_t N = mxGetN(prhs[0]);
        double *init_weight = mxGetPr(prhs[5]);
        double mu = mxGetScalar(prhs[6]);
        double L = mxGetScalar(prhs[7]);
        double param = mxGetScalar(prhs[8]);
        size_t iteration_no = (size_t) mxGetScalar(prhs[9]);
        int regularizer;
        char* _regul = new char[MAX_PARAM_STR_LEN];
        mxGetString(prhs[4], _regul, MAX_PARAM_STR_LEN);
        if(strcmp(_regul, "L2") == 0) {
            regularizer = regularizer::L2;
        }
        else mexErrMsgTxt("400 Unrecognized regularizer.");
        delete[] _regul;

        blackbox* model;
        char* _model = new char[MAX_PARAM_STR_LEN];
        mxGetString(prhs[3], _model, MAX_PARAM_STR_LEN);
        if(strcmp(_model, "logistic") == 0) {
            model = new logistic(mu, regularizer);
        }
        else mexErrMsgTxt("400 Unrecognized model.");
        //delete[] _model;
        model->set_init_weights(init_weight);

        char* _algo = new char[MAX_PARAM_STR_LEN];
        mxGetString(prhs[2], _algo, MAX_PARAM_STR_LEN);
        double *stored_F, *stored_time;
        size_t len_stored_F, len_stored_time;
        if(strcmp(_algo, "SAGA") == 0) {
            grad_desc_dense::outputs outputs = grad_desc_dense::SAGA(X, Y, N, model, iteration_no, param);
            stored_F = &(*outputs.losses)[0];
            stored_time = &(*outputs.times)[0];
            len_stored_F = outputs.losses->size();
            len_stored_time = outputs.times->size();
        }
        else if(strcmp(_algo, "NAG") == 0) {
            grad_desc_dense::outputs outputs = grad_desc_dense::Uni_Acc(X, Y, N, model, iteration_no, L, mu, 
                0);
            stored_F = &(*outputs.losses)[0];
            stored_time = &(*outputs.times)[0];
            len_stored_F = outputs.losses->size();
            len_stored_time = outputs.times->size();
        }
        else if(strcmp(_algo, "TM") == 0) {
            grad_desc_dense::outputs outputs = grad_desc_dense::Uni_Acc(X, Y, N, model, iteration_no, L, mu, 
                1);
            stored_F = &(*outputs.losses)[0];
            stored_time = &(*outputs.times)[0];
            len_stored_F = outputs.losses->size();
            len_stored_time = outputs.times->size();
        }
        else if(strcmp(_algo, "G_TM") == 0) {
            grad_desc_dense::outputs outputs = grad_desc_dense::G_TM(X, Y, N, model, iteration_no, L, mu);
            stored_F = &(*outputs.losses)[0];
            stored_time = &(*outputs.times)[0];
            len_stored_F = outputs.losses->size();
            len_stored_time = outputs.times->size();
        }
        else if(strcmp(_algo, "Katyusha") == 0) {
            grad_desc_dense::outputs outputs = grad_desc_dense::Katyusha(X, Y, N, model, iteration_no
                , L, mu, param);
            stored_F = &(*outputs.losses)[0];
            stored_time = &(*outputs.times)[0];
            len_stored_F = outputs.losses->size();
            len_stored_time = outputs.times->size();
        }
        else if(strcmp(_algo, "BS_SVRG") == 0) {
            double choice = mxGetScalar(prhs[10]);
            grad_desc_dense::outputs outputs = grad_desc_dense::BS_SVRG(X, Y, N, model, iteration_no
                , L, mu, param, choice);
            stored_F = &(*outputs.losses)[0];
            stored_time = &(*outputs.times)[0];
            len_stored_F = outputs.losses->size();
            len_stored_time = outputs.times->size();
        }
        else mexErrMsgTxt("400 Unrecognized algorithm.");
        delete[] _algo;

        plhs[0] = mxCreateDoubleMatrix(len_stored_time, 1, mxREAL);
        double* res_stored_times = mxGetPr(plhs[0]);
        for(size_t i = 0; i < len_stored_time; i ++)
            res_stored_times[i] = stored_time[i];
        plhs[1] = mxCreateDoubleMatrix(len_stored_F, 1, mxREAL);
        double* res_stored_F = mxGetPr(plhs[1]);
        for(size_t i = 0; i < len_stored_F; i ++)
            res_stored_F[i] = stored_F[i];

        delete[] stored_F;
        delete[] stored_time;
        delete model;
        delete[] _model;
    } catch(std::string c) {
        std::cerr << c << std::endl;
        //exit(EXIT_FAILURE);
    }
}
