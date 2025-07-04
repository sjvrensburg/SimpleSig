// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// sig
Rcpp::RObject sig(Rcpp::NumericMatrix path, int m, bool flat);
RcppExport SEXP _SimpleSig_sig(SEXP pathSEXP, SEXP mSEXP, SEXP flatSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type path(pathSEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< bool >::type flat(flatSEXP);
    rcpp_result_gen = Rcpp::wrap(sig(path, m, flat));
    return rcpp_result_gen;
END_RCPP
}
// sig_dimensions
Rcpp::IntegerVector sig_dimensions(int d, int m);
RcppExport SEXP _SimpleSig_sig_dimensions(SEXP dSEXP, SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type d(dSEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    rcpp_result_gen = Rcpp::wrap(sig_dimensions(d, m));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_SimpleSig_sig", (DL_FUNC) &_SimpleSig_sig, 3},
    {"_SimpleSig_sig_dimensions", (DL_FUNC) &_SimpleSig_sig_dimensions, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_SimpleSig(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
