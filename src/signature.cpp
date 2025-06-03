#include <Rcpp.h>
#include <vector>
#include <omp.h>

// [[Rcpp::plugins(openmp)]]

//' Compute Path Signature
//'
//' Computes the signature of a path up to a specified order. The signature is a
//' sequence of iterated integrals that characterizes the path and has useful
//' properties for machine learning, quantitative finance, and time series analysis.
//'
//' @param path A numeric matrix with dimensions (n x d), where n is the number
//'   of time steps and d is the path dimension. Each row represents a time point
//'   and each column represents a dimension (standard time series format).
//' @param m An integer specifying the maximum order of the signature (1 <= m <= 10).
//' @param flat A logical value. If TRUE (default), returns a flattened numeric vector
//'   containing all signature levels concatenated. If FALSE, returns a list with
//'   each signature level as a separate element.
//'
//' @return If flat=TRUE, returns a numeric vector of length sum(d^k) for k=1 to m.
//'   If flat=FALSE, returns a list of length m, where element k contains a numeric
//'   vector of length d^k representing the k-th order signature.
//'
//' @details
//' The signature computation uses Chen's identity for efficient combination of
//' path segments. The function includes OpenMP parallelization and optimized
//' memory management.
//'
//' Memory usage grows as sum(d^k) for k=1 to m. For large dimensions or orders,
//' consider using lower values to avoid memory issues.
//'
//' The input follows standard time series convention: rows are time points,
//' columns are variables/dimensions.
//'
//' @examples
//' \dontrun{
//' # Create a simple 2D path (100 time points, 2 dimensions)
//' path <- matrix(rnorm(200), ncol = 2)
//'
//' # Or using cumulative sums for more realistic paths
//' path <- apply(matrix(rnorm(200), ncol = 2), 2, cumsum)
//'
//' # Compute signature up to order 3
//' sig_flat <- sig(path, m = 3, flat = TRUE)
//' sig_list <- sig(path, m = 3, flat = FALSE)
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::RObject sig(Rcpp::NumericMatrix path, int m, bool flat = true) {

    // Input validation
    if (m < 1 || m > 10) {
        Rcpp::stop("Order m must be between 1 and 10");
    }

    int n = path.nrow(); // Number of time steps (rows)
    int d = path.ncol(); // Dimension of path (columns)

    if (d < 1 || d > 20) {
        Rcpp::stop("Path dimension d must be between 1 and 20");
    }
    if (n < 2) {
        Rcpp::stop("Path must have at least 2 time points");
    }

    // Check for invalid values
    if (Rcpp::is_true(Rcpp::any(Rcpp::is_na(path))) ||
        Rcpp::is_true(Rcpp::any(!Rcpp::is_finite(path)))) {
      Rcpp::stop("Path matrix contains NA, NaN, or infinite values");
    }

    // Precompute powers of d
    std::vector<int> pow_d(m + 1);
    pow_d[0] = 1;
    for (int k = 1; k <= m; ++k) {
        pow_d[k] = pow_d[k - 1] * d;
    }

    // Precompute factorials
    std::vector<double> factorials(m + 1);
    factorials[0] = factorials[1] = 1.0;
    for (int k = 2; k <= m; ++k) {
        factorials[k] = factorials[k - 1] * k;
    }

    // Compute increments (note the changed indexing)
    std::vector<std::vector<double>> diffs(n - 1, std::vector<double>(d));
    for (int j = 0; j < n - 1; ++j) {
        for (int i = 0; i < d; ++i) {
            diffs[j][i] = path(j + 1, i) - path(j, i);  // Changed: path(row, col)
        }
    }

    // Compute tensor powers for each segment in parallel
    std::vector<std::vector<std::vector<double>>> r(n - 1, std::vector<std::vector<double>>(m));

    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < n - 1; ++j) {
        // Order 1: Copy increment
        r[j][0] = diffs[j];

        // Higher orders: Compute Kronecker products
        for (int k = 1; k < m; ++k) {
            int len = pow_d[k + 1];
            r[j][k].resize(len);

            // Kronecker product of r[j][k-1] with diffs[j]
            int prev_len = pow_d[k];
            for (int i = 0; i < prev_len; ++i) {
                for (int l = 0; l < d; ++l) {
                    r[j][k][i * d + l] = r[j][k - 1][i] * diffs[j][l];
                }
            }

            // Normalize by factorial
            double factorial_inv = 1.0 / factorials[k + 1];
            for (int i = 0; i < len; ++i) {
                r[j][k][i] *= factorial_inv;
            }
        }
    }

    // Chen's identity implementation
    auto chen = [&](const std::vector<std::vector<double>>& x,
                     const std::vector<std::vector<double>>& y) {
        std::vector<std::vector<double>> result(m);

        // Order 1: Sum increments
        result[0].resize(d);
        for (int i = 0; i < d; ++i) {
            result[0][i] = x[0][i] + y[0][i];
        }

        // Higher orders
        #pragma omp parallel for schedule(dynamic) if(m > 3)
        for (int k = 1; k < m; ++k) {
            int len = pow_d[k + 1];
            result[k].resize(len);

            // x[k] + y[k]
            for (int i = 0; i < len; ++i) {
                result[k][i] = x[k][i] + y[k][i];
            }

            // Correction term: sum of kronecker(x[i], y[k-i-1]) for i = 0 to k-1
            std::vector<double> correction(len, 0.0);

            for (int i = 0; i < k; ++i) {
                int len_x = pow_d[i + 1];
                int len_y = pow_d[k - i];

                for (int ix = 0; ix < len_x; ++ix) {
                    for (int iy = 0; iy < len_y; ++iy) {
                        int idx = ix * len_y + iy;
                        correction[idx] += x[i][ix] * y[k - i - 1][iy];
                    }
                }
            }

            // Add correction to result
            for (int i = 0; i < len; ++i) {
                result[k][i] += correction[i];
            }
        }
        return result;
    };

    // Reduce using Chen's identity
    std::vector<std::vector<double>> signature = r[0];
    for (int j = 1; j < n - 1; ++j) {
        signature = chen(signature, r[j]);
    }

    // Format output
    if (flat) {
        int total_len = 0;
        for (int k = 0; k < m; ++k) total_len += pow_d[k + 1];

        Rcpp::NumericVector result(total_len);
        int pos = 0;
        for (int k = 0; k < m; ++k) {
            int len = pow_d[k + 1];
            for (int i = 0; i < len; ++i) {
                result[pos++] = signature[k][i];
            }
        }
        return result;
    }

    // Return as list
    Rcpp::List result(m);
    for (int k = 0; k < m; ++k) {
        result[k] = Rcpp::NumericVector(signature[k].begin(), signature[k].end());
    }
    return result;
}

//' Get Signature Dimensions
//'
//' Returns the dimensions of each signature level for given path dimension and order.
//'
//' @param d An integer specifying the path dimension.
//' @param m An integer specifying the maximum signature order.
//'
//' @return An integer vector of length m, where element k contains the dimension
//'   of the k-th order signature (d^k).
//'
//' @examples
//' \dontrun{
//' dims <- sig_dimensions(3, 4)  # c(3, 9, 27, 81)
//' total_length <- sum(dims)     # 120
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::IntegerVector sig_dimensions(int d, int m) {
    if (d < 1 || m < 1) {
        Rcpp::stop("Both d and m must be positive integers");
    }

    Rcpp::IntegerVector dims(m);
    long long pow_d_k = d;
    for (int k = 0; k < m; ++k) {
        if (pow_d_k > INT_MAX) {
            Rcpp::stop("Dimension overflow at order %d", k + 1);
        }
        dims[k] = static_cast<int>(pow_d_k);
        if (k < m - 1) pow_d_k *= d;
    }
    return dims;
}
