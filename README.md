# SimpleSig

Fast path signature computation for multivariate time series and rough path analysis in R.

## Overview

Path signatures are sequences of iterated integrals that uniquely characterize paths and have proven valuable across multiple domains:

- **Machine Learning**: Feature extraction from sequential data
- **Quantitative Finance**: Analysis of price paths and market data  
- **Time Series Analysis**: Capturing complex temporal dependencies
- **Pattern Recognition**: Invariant features for classification

## Installation

You can install the development version of SimpleSig from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("sjvrensburg/SimpleSig")
```

## Quick Start

```r
library(SimpleSig)

# Create a simple 2D random walk
set.seed(123)
path <- apply(matrix(rnorm(200), ncol = 2), 2, cumsum)

# Compute signature up to order 3
signature_flat <- sig(path, m = 3, flat = TRUE)
signature_list <- sig(path, m = 3, flat = FALSE)

# Check signature dimensions
dims <- sig_dimensions(d = 2, m = 3)  # c(2, 4, 8)
print(paste("Total signature length:", sum(dims)))
```

## Key Features

- **Efficient Implementation**: Uses Chen's identity for fast computation
- **Parallel Processing**: OpenMP support for multi-core performance
- **Flexible Output**: Returns flattened vectors or structured lists
- **Robust Validation**: Comprehensive input checking and error handling
- **Memory Optimized**: Handles large paths with careful memory management

## Usage Examples

### Financial Time Series

```r
# Stock price analysis
returns <- diff(log(stock_prices))
price_path <- apply(returns, 2, cumsum)
financial_signature <- sig(price_path, m = 4)
```

### Multi-dimensional Trajectories

```r
# 3D trajectory data
trajectory <- array(rnorm(300), dim = c(100, 3))
trajectory_signature <- sig(trajectory, m = 3, flat = FALSE)

# Access different signature levels
level_1 <- trajectory_signature[[1]]  # Linear features
level_2 <- trajectory_signature[[2]]  # Quadratic interactions
level_3 <- trajectory_signature[[3]]  # Cubic interactions
```

## Technical Details

### Signature Computation

For a path **X** in **R**^d, the signature up to order **m** consists of:
- **S¹(X)**: Path increments (dimension d)
- **S²(X)**: Second-order iterated integrals (dimension d²)
- **S^m(X)**: m-th order iterated integrals (dimension d^m)

Total signature dimension: **d + d² + ... + d^m**

### Performance Considerations

- **Path dimension**: Limited to d ≤ 20
- **Signature order**: Limited to m ≤ 10  
- **Memory usage**: Grows as sum(d^k) for k=1 to m
- **Parallelization**: Automatic for higher orders and longer paths

## Function Reference

### `sig(path, m, flat = TRUE)`

Compute path signature up to order m.

**Parameters:**
- `path`: Numeric matrix (n × d) where rows are time points, columns are dimensions
- `m`: Maximum signature order (1-10)
- `flat`: Return flattened vector (TRUE) or list of levels (FALSE)

**Returns:**
- If `flat=TRUE`: Numeric vector of length sum(d^k) for k=1 to m
- If `flat=FALSE`: List with m elements, each containing signature level k

### `sig_dimensions(d, m)`

Calculate signature dimensions for given path dimension and order.

**Parameters:**
- `d`: Path dimension
- `m`: Maximum signature order

**Returns:** Integer vector of signature dimensions at each level

## Mathematical Background

Path signatures are grounded in rough path theory and provide a principled way to extract features from sequential data. They satisfy important properties:

- **Uniqueness**: Different paths have different signatures (with probability 1)
- **Concatenation**: Chen's identity enables efficient computation
- **Invariance**: Natural under certain transformations
- **Universality**: Can approximate any continuous function of the path

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests on [GitHub](https://github.com/sjvrensburg/SimpleSig).

## License

GPL (>= 3)
