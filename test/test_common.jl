# Test of common units

using Regression
using Base.Test

x = [1 2 3; 4 5 6]

@test append_zeros(x, 1) == [1 2 3; 4 5 6; 0 0 0]
@test append_zeros(x, 2) == [1 2 3 0; 4 5 6 0]

@test append_ones(x, 1) == [1 2 3; 4 5 6; 1 1 1]
@test append_ones(x, 2) == [1 2 3 1; 4 5 6 1]

