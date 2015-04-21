
using Regression
using Base.Test

X = randn(6, 8)

@test Regression.augment_rows(X, 2.0) == [X fill(2.0, 6)]
@test Regression.augment_cols(X, 2.0) == [X; fill(2.0, 1, 8)]
