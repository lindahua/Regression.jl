# common facilities

function augment_rows{T}(X::StridedMatrix{T}, v::T)
    m, n = size(X)
    Xa = Array(T, m, n+1)
    copy!(view(Xa, :, 1:n), X)
    fill!(view(Xa, :, n+1), v)
    Xa
end

function augment_cols{T}(X::StridedMatrix{T}, v::T)
    m, n = size(X)
    Xa = Array(T, m+1, n)
    copy!(view(Xa, 1:m, :), X)
    fill!(rowvec_view(Xa, m+1), v)
    Xa
end
