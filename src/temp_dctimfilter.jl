# This is the "workhorse" function that performs DCT/DST
# filtering along a particular dimension. The "pre" dimensions are
# encoded in Rbegin, the "post" dimensions in Rend, and the dimension
# we're filtering is sandwiched between these. This design is
# type-stable and cache-friendly for any dimension---we update values
# in memory-order rather than along the chosen dimension. Nor does it
# require that the arrays have efficient linear indexing. For more
# information, see http://julialang.org/blog/2016/02/iteration.
@noinline function _imfilter_dim!(r::AbstractResource,
    out, img, kernel::AnyDCT_DST{T},
    Rbegin::CartesianIndices, ind::AbstractUnitRange,
    Rend::CartesianIndices, border::AbstractBorder) where T

    @noinline function throw_imfilter_dim(R, n, l)
        dim = ndims(R) + 1
        throw(DimensionMismatch("size $n of img along dimension $dim is too small for filtering with DCT kernel of length $l"))
    end

    if iscopy(kernel)
        if !(out === img)
            copyto!(out, img)
        end
        return out
    end
    R = length(kernel)
    if length(ind) <= R
        throw_imfilter_dim(Rbegin, length(ind), R)
    end

    # Create a buffer array to allows safe inplace operations for any border instance 
    buffer = similar(out, eltype(out), R + 1)

    # Create a matrix that will store for each K coefficients the last and penultimate signal weighted cosine/sine basis
    Z = similar(out, eltype(out), (size(kernel.basis, 2), 2))
    for Iend in Rend
        # Initialize the 2 first filtered coefficients
        for Ibegin in Rbegin
            dctdst_2first!(buffer, Z, img, kernel, Ibegin, ind, Iend, border)
        end

        # Propagate forwards
        for i = range(first(ind) + k, stop=ind[end-1])
            @inbounds for Ibegin in Rbegin
                tmp = accumfilter(img[Ibegin, i, Iend], one(T))
                for j = 1:k
                    tmp += kernel.a[j] * safe_for_prod(out[Ibegin, i-j, Iend], tmp)
                end
                out[Ibegin, i, Iend] = tmp
            end
        end
    end
    out
end

# This function would also need to handle "virtual" padding
function dctdst_2first!(buffer, Z, img, kernel, Ibegin, ind, Iend, border::Fill)
    _dctdst_2first_uniqueval!(buffer, Z, img, kernel, Ibegin, ind, Iend, convert(eltype(img), border.value))
end
function dctdst_2first!(buffer, Z, img, kernel, Ibegin, ind, Iend, border::Pad)
    border.style == :replicate && return _dctdst_2first_uniqueval!(buffer, Z, img, kernel, Ibegin, ind, Iend, img[Ibegin, ind[firstindex(ind)], Iend])
    border.style == :circular && return _dctdst_2first_circular!(buffer, Z, img, kernel, Ibegin, ind, Iend)
    border.style == :symmetric && return _dctdst_2first_symmetric!(buffer, Z, img, kernel, Ibegin, ind, Iend)
    border.style == :reflect && return _dctdst_2first_reflect!(buffer, Z, img, kernel, Ibegin, ind, Iend)
end
function _dctdst_2first_uniqueval!(buffer, Z, img, kernel, Ibegin, ind, Iend, borderval)
    C = kernel.basis
    h = kernel.coefficients
    R, K = size(C)
    for k in 1:K
        Z[k, :] .= accumfilter(borderval, sum(view(C, 2:R, k)))
        Z[k, 1] -= accumfilter(borderval, C[2, k]) 
        Z[k, 1] += accumfilter(out[Ibegin, ind[firstindex(ind)], Iend], C[2, k])
    end
    n = 0
    for i in firstindex(ind):(firstindex(ind) + R - 1)
        n += 1
        for k in 1:K
            Z[k, 1] += accumfilter(out[Ibegin, ind[i], Iend], C[n, K])
            Z[k, 2] += accumfilter(out[Ibegin, ind[i + 1], Iend], C[n, K])
        end
    end
    tmp1 = zero(eltype(h))
    tmp2 = zero(eltype(h))
    for k in 1:K
        Z[k, 1] = accumfilter(Z[k, 1], h[k])
        tmp1 += Z[k, 1]
        Z[k, 2] = accumfilter(Z[k, 2], h[k])
        tmp2 += Z[k, 2]
    end
    buffer[1] = tmp1
    buffer[2] = tmp2
end
function _dctdst_2first_circular!(buffer, Z, img, kernel, Ibegin, ind, Iend)
end
function _dctdst_2first_symmetric!(buffer, Z, img, kernel, Ibegin, ind, Iend)
end
function _dctdst_2first_reflect!(buffer, Z, img, kernel, Ibegin, ind, Iend)
end