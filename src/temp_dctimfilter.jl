### DCT-7 recursive filtering
# K. Sugimoto, S. Kyochi and S. Kamata, "Universal approach for dct-based constant-time 
# gaussian filter with moment preservation". IEEE international conference on acoustics,
# speech and signal processing (ICASSP) 1498-1502 (2018).

# Note this is safe for inplace use, i.e., out === img

function imfilter!(r::AbstractResource{DCT},
    out::AbstractArray{S,N},
    img::AbstractArray{T,N},
    kernel::Tuple{DCT_7,Vararg{DCT_7}},
    border::BorderSpec) where {S,T,N}
    #isa(border, Pad) && border.style != :replicate && throw(ArgumentError("only \"replicate\" is supported"))
    length(kernel) <= N || throw(DimensionMismatch("cannot have more kernels than dimensions"))
    inds = axes(img)
    _imfilter_inplace_tuple!(r, out, img, kernel, CartesianIndices(()), inds, CartesianIndices(tail(inds)), border)
end

"""
    imfilter!(r::AbstractResource, imgfilt, img, kernel::Tuple{DCT_7...}, border)
    imfilter!(r::AbstractResource, imgfilt, img, kernel::DCT_7, dim::Integer, border)

Filter an array `img` with a discrete cosine transform 7 (DCT-7) 
`kernel`, storing the result in `imgfilt`. Unlike the `FIR` and
`FFT` algorithms, this version is safe for inplace operations, i.e.,
`imgfilt` can be the same array as `img`.

Either specify one kernel per dimension (as a tuple), or a particular
dimension `dim` along which to filter. If you exhaust `kernel`s before
you run out of array dimensions, the remaining dimension(s) will not
be filtered.

See also: [`imfilter`](@ref), [`KernelFactors.DCT_7`](@ref), [`KernelFactors.DCTGaussian`](@ref).
"""
function imfilter!(r::AbstractResource, out::AbstractArray, img::AbstractArray, kernel::DCT_7, dim::Integer, border::BorderSpec)
    inds = axes(img)
    k, l = length(kernel.a), length(kernel.b)
    # This next part is not type-stable, which is why _imfilter_dim! has a @noinline
    Rbegin = CartesianIndices(inds[1:dim-1])
    Rend = CartesianIndices(inds[dim+1:end])
    _imfilter_dim!(r, out, img, kernel, Rbegin, inds[dim], Rend, border)
end
function imfilter!(r::AbstractResource, out::AbstractArray, img::AbstractArray, kernel::DCT_7, dim::Integer, border::AbstractString)
    imfilter!(r, out, img, kernel, dim, Pad(Symbol(border)))
end


function imfilter!(r::AbstractResource, out::AbstractArray, A::AbstractVector, kern::DCT_7, border::NoPad, inds::Indices=axes(out))
    indspre, ind, indspost = iterdims(inds, kern)
    _imfilter_dim!(r, out, A, kern, CartesianIndices(indspre), ind, CartesianIndices(indspost), border[])
end

# Lispy and type-stable inplace (currently just Triggs-Sdika) filtering over each dimension
function _imfilter_inplace_tuple!(r, out, img, kernel, Rbegin, inds, Rend, border)
    ind = first(inds)
    _imfilter_dim!(r, out, img, first(kernel), Rbegin, ind, Rend, border)
    _imfilter_inplace_tuple!(r,
        out,
        out,
        tail(kernel),
        CartesianIndices((Rbegin.indices..., ind)),
        tail(inds),
        _tail(Rend),
        border)
end
# When the final kernel has been used, return the output
_imfilter_inplace_tuple!(r, out, img, ::Tuple{}, Rbegin, inds, Rend, border) = out

# This is the "workhorse" function that performs DCT
# filtering along a particular dimension. The "pre" dimensions are
# encoded in Rbegin, the "post" dimensions in Rend, and the dimension
# we're filtering is sandwiched between these. This design is
# type-stable and cache-friendly for any dimension---we update values
# in memory-order rather than along the chosen dimension. Nor does it
# require that the arrays have efficient linear indexing. For more
# information, see http://julialang.org/blog/2016/02/iteration.
@noinline function _imfilter_dim!(r::AbstractResource,
    out, img, kernel::DCT_7{T},
    Rbegin::CartesianIndices, ind::AbstractUnitRange,
    Rend::CartesianIndices, border::AbstractBorder) where T

    @noinline function throw_imfilter_dim(R, n, l)
        dim = ndims(R) + 1
        throw(DimensionMismatch("size $n of img along dimension $dim is too small for filtering with IIR kernel of length $l"))
    end

    if iscopy(kernel)
        if !(out === img)
            copyto!(out, img)
        end
        return out
    end
    if length(ind) <= max(k, l)
        throw_imfilter_dim(Rbegin, length(ind), max(k, l))
    end
    indleft = ind[firstindex(ind):firstindex(ind)+k-1]
    indright = ind[end-l+1:end]
    for Iend in Rend
        # Initialize the left border
        for Ibegin in Rbegin
            leftborder!(out, img, kernel, Ibegin, indleft, Iend, border)
        end
        # Propagate forwards. We omit the final point in case border
        # is "replicate", so that the original value is still
        # available. rightborder! will handle that point.
        for i = range(first(ind) + k, stop=ind[end-1])
            @inbounds for Ibegin in Rbegin
                tmp = accumfilter(img[Ibegin, i, Iend], one(T))
                for j = 1:k
                    tmp += kernel.a[j] * safe_for_prod(out[Ibegin, i-j, Iend], tmp)
                end
                out[Ibegin, i, Iend] = tmp
            end
        end
        # Initialize the right border
        for Ibegin in Rbegin
            rightborder!(out, img, kernel, Ibegin, indright, Iend, border)
        end
        # Propagate backwards
        for i = ind[end-l]:-1:first(ind)
            @inbounds for Ibegin in Rbegin
                tmp = accumfilter(out[Ibegin, i, Iend], one(T))
                for j = 1:l
                    tmp += kernel.b[j] * safe_for_prod(out[Ibegin, i+j, Iend], tmp)
                end
                out[Ibegin, i, Iend] = tmp
            end
        end
        # Final scaling
        for i in ind
            @inbounds for Ibegin in Rbegin
                out[Ibegin, i, Iend] *= kernel.scale
            end
        end
    end
    out
end

# Implements the initialization in the first paragraph of Triggs & Sdika, section II
function leftborder!(out, img, kernel, Ibegin, indleft, Iend, border::Fill)
    _leftborder!(out, img, kernel, Ibegin, indleft, Iend, convert(eltype(img), border.value))
end
function leftborder!(out, img, kernel, Ibegin, indleft, Iend, border::Pad)
    _leftborder!(out, img, kernel, Ibegin, indleft, Iend, img[Ibegin, indleft[1], Iend])
end
function _leftborder!(out, img, kernel::DCT_7{T,k,l}, Ibegin, indleft, Iend, iminus) where {T,k,l}
    uminus = iminus / (1 - kernel.asum)
    n = 0
    for i in indleft
        n += 1
        tmp = accumfilter(img[Ibegin, i, Iend], one(T))
        for j = 1:n-1
            tmp += kernel.a[j] * safe_for_prod(out[Ibegin, i-j, Iend], tmp)
        end
        for j = n:k
            tmp += kernel.a[j] * uminus
        end
        out[Ibegin, i, Iend] = tmp
    end
    out
end

# Implements Triggs & Sdika, Eqs 14-15
function rightborder!(out, img, kernel, Ibegin, indright, Iend, border::Fill)
    _rightborder!(out, img, kernel, Ibegin, indright, Iend, convert(eltype(img), border.value))
end
function rightborder!(out, img, kernel, Ibegin, indright, Iend, border::Pad)
    _rightborder!(out, img, kernel, Ibegin, indright, Iend, img[Ibegin, indright[end], Iend])
end
function _rightborder!(out, img, kernel::DCT_7{T,k,l}, Ibegin, indright, Iend, iplus) where {T,k,l}
    # The final value from forward-filtering was not calculated, so do that here
    i = last(indright)
    tmp = accumfilter(img[Ibegin, i, Iend], one(T))
    for j = 1:k
        tmp += kernel.a[j] * safe_for_prod(out[Ibegin, i-j, Iend], tmp)
    end
    out[Ibegin, i, Iend] = tmp
    # Initialize the v values at and beyond the right edge
    uplus = iplus / (1 - kernel.asum)
    vplus = uplus / (1 - kernel.bsum)
    vright = kernel.M * rightΔu(out, uplus, Ibegin, last(indright), Iend, kernel) .+ vplus
    out[Ibegin, last(indright), Iend] = vright[1]
    # Propagate inward
    n = 1
    for i in last(indright)-1:-1:first(indright)
        n += 1
        tmp = accumfilter(out[Ibegin, i, Iend], one(T))
        for j = 1:n-1
            tmp += kernel.b[j] * safe_for_prod(out[Ibegin, i+j, Iend], tmp)
        end
        for j = n:l
            tmp += kernel.b[j] * safe_for_prod(vright[j-n+2], tmp)
        end
        out[Ibegin, i, Iend] = tmp
    end
    out
end

# Part of Triggs & Sdika, Eq. 14
function rightΔu(img, uplus, Ibegin, i, Iend, kernel::DCT_7{T}) where T
    @inbounds ret = SVector(img[Ibegin, i, Iend] - uplus,
        img[Ibegin, i-1, Iend] - uplus,
        img[Ibegin, i-2, Iend] - uplus)
    ret
end

### NA boundary conditions

function imfilter_na_inseparable!(r, out::AbstractArray{T}, img, naflag, kernel::Tuple{Vararg{AnyDCT}}) where {T}
    fc, fn = Fill(zero(T)), Fill(zero(eltype(T)))  # color, numeric
    copyto!(out, img)
    out[naflag] .= zero(T)
    validpixels = copyto!(similar(Array{eltype(T)}, axes(img)), mappedarray(!, naflag))
    # DCT_7 is safe for inplace operations
    imfilter!(r, out, out, kernel, fc)
    imfilter!(r, validpixels, validpixels, kernel, fn)
    for I in eachindex(out)
        out[I] /= validpixels[I]
    end
    out
end

function imfilter_na_inseparable!(r, out::AbstractArray{T}, img, naflag, kernel::Tuple) where {T}
    fc, fn = Fill(zero(T)), Fill(zero(eltype(T)))  # color, numeric
    imgtmp = copyto!(similar(out, axes(img)), img)
    imgtmp[naflag] .= Ref(zero(T))
    validpixels = copyto!(similar(Array{eltype(T)}, axes(img)), mappedarray(x -> !x, naflag))
    imfilter!(r, out, imgtmp, kernel, fc)
    vp = imfilter(r, validpixels, kernel, fn)
    for I in eachindex(out)
        out[I] /= vp[I]
    end
    out
end

function imfilter_na_separable!(r, out::AbstractArray{T}, img, kernel::Tuple) where {T}
    fc, fn = Fill(zero(T)), Fill(zero(eltype(T)))  # color, numeric
    imfilter!(r, out, img, kernel, fc)
    normalize_separable!(r, out, kernel, fn)
end