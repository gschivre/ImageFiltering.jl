"""
`KernelFactors` is a module implementing separable filtering kernels,
each stored in terms of their factors. The following kernels are
supported:

  - `box`
  - `sobel`
  - `prewitt`
  - `ando3`, `ando4`, and `ando5` (the latter in 2d only)
  - `scharr`
  - `bickley`
  - `gaussian`
  - `IIRGaussian` (approximate gaussian filtering, fast even for large σ)
  - `DCTGaussian` (approximate gaussian filtering in constant time per pixel, fast even for large σ)
  - `SlidingDCT` (constant-time approximation of an arbitrary even kernel)

See also: [`Kernel`](@ref).
"""
module KernelFactors

using StaticArrays, OffsetArrays
using OffsetArrays: IdentityUnitRange
using ..ImageFiltering: centered, dummyind
import ..ImageFiltering: _reshape, _vec, nextendeddims
using Base: tail, Indices, @pure, checkbounds_indices, throw_boundserror, @propagate_inbounds

# We would like to do `using ..ImageFiltering.Kernel` but we cannot because
# `kernelfactors.jl` is included before  `kernel.jl` in the ImageFiltering.jl
# file. Hence, ImageFiltering.Kernel does not yet exist when this module is
# loaded. The reason we want to have the Kernel module defined is so that that
# Documenter.jl (the documentation system) can parse a reference such as `See
# also: [`Kernel.ando3`](@ref)`. With the more general `using ImageFiltering`,
# we seem to sidestep this scope problem, although I don't actually understand
# the mechanism form why this works. - ZS
using ImageFiltering

abstract type IIRFilter{T} end

Base.eltype(kernel::IIRFilter{T}) where {T} = T

"""
    ReshapedOneD{N,Npre}(data)

Return an object of dimensionality `N`, where `data` must have
dimensionality 1. The axes are `0:0` for the first `Npre`
dimensions, have the axes of `data` for dimension `Npre+1`, and are
`0:0` for the remaining dimensions.

`data` must support `eltype` and `ndims`, but does not have to be an
AbstractArray.

ReshapedOneDs allow one to specify a "filtering dimension" for a
1-dimensional filter.
"""
struct ReshapedOneD{T,N,Npre,V}  # not <: AbstractArray{T,N} (more general, incl. IIR)
    data::V

    function ReshapedOneD{T,N,Npre,V}(data::V) where {T,N,Npre,V}
        ndims(V) == 1 || throw(DimensionMismatch("must be one dimensional, got $(ndims(V))"))
        new{T,N,Npre,V}(data)
    end
end

ReshapedOneD{N,Npre}(data::V) where {N,Npre,V} = ReshapedOneD{eltype(data),N,Npre,V}(data)
# Convenient loop constructor that uses dummy NTuple{N,Bool} to keep
# track of dimensions for type-stability
@inline function ReshapedOneD(pre::NTuple{Npre,Bool}, data, post::NTuple{Npost,Bool}) where {Npre,Npost}
    total = (pre..., true, post...)
    _reshapedvector(total, pre, data)
end
_reshapedvector(::NTuple{N,Bool}, ::NTuple{Npre,Bool}, data) where {N,Npre} = ReshapedOneD{eltype(data),N,Npre,typeof(data)}(data)

# Give ReshapedOneD many of the characteristics of AbstractArray
Base.eltype(A::ReshapedOneD{T}) where {T} = T
Base.ndims(A::ReshapedOneD{_,N}) where {_,N} = N
Base.BroadcastStyle(::Type{ReshapedOneD{T,N,Npre,V}}) where {T,N,Npre,V} = Broadcast.DefaultArrayStyle{N}()
Base.broadcastable(A::ReshapedOneD) = A
Base.isempty(A::ReshapedOneD) = isempty(A.data)

@inline Base.axes(A::ReshapedOneD{_,N,Npre}) where {_,N,Npre} = Base.fill_to_length((Base.ntuple(d->0:0, Val(Npre))..., UnitRange(Base.axes1(A.data))), 0:0, Val(N))

function Base.show(io::IO, ::MIME"text/plain", A::ReshapedOneD{T,N,Npre,<:AbstractVector}) where {T,N,Npre}
    println(io, "Reshaped 1d stencil with axes ", axes(A), ":")
    if ndims(A) <= 2
        # print in matrix style
        a = OffsetArray{T}(undef, axes(A)...)
        for i in CartesianIndices(axes(A))
            a[i] = A[i]
        end
        Base.print_array(io, a)
    else
        println(io, "  with data: ", A.data)
    end
end

Base.iterate(A::ReshapedOneD) = iterate(A.data)
Base.iterate(A::ReshapedOneD, state) = iterate(A.data, state)

@inline @propagate_inbounds function Base.getindex(A::ReshapedOneD, i::Int)
    @boundscheck checkbounds(A.data, i)
    @inbounds ret = A.data[i]
    ret
end
@inline @propagate_inbounds function Base.getindex(A::ReshapedOneD{T,N,Npre}, I::Vararg{Int,N}) where {T,N,Npre}
    @boundscheck checkbounds_indices(Bool, axes(A), I) || throw_boundserror(A, I)
    @inbounds ret = A.data[I[Npre+1]]
    ret
end
@inline @propagate_inbounds function Base.getindex(A::ReshapedOneD, I::Union{CartesianIndex,Int}...)
    A[Base.IteratorsMD.flatten(I)...]
end

Base.convert(::Type{AA}, A::ReshapedOneD) where {AA<:AbstractArray} = convert(AA, reshape(A.data, axes(A)))

import Base: ==
==(A::ReshapedOneD, B::ReshapedOneD) = convert(AbstractArray, A) == convert(AbstractArray, B)
==(A::ReshapedOneD, B::AbstractArray) = convert(AbstractArray, A) == B
==(A::AbstractArray, B::ReshapedOneD) = A == convert(AbstractArray, B)

import Base: +, -, *, /
for op in (:+, :-, :*, :/)
    @eval begin
        ($op)(A::ReshapedOneD, B::ReshapedOneD) =
            broadcast($op, convert(AbstractArray, A), convert(AbstractArray, B))
        @inline ($op)(A::ReshapedOneD, Bs::ReshapedOneD...) =
            broadcast($op, convert(AbstractArray, A), map(B->convert(AbstractArray, B), Bs)...)
    end
end
Base.BroadcastStyle(::Type{R}) where {R<:ReshapedOneD{T,N}} where {T,N} = Broadcast.DefaultArrayStyle{N}()

_reshape(A::ReshapedOneD{T,N}, ::Val{N}) where {T,N} = A
_vec(A::ReshapedOneD) = A.data
Base.vec(A::ReshapedOneD) = A.data  # is this OK? (note indices won't nec. start with 1)
nextendeddims(a::ReshapedOneD) = 1

"""
    iterdims(inds, v::ReshapedOneD{T,N,Npre}) -> Rpre, ind, Rpost

Return a pair `Rpre`, `Rpost` of CartesianIndicess that correspond to
pre- and post- dimensions for iterating over an array with axes
`inds`. `Rpre` corresponds to the `Npre` "pre" dimensions, and `Rpost`
to the trailing dimensions (not including the vector object wrapped in
`v`).  Concretely,

    Rpre  = CartesianIndices(inds[1:Npre])
    ind   = inds[Npre+1]
    Rpost = CartesianIndices(inds[Npre+2:end])

although the implementation differs for reason of type-stability.
"""
function iterdims(inds::Indices{N}, v::ReshapedOneD{T,N,Npre}) where {T,N,Npre}
    indspre, ind, indspost = _iterdims((), (), inds, v)
    CartesianIndices(indspre), ind, CartesianIndices(indspost)
end
@inline function _iterdims(indspre, ::Tuple{}, inds, v)
    _iterdims((indspre..., inds[1]), (), tail(inds), v)  # consume inds and push to indspre
end
@inline function _iterdims(indspre::NTuple{Npre}, ::Tuple{}, inds, v::ReshapedOneD{<:Any,N,Npre}) where {N,Npre}
    indspre, inds[1], tail(inds)   # return the "central" and trailing dimensions
end

function indexsplit(I::CartesianIndex{N}, v::ReshapedOneD{<:Any,N}) where N
    ipre, i, ipost = _iterdims((), (), Tuple(I), v)
    CartesianIndex(ipre), i, CartesianIndex(ipost)
end

#### FIR filters

"""
    kern1, kern2 = box(m, n)
    kerns = box((m, n, ...))

Return a tuple of "flat" kernels, where, for example, `kern1[i] == 1/m` and has length `m`.
"""
function box(sz::Dims)
    all(isodd, sz) || throw(ArgumentError("kernel dimensions must be odd, got $sz"))
    return kernelfactors(ntuple(d -> centered([1/sz[d] for i=1:sz[d]]), length(sz)))
end
box(sz::Int...) = box(sz)

## gradients

"""
```julia
    kern1, kern2 = sobel()
```
Return factored  Sobel filters for dimensions 1 and 2 of a two-dimensional
image. Each is a 2-tuple of one-dimensional filters.

# Citation
P.-E. Danielsson and O. Seger, "Generalized and separable sobel operators," in  *Machine Vision for Three-Dimensional Scenes*,  H. Freeman, Ed.  Academic Press, 1990,  pp. 347–379. [doi:10.1016/b978-0-12-266722-0.50016-6](https://doi.org/doi:10.1016/b978-0-12-266722-0.50016-6)

See also: [`Kernel.sobel`](@ref)  and [`ImageFiltering.imgradients`](@ref).
"""
function sobel()
    f1 = centered(SVector( 1.0, 2.0, 1.0)/4)
    f2 = centered(SVector(-1.0, 0.0, 1.0)/2)
    return kernelfactors((f2, f1)), kernelfactors((f1, f2))
end

"""
```julia
    kern = sobel(extended::NTuple{N,Bool}, d)
```
Return a factored Sobel filter for computing the gradient in `N` dimensions
along axis `d`. If `extended[dim]` is false, `kern` will have size 1 along that
dimension.

See also: [`Kernel.sobel`](@ref) and [`ImageFiltering.imgradients`](@ref).
"""
function sobel(extended::NTuple{N,Bool}, d) where N
    gradfactors(extended, d, [-1, 0, 1]/2, [1, 2, 1]/4)
end

"""
```julia
    kern1, kern2 = prewitt()
```
Return factored Prewitt filters for dimensions 1 and 2 of your image.
Each is a 2-tuple of one-dimensional filters.

# Citation
J. M. Prewitt, "Object enhancement and extraction," *Picture processing and Psychopictorics*, vol. 10, no. 1, pp. 15–19, 1970.

See also: [`Kernel.prewitt`](@ref) and [`ImageFiltering.imgradients`](@ref).
"""
function prewitt()
    f1 = centered(SVector( 1.0, 1.0, 1.0)/3)
    f2 = centered(SVector(-1.0, 0.0, 1.0)/2)
    return kernelfactors((f2, f1)), kernelfactors((f1, f2))
end

"""
```julia
    kern = prewitt(extended::NTuple{N,Bool}, d)
```
Return a factored Prewitt filter for computing the gradient in `N` dimensions
along axis `d`. If `extended[dim]` is false, `kern` will have size 1 along that
dimension.

See also: [`Kernel.prewitt`](@ref) and [`ImageFiltering.imgradients`](@ref).
"""
function prewitt(extended::NTuple{N,Bool}, d) where N
    gradfactors(extended, d, [-1, 0, 1]/2, [1, 1, 1]/3)
end


"""
```julia
    kern1, kern2 = scharr()
```
Return factored Scharr filters for dimensions 1 and 2 of your image.  Each is a
2-tuple of one-dimensional filters.

# Citation
H. Scharr and  J. Weickert, "An anisotropic diffusion algorithm with optimized rotation invariance," *Mustererkennung 2000*, pp. 460–467, 2000. [doi:10.1007/978-3-642-59802-9_58](https://doi.org/doi:10.1007/978-3-642-59802-9_58)


See also: [`Kernel.scharr`](@ref) and [`ImageFiltering.imgradients`](@ref).
"""
function scharr()
    f1 = centered(SVector( 3.0/32.0, 5.0/16.0, 3.0/32.0)*2)
    f2 = centered(SVector(-1.0, 0.0, 1.0)/2)
    return kernelfactors((f2, f1)), kernelfactors((f1, f2))
end

"""
```julia
    kern = scharr(extended::NTuple{N,Bool}, d)
```
Return a factored Scharr filter for computing the gradient in `N` dimensions
along axis `d`. If `extended[dim]` is false, `kern` will have size 1 along that
dimension.

See also: [`Kernel.scharr`](@ref) and [`ImageFiltering.imgradients`](@ref).
"""
function scharr(extended::NTuple{N,Bool}, d) where N
    # The first factor is the central difference, and since we assume a pixel
    # spacing of one, we divide by 2.
    gradfactors(extended, d, [-1.0, 0.0, 1.0]/2,[3.0/32.0, 5.0/16.0, 3.0/32.0]*2,)
end

"""
```julia
    kern1, kern2 = bickley()
```
Return factored Bickley filters for dimensions 1 and 2 of your image.  Each is
a 2-tuple of one-dimensional filters.

# Citation
W. G. Bickley, "Finite difference formulae for the square lattice," *The Quarterly Journal of Mechanics and Applied Mathematics*, vol. 1, no. 1, pp. 35–42, 1948.  [doi:10.1093/qjmam/1.1.35](https://doi.org/doi:10.1137/12087092x)

See also: [`Kernel.bickley`](@ref) and [`ImageFiltering.imgradients`](@ref).
"""
function bickley()
    f1 = centered(SVector( 1.0/12.0, 1.0/3.0, 1.0/12.0)*2)
    f2 = centered(SVector(-1.0, 0.0, 1.0)/2)
    return kernelfactors((f2, f1)), kernelfactors((f1, f2))
end

"""
```julia
    kern = bickley(extended::NTuple{N,Bool}, d)
```
Return a factored Bickley filter for computing the gradient in `N` dimensions
along axis `d`. If `extended[dim]` is false, `kern` will have size 1 along that
dimension.

See also: [`Kernel.bickley`](@ref) and [`ImageFiltering.imgradients`](@ref).
"""
function bickley(extended::NTuple{N,Bool}, d) where N
    # The first factor is the central difference, and since we assume a pixel
    # spacing of one, we divide by 2.
    gradfactors(extended, d, [-1.0, 0.0, 1.0]/2, [1.0/12.0, 1.0/3.0, 1.0/12.0]*2)
end

# Consistent Gradient Operators
# Ando Shigeru
# IEEE Trans. Pat. Anal. Mach. Int., vol. 22 no 3, March 2000
#
# TODO: These coefficients were taken from the paper. It would be nice
#       to resolve the optimization problem and use higher precision
#       versions, which might allow better separable approximations of
#       ando4 and ando5.

"""
```julia
    kern1, kern2 = ando3()
```
Return a factored form of Ando's "optimal" ``3 \\times 3`` gradient filters for dimensions 1 and 2 of your image.

# Citation
S. Ando, "Consistent gradient operators," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 22, no.3, pp. 252–265, 2000. [doi:10.1109/34.841757](https://doi.org/doi:10.1109/34.841757)

See also: [`Kernel.ando3`](@ref),[`KernelFactors.ando4`](@ref),
[`KernelFactors.ando5`](@ref) and [`ImageFiltering.imgradients`](@ref).
"""
function ando3()
    f1 = centered(2*SVector(0.112737, 0.274526, 0.112737))
    f2 = centered(SVector(-1.0, 0.0, 1.0)/2)
    return kernelfactors((f2, f1)), kernelfactors((f1, f2))
end

"""
```julia
    kern = ando3(extended::NTuple{N,Bool}, d)
```
Return a factored Ando filter (size 3) for computing the gradient in
`N` dimensions along axis `d`.  If `extended[dim]` is false, `kern`
will have size 1 along that dimension.

See also: [`KernelFactors.ando4`](@ref), [`KernelFactors.ando5`](@ref) and
[`ImageFiltering.imgradients`](@ref).
"""
function ando3(extended::NTuple{N,Bool}, d) where N
    gradfactors(extended, d, [-1.0, 0.0, 1.0]/2, 2*[0.112737, 0.274526, 0.112737])
end

"""
```julia
    kern1, kern2 = ando4()
```
Return separable approximations of Ando's "optimal" 4x4 filters for dimensions 1
and 2 of your image.

# Citation
S. Ando, "Consistent gradient operators," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 22, no.3, pp. 252–265, 2000. [doi:10.1109/34.841757](https://doi.org/doi:10.1109/34.841757)

See also: [`Kernel.ando4`](@ref) and [`ImageFiltering.imgradients`](@ref).
"""
function ando4()
    f1 = centered(SVector( 0.0919833,  0.408017, 0.408017, 0.0919833))
    f2 = centered(1.46205884*SVector(-0.0919833, -0.408017, 0.408017, 0.0919833))
    return kernelfactors((f2, f1)), kernelfactors((f1, f2))
end

"""
```julia
    kern = ando4(extended::NTuple{N,Bool}, d)
```
Return a factored Ando filter (size 4) for computing the gradient in
`N` dimensions along axis `d`.  If `extended[dim]` is false, `kern`
will have size 1 along that dimension.

# Citation
S. Ando, "Consistent gradient operators," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 22, no.3, pp. 252–265, 2000. [doi:10.1109/34.841757](https://doi.org/doi:10.1109/34.841757)

See also: [`Kernel.ando4`](@ref) and [`ImageFiltering.imgradients`](@ref).
"""
function ando4(extended::NTuple{N,Bool}, d) where N
    if N == 2 && all(extended)
        return ando4()[d]
    else
        error("dimensions other than 2 are not yet supported")
    end
end

"""
```julia
    kern1, kern2 = ando5()
```
Return a separable approximations of Ando's "optimal" 5x5 gradient filters for
dimensions 1 and 2 of your image.

# Citation
S. Ando, "Consistent gradient operators," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 22, no.3, pp. 252–265, 2000. [doi:10.1109/34.841757](https://doi.org/doi:10.1109/34.841757)

See also: [`Kernel.ando5`](@ref) and [`ImageFiltering.imgradients`](@ref).
"""
function ando5()
    f1 = centered(SVector( 0.0357338, 0.248861, 0.43081, 0.248861, 0.0357338))
    f2 = centered(0.784406*SVector(-0.137424, -0.362576, 0.0, 0.362576, 0.137424))
    return kernelfactors((f2, f1)), kernelfactors((f1, f2))
end

"""
```julia
    kern = ando5(extended::NTuple{N,Bool}, d)
```
Return a factored Ando filter (size 5) for computing the gradient in
`N` dimensions along axis `d`.  If `extended[dim]` is false, `kern`
will have size 1 along that dimension.
"""
function ando5(extended::NTuple{N,Bool}, d) where N
    if N == 2 && all(extended)
        return ando5()[d]
    else
        error("dimensions other than 2 are not yet supported")
    end
end

## Gaussian

"""
    gaussian(σ::Real, [l]) -> g

Construct a 1d gaussian kernel `g` with standard deviation `σ`, optionally
providing the kernel length `l`. The default is to extend by two `σ`
in each direction from the center. `l` must be odd.
"""
function gaussian(σ::Real, l::Int = 4*ceil(Int,σ)+1)
    isodd(l) || throw(ArgumentError("length must be odd"))
    w = l>>1
    g = σ == 0 ? [exp(0/(2*oftype(σ, 1)^2))] : [exp(-x^2/(2*σ^2)) for x=-w:w]
    centered(g/sum(g))
end
gaussian(σ::Real, l::Integer) = gaussian(σ, Int(l)::Int)

"""
    gaussian((σ1, σ2, ...), [l]) -> (g1, g2, ...)

Construct a multidimensional gaussian filter as a product of single-dimension
factors, with standard deviation `σd` along dimension `d`. Optionally
provide the kernel length `l`, which must be a tuple of the same
length.
"""
gaussian(σs::NTuple{N,Real}, ls::NTuple{N,Integer}) where {N} =
    kernelfactors( map(gaussian, σs, ls) )
gaussian(σs::NTuple{N,Real}) where {N} = kernelfactors(map(gaussian, σs))

gaussian(σs::AbstractVector, ls::AbstractVector) = gaussian((σs...,), (ls...,))
gaussian(σs::AbstractVector) = gaussian((σs...,))

#### IIR

struct TriggsSdika{T,k,l,L} <: IIRFilter{T}
    a::SVector{k,T}
    b::SVector{l,T}
    scale::T
    M::SMatrix{l,k,T,L}
    asum::T
    bsum::T

    TriggsSdika{T,k,l,L}(a, b, scale, M) where {T,k,l,L} =
        new{T,k,l,L}(a, b, scale, M, sum(a), sum(b))
end
"""
    TriggsSdika(a, b, scale, M)

Defines a kernel for one-dimensional infinite impulse response (IIR)
filtering. `a` is a "forward" filter, `b` a "backward" filter, `M` is
a matrix for matching boundary conditions at the right edge, and
`scale` is a constant scaling applied to each element at the
conclusion of filtering.

# Citation

B. Triggs and M. Sdika, "Boundary conditions for Young-van Vliet
recursive filtering". IEEE Trans. on Sig. Proc. 54: 2365-2367
(2006).
"""
TriggsSdika(a::SVector{k,T}, b::SVector{l,T}, scale, M::SMatrix{l,k,T,L}) where {T,k,l,L} = TriggsSdika{T,k,l,L}(a, b, scale, M)

"""
    TriggsSdika(ab, scale)

Create a symmetric Triggs-Sdika filter (with `a = b = ab`). `M` is
calculated for you. Only length 3 filters are currently supported.
"""
function TriggsSdika(a::SVector{3,T}, scale) where T
    a1, a2, a3 = a[1], a[2], a[3]
    Mdenom = (1+a1-a2+a3)*(1-a1-a2-a3)*(1+a2+(a1-a3)*a3)
    M = @SMatrix([-a3*a1+1-a3^2-a2     (a3+a1)*(a2+a3*a1)  a3*(a1+a3*a2);
                  a1+a3*a2            -(a2-1)*(a2+a3*a1)  -(a3*a1+a3^2+a2-1)*a3;
                  a3*a1+a2+a1^2-a2^2   a1*a2+a3*a2^2-a1*a3^2-a3^3-a3*a2+a3  a3*(a1+a3*a2)]);
    TriggsSdika(a, a, scale, M/Mdenom)
end
Base.vec(kernel::TriggsSdika) = kernel
Base.ndims(kernel::TriggsSdika) = 1
Base.ndims(::Type{T}) where {T<:TriggsSdika} = 1
Base.axes1(kernel::TriggsSdika) = 0:0
Base.axes(kernel::TriggsSdika) = (Base.axes1(kernel),)
Base.isempty(kernel::TriggsSdika) = false

iterdims(inds::Indices{1}, kern::TriggsSdika) = (), inds[1], ()
_reshape(kern::TriggsSdika, ::Val{1}) = kern

# Note that there's a sign reversal between Young & Triggs.
"""
    IIRGaussian([T], σ; emit_warning::Bool=true)

Construct an infinite impulse response (IIR) approximation to a
Gaussian of standard deviation `σ`. `σ` may either be a single real
number or a tuple of numbers; in the latter case, a tuple of such filters
will be created, each for filtering a different dimension of an array.

Optionally specify the type `T` for the filter coefficients; if not
supplied, it will match `σ` (unless `σ` is not floating-point, in
which case `Float64` will be chosen).

# Citation

I. T. Young, L. J. van Vliet, and M. van Ginkel, "Recursive Gabor
Filtering". IEEE Trans. Sig. Proc., 50: 2798-2805 (2002).
"""
function IIRGaussian(::Type{T}, σ::Real; emit_warning::Bool = true) where T
    if emit_warning && σ < 1 && σ != 0
        @warn("σ is too small for accuracy")
    end
    m0 = convert(T,1.16680)
    m1 = convert(T,1.10783)
    m2 = convert(T,1.40586)
    q = convert(T,1.31564*(sqrt(1+0.490811*σ*σ) - 1))
    ascale = (m0+q)*(m1*m1 + m2*m2  + 2m1*q + q*q)
    B = (m0*(m1*m1 + m2*m2)/ascale)^2
    # This is what Young et al call -b, but in filt() notation would be called a
    a1 = q*(2*m0*m1 + m1*m1 + m2*m2 + (2*m0+4*m1)*q + 3*q*q)/ascale
    a2 = -q*q*(m0 + 2m1 + 3q)/ascale
    a3 = q*q*q/ascale
    a = SVector(a1,a2,a3)
    TriggsSdika(a, B)
end
IIRGaussian(σ::Real; emit_warning::Bool = true) = IIRGaussian(iirgt(σ), σ; emit_warning=emit_warning)

function IIRGaussian(::Type{T}, σs::NTuple{N,Real}; emit_warning::Bool = true) where {T,N}
    iirg(T, (), σs, tail(ntuple(d->true, Val(N))), emit_warning)
end
IIRGaussian(σs::Tuple; emit_warning::Bool = true) = IIRGaussian(iirgt(σs), σs; emit_warning=emit_warning)

IIRGaussian(σs::AbstractVector; kwargs...) = IIRGaussian((σs...,); kwargs...)
IIRGaussian(::Type{T}, σs::AbstractVector; kwargs...) where {T} = IIRGaussian(T, (σs...,); kwargs...)

iirgt(σ::AbstractFloat) = typeof(σ)
iirgt(σ::Real) = Float64
iirgt(σs::Tuple) = promote_type(map(iirgt, σs)...)

@inline function iirg(::Type{T}, pre, σs, post, emit_warning) where T
    kern = ReshapedOneD(pre, IIRGaussian(T, σs[1]; emit_warning=emit_warning), post)
    (kern, iirg(T, (pre..., post[1]), tail(σs), tail(post), emit_warning)...)
end
iirg(::Type{T}, pre, σs::Tuple{Real}, ::Tuple{}, emit_warning) where {T} =
    (ReshapedOneD(pre, IIRGaussian(T, σs[1]; emit_warning=emit_warning), ()),)

#### Sliding DCT

"""
    SlidingDCT{T,K,DC} <: AbstractVector{T}

A symmetric kernel with axes `-N:N` that is a short sum of cosines,

    h[r] = c₀ + Σₖ cₖ cos(ωₖ r)    for -N ≤ r ≤ N,

where the constant term `c₀` is only present if `DC` is `true`, and `K` is the
number of (non-constant) cosine terms. Filtering with such a kernel can be
performed with a sliding (recursive) discrete cosine transform whose cost per
pixel is `O(K)` regardless of the kernel width `2N+1`.

A `SlidingDCT` is a genuine finite-support kernel: it can be indexed like any
other centered kernel, and it can be used anywhere an `AbstractVector` kernel
is accepted. `imfilter` automatically uses the constant-time algorithm when it
encounters such a kernel.

!!! note
    As with recursive (IIR) filters, a non-finite value (`NaN` or `Inf`) in the input
    contaminates the rest of the line being filtered, rather than only its
    neighborhood as with direct (FIR) filtering. Use the `NA()` border to filter
    images with missing values.

Construct one with [`KernelFactors.DCTGaussian`](@ref) or by fitting an arbitrary
even kernel with `SlidingDCT(h; K, kind, moments)`.
"""
struct SlidingDCT{T,K,DC} <: AbstractVector{T}
    N::Int
    c::SVector{K,T}   # coefficients of the cosine terms
    ω::SVector{K,T}   # angular frequencies of the cosine terms (all nonzero)
    c0::T             # coefficient of the constant (DC) term, zero unless DC

    function SlidingDCT{T,K,DC}(N::Integer, c, ω, c0) where {T,K,DC}
        N >= 0 || throw(ArgumentError("the kernel radius `N` must be non-negative, got $N"))
        DC || iszero(c0) || throw(ArgumentError("a nonzero DC coefficient requires `DC = true`"))
        all(!iszero, ω) || throw(ArgumentError("the DC term must be specified via `c0`, not as a zero frequency"))
        new{T,K,DC}(N, c, ω, c0)
    end
end
SlidingDCT(N::Integer, c::SVector{K,T}, ω::SVector{K,T}, c0::T, DC::Bool) where {T,K} = SlidingDCT{T,K,DC}(N, c, ω, c0)
Base.convert(::Type{SlidingDCT{T}}, k::SlidingDCT{T}) where {T} = k
Base.convert(::Type{SlidingDCT{T}}, k::SlidingDCT{S,K,DC}) where {T,S,K,DC} =
    SlidingDCT{T,K,DC}(k.N, SVector{K,T}(k.c), SVector{K,T}(k.ω), T(k.c0))

_reshape(kernel::SlidingDCT, ::Val{1}) = kernel
_reshape(kernel::SlidingDCT, ::Val{N}) where {N} = ReshapedOneD{N,0}(kernel)

Base.axes(kernel::SlidingDCT) = (IdentityUnitRange(-kernel.N:kernel.N),)
Base.size(kernel::SlidingDCT) = (2 * kernel.N + 1,)
@inline function Base.getindex(kernel::SlidingDCT{T}, i::Int) where {T}
    @boundscheck checkbounds(kernel, i)
    v = kernel.c0
    for k in eachindex(kernel.c)
        v += kernel.c[k] * cos(kernel.ω[k] * i)
    end
    return v
end

"""
    SlidingDCT(h; K = 3, kind = 7, moments = K >= 2 ? (0, 2) : (0,))

Approximate the even, centered kernel `h` (with axes `-N:N`) by a sum of `K`
discrete cosine transform (DCT) basis functions, so that filtering with it can
be performed in constant time per pixel, independent of `N`.

The coefficients are the least-squares fit of `h` subject to exactly preserving
the moments `Σᵣ rᵐ h[r]` for each order `m` listed in `moments`. The default
preserves the sum (0th moment) and, if `K ≥ 2`, the second moment (the variance
of a normalized kernel); pass `moments = ()` for an unconstrained least-squares fit.
There cannot be more moments than terms.

`kind` selects the DCT basis, i.e., the frequencies `ωₖ`, among the DCT types
without spatial phase shift:

| `kind` | `ωₖ`, `k = 0, …, K-1` |
|:------:|:-----------------------|
| `1`    | `πk/N`                 |
| `3`    | `π(2k+1)/(2N+2)`       |
| `5`    | `2πk/(2N+1)`           |
| `7`    | `π(2k+1)/(2N+1)`       |

DCT-3 and DCT-7 (the default) do not contain a constant term and are usually
the most accurate for a given `K`.

The element type of the result matches `float(eltype(h))`.

# Citation

K. Sugimoto and S. Kamata, "Efficient constant-time Gaussian filtering with
sliding DCT/DST-5 and dual-domain error minimization", ITE Transactions on
Media Technology and Applications 3: 12-21 (2015).

K. Sugimoto, S. Kyochi and S. Kamata, "Universal approach for DCT-based
constant-time Gaussian filter with moment preservation", IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP), 1498-1502 (2018).

See also: [`KernelFactors.DCTGaussian`](@ref).
"""
function SlidingDCT(h::AbstractVector; K::Integer = 3, kind::Integer = 7, moments = K >= 2 ? (0, 2) : (0,))
    ax = axes(h, 1)
    N = last(ax)
    first(ax) == -N || throw(ArgumentError("the kernel must have axes `-N:N`, got $ax (use `centered`)"))
    T = float(eltype(h))
    all(r -> h[r] ≈ h[-r], 1:N) || throw(ArgumentError("the kernel must be even, i.e., h[r] == h[-r]"))
    μ = map(m -> (m, sum(r -> T(r)^m * h[r], ax)), Tuple(moments))
    return dctfit(T, r -> h[r], N, K, kind, μ)
end

# Fit `target(r)` for `r ∈ -N:N` by `K` cosine terms of the DCT type `kind`,
# subject to Σᵣ rᵐ ĥ(r) = μₘ for each `(m, μₘ)` in `μ`.
function dctfit(::Type{T}, target, N::Integer, K::Integer, kind::Integer, μ) where {T}
    kind ∈ (1, 3, 5, 7) || throw(ArgumentError("`kind` must be one of 1, 3, 5 or 7, got $kind"))
    K >= 1 || throw(ArgumentError("`K` must be at least 1, got $K"))
    M = length(μ)
    # h[0:N] determines an even kernel, so at most N+1 cosines are linearly independent
    K <= N + 1 || throw(ArgumentError("`K` must not exceed `N + 1` = $(N + 1); increase the kernel radius"))
    M <= K || throw(ArgumentError("cannot preserve $M moments with only `K` = $K terms"))
    Tf = promote_type(T, Float64)
    ω = dctfrequencies(Tf, kind, N, K)
    rs = -N:N
    C = [cos(ωk * r) for r in rs, ωk in ω]
    h = Tf[target(r) for r in rs]
    if M == 0
        c = C \ h
    else
        # Lagrange multipliers; each moment row is scaled by N^m for conditioning
        s = [Tf(max(N, 1))^m for (m, _) in μ]
        U = [Tf(r)^m / s[i] for (i, (m, _)) in enumerate(μ), r in rs] * C
        rhs = [2 * (C' * h); [Tf(μm) / s[i] for (i, (_, μm)) in enumerate(μ)]]
        c = ([2 * (C' * C) U'; U zeros(Tf, M, M)] \ rhs)[1:K]
    end
    DC = iszero(first(ω))
    c0 = DC ? T(c[1]) : zero(T)
    idx = DC ? (2:K) : (1:K)
    Kosc = length(idx)
    return SlidingDCT(N, SVector{Kosc,T}(c[idx]), SVector{Kosc,T}(ω[idx]), c0, DC)
end

function dctfrequencies(::Type{T}, kind::Integer, N::Integer, K::Integer) where {T}
    kind == 1 && return [T(π) * k / N for k in 0:K-1]
    kind == 3 && return [T(π) * (2k + 1) / (2N + 2) for k in 0:K-1]
    kind == 5 && return [2 * T(π) * k / (2N + 1) for k in 0:K-1]
    return [T(π) * (2k + 1) / (2N + 1) for k in 0:K-1]
end

# Initial guess for the radius of a DCT approximation of a Gaussian with K cosine
# terms, from the dual-domain error minimization of Sugimoto & Kamata (2015): the
# spatial truncation error Es(R) = erfc(ϕ) decreases with R while the frequency
# truncation error Ef(K, R) ≈ erfc(ψ) increases, with ϕ = (2R+1)/(2σ) and
# ψ = πσ(2K+1)/(2R+1); their sum is unimodal and we find the zero of its derivative
# by bisection.
function dctgaussian_radius(σ::Real, K::Integer)
    function dE(R)
        ϕ = (2R + 1) / (2σ)
        ψ = π * σ * (2K + 1) / (2R + 1)
        return (4ψ^2 - 1) * sqrt(π) * σ * (2K + 1) / (R^2 + R + 1/4) * exp(-ψ^2) - 4ϕ^2 / (sqrt(π) * σ) * exp(-ϕ^2)
    end
    lo, hi = max(floor(Int, σ), 1), ceil(Int, (K + 3) * σ)
    while hi > lo + 1
        mid = (lo + hi) >> 1
        dE(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    return abs(dE(lo)) <= abs(dE(hi)) ? lo : hi
end

# Starting from N, descend the exact L2 error `err(N)` of the fitted kernel
function refine_radius(err, N::Integer, Nmin::Integer)
    N = max(N, Nmin)
    e = err(N)
    for step in (1, -1)
        while N + step >= Nmin
            e2 = err(N + step)
            e2 < e || break
            N, e = N + step, e2
        end
    end
    return N
end

"""
    DCTGaussian([T], σ; K = 3, N, kind = 7, emit_warning = true)

Construct a [`KernelFactors.SlidingDCT`](@ref) approximation of a Gaussian of
standard deviation `σ`. Filtering with it costs `O(K)` operations per pixel,
independently of `σ`, which makes it the fastest choice for large `σ`.

The kernel is the least-squares approximation, on `-N:N`, of the sampled Gaussian
by `K` DCT basis functions, constrained to preserve the sum (so that constant
images are unchanged) and, if `K ≥ 2`, the variance of the Gaussian. Increasing `K` improves
the accuracy at the cost of speed: the maximal error relative to the peak of the
kernel is about `5e-2`, `4e-3`, `1e-3`, `2e-4` and `4e-5` for `K = 2, …, 6`.
Unless specified, the radius `N` minimizes the L2 (energy) error of the kernel with
respect to the untruncated Gaussian, which balances the truncation of the kernel in
space and in frequency ("dual-domain error minimization", Sugimoto & Kamata 2015).
See [`KernelFactors.SlidingDCT`](@ref) for the meaning of `kind`.

`σ` may either be a single real number or a tuple of numbers; in the latter case,
a tuple of such filters will be created, each for filtering a different dimension
of an array.

Optionally specify the type `T` for the filter coefficients; if not
supplied, it will match `σ` (unless `σ` is not floating-point, in
which case `Float64` will be chosen).

# Examples

```julia
imgf = imfilter(img, KernelFactors.DCTGaussian((10, 10)))
```

# Citation

K. Sugimoto and S. Kamata, "Efficient constant-time Gaussian filtering with
sliding DCT/DST-5 and dual-domain error minimization", ITE Transactions on
Media Technology and Applications 3: 12-21 (2015).

K. Sugimoto, S. Kyochi and S. Kamata, "Universal approach for DCT-based
constant-time Gaussian filter with moment preservation", IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP), 1498-1502 (2018).

See also: [`KernelFactors.IIRGaussian`](@ref), [`KernelFactors.gaussian`](@ref).
"""
function DCTGaussian(::Type{T}, σ::Real; K::Integer = 3, N::Union{Integer,Nothing} = nothing,
                     kind::Integer = 7, emit_warning::Bool = true) where {T}
    σ >= 0 || throw(ArgumentError("σ must be non-negative, got $σ"))
    if iszero(σ)
        return SlidingDCT(0, SVector{0,T}(), SVector{0,T}(), one(T), true)  # the identity kernel
    end
    emit_warning && σ < 1 && @warn("σ is too small for accuracy")
    # The (untruncated) sampled Gaussian and its moments
    Tf = promote_type(T, Float64)
    M = max(something(N, 0), ceil(Int, (K + 3) * σ)) + ceil(Int, 10σ)
    g = OffsetArray([exp(-Tf(r)^2 / (2 * Tf(σ)^2)) for r in -M:M], -M:M)
    g ./= sum(g)
    μ2 = sum(r -> Tf(r)^2 * g[r], -M:M)
    μ = K == 1 ? ((0, one(Tf)),) : ((0, one(Tf)), (2, μ2))
    fit(N) = dctfit(Tf, r -> g[r], N, K, kind, μ)
    if N === nothing
        # Minimize the L2 error, including the truncation outside -N:N
        Nmin = max(K - 1, 1)  # K ≤ N + 1 is required by the fit
        gnorm2 = sum(abs2, g)
        err(n) = (k = fit(n); (sum(r -> abs2(k[r] - g[r]), -n:n) + gnorm2 - sum(r -> abs2(g[r]), -n:n)))
        N = refine_radius(err, dctgaussian_radius(σ, kind ∈ (1, 5) ? K - 1 : K), Nmin)
    end
    return convert(SlidingDCT{T}, fit(N))
end
DCTGaussian(σ::Real; kwargs...) = DCTGaussian(iirgt(σ), σ; kwargs...)

function DCTGaussian(::Type{T}, σs::NTuple{D,Real}; N = map(σ -> nothing, σs), kwargs...) where {T,D}
    Ns = N isa Tuple ? N : ntuple(_ -> N, Val(D))
    kernelfactors(ntuple(d -> _dctgaussian(T, σs[d], Ns[d]; kwargs...), Val(D)))
end
DCTGaussian(σs::Tuple; kwargs...) = DCTGaussian(iirgt(σs), σs; kwargs...)

DCTGaussian(σs::AbstractVector; kwargs...) = DCTGaussian((σs...,); kwargs...)
DCTGaussian(::Type{T}, σs::AbstractVector; kwargs...) where {T} = DCTGaussian(T, (σs...,); kwargs...)

_dctgaussian(::Type{T}, σ, ::Nothing; kwargs...) where {T} = DCTGaussian(T, σ; kwargs...)
_dctgaussian(::Type{T}, σ, N::Integer; kwargs...) where {T} = DCTGaussian(T, σ; N = N, kwargs...)

###### Utilities

"""
    kernelfactors(factors::Tuple)

Prepare a factored kernel for filtering. If passed a 2-tuple of
vectors of lengths `m` and `n`, this will return a 2-tuple of
`ReshapedVector`s that are effectively of sizes `m×1` and `1×n`. In
general, each successive `factor` will be reshaped to extend along the
corresponding dimension.

If passed a tuple of general arrays, it is assumed that each is shaped
appropriately along its "leading" dimensions; the dimensionality of each is
"extended" to `N = length(factors)`, appending 1s to the size as needed.
"""
function kernelfactors(factors::NTuple{N,AbstractVector}) where {N}
    total = ntuple(d->true, Val(N))
    _kernelfactors(total, factors)
end

@inline _kernelfactors(total::NTuple{N,Bool}, factors::NTuple{M,Any}) where {N,M} =
    (ReshapedOneD{N,N-M}(factors[1]), _kernelfactors(total, Base.tail(factors))...)
_kernelfactors(::NTuple{N,Bool}, ::Tuple{}) where N = ()

# A variant in which we just need to fill out to N dimensions
kernelfactors(factors::NTuple{N,AbstractArray}) where {N} = map(f->_reshape(f, Val(N)), factors)

function gradfactors(extended::NTuple{N,Bool}, d::Int, k1, k2) where N
    kernelfactors(ntuple(i -> kdim(extended[i], i==d ? k1 : k2), Val(N)))
end

kdim(keep::Bool, k) = keep ? centered(k) : OffsetArray([oneunit(eltype(k))], 0:0)

if Base.VERSION >= v"1.4.2" && ccall(:jl_generating_output, Cint, ()) == 1
    precompile(sobel, ())
    for T in (Int, Float64, Float32)
        precompile(gaussian, (Tuple{T,T},))
        precompile(gaussian, (T,))
        precompile(gaussian, (T, Int))
        precompile(IIRGaussian, (Tuple{T,T},))
        precompile(DCTGaussian, (Tuple{T,T},))
    end
end

end
