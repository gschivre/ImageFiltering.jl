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
  - `DCTGaussian` (approximate gaussian filtering, fast even for large σ)

See also: [`Kernel`](@ref).
"""
module KernelFactors

using LinearAlgebra, StaticArrays, OffsetArrays
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

abstract type DCTFilter{T} end

Base.eltype(kernel::DCTFilter{T}) where {T} = T

abstract type DSTFilter{T} end

Base.eltype(kernel::DSTFilter{T}) where {T} = T

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

###### DCT/DST

struct DCT_1{T} <: DCTFilter{T}
    basis::Matrix{T}
    coefficients::Vector{T}
end
struct DCT_3{T} <: DCTFilter{T}
    basis::Matrix{T}
    coefficients::Vector{T}
end
struct DCT_5{T} <: DCTFilter{T}
    basis::Matrix{T}
    coefficients::Vector{T}
end
struct DCT_7{T} <: DCTFilter{T}
    basis::Matrix{T}
    coefficients::Vector{T}
end

# only the DCT types with no spatial phase shift
DCTtypes = Union{DCT_1, DCT_3, DCT_5, DCT_7}
Base.vec(kernel::DCTtypes) = kernel
Base.ndims(kernel::DCTtypes) = 1
Base.ndims(::Type{T}) where {T <: DCTtypes} = 1
Base.axes1(kernel::DCTtypes) = 0:0
Base.axes(kernel::DCTtypes) = (Base.axes1(kernel), )
Base.isempty(kernel::DCTtypes) = false

iterdims(inds::Indices{1}, kern::DCTtypes) = (), inds[1], ()
_reshape(kern::DCTtypes, ::Val{1}) = kern

approxorder(d::DCTtypes) = size(d.basis, 2)
@inline function Base.show(io::IO, mime::MIME"text/plain", d::DCTtypes)
    halfh = d.basis * d.coefficients
    h = centered(vcat(view(halfh, lastindex(halfh):-1:2), halfh))
    show(io, mime, h)
end

"""
    DCTGaussian([T], σ; R::Int = (6 * ceil(Int, σ) + 1) >> 1, K::Int = 3)

Defines a kernel for one-dimensional discrete cosine transform 7 (DCT-7) 
approximation to a Gaussian of standard deviation `σ` which preserves
the first 2 moments. `R` is the filter window radius and `K` is the
approximation order. `σ` may either be a single real number or a tuple
of numbers; in the latter case, a tuple of such filters will be created,
each for filtering a different dimension of an array.

Optionally specify the type `T` for the filter coefficients; if not
supplied, it will match `σ` (unless `σ` is not floating-point, in
which case `Float64` will be chosen).

# Citation

K. Sugimoto, S. Kyochi and S. Kamata, "Universal approach for dct-based constant-time 
gaussian filter with moment preservation". IEEE international conference on acoustics,
speech and signal processing (ICASSP) 1498-1502 (2018).
"""
function DCTGaussian(::Type{T}, σ::Real; R::Int = (6 * ceil(Int, σ) + 1) >> 1, K::Int = 3) where T
    h = OffsetArray([exp(-x ^ 2 / T(2 * σ ^ 2)) for x in 0:(R - 1)], 0:(R - 1))
    h ./= h[0] + 2 * sum(view(h, 1:(R - 1)))
    μ = ((0, one(T)), (2, T(σ ^ 2)))
    return DCTcoeff(DCT_7, h, μ; K = K)
end
DCTGaussian(σ::Real; R::Int = (6 * ceil(Int, σ) + 1) >> 1, K::Int = 3) = DCTGaussian(dctgt(σ), σ; R = R, K = K)

function DCTGaussian(::Type{T}, σs::NTuple{N, Real}; R::NTuple{N, Int} = ntuple(d -> (6 * ceil(Int, σs[d]) + 1) >> 1, Val(N)), K::Int = 3) where {T, N}
    dctg(T, (), σs, tail(ntuple(d -> true, Val(N))), R, K)
end
DCTGaussian(σs::Tuple; R::Tuple = map(s -> (6 * ceil(Int, s) + 1) >> 1, σs), K::Int = 3) = DCTGaussian(dctgt(σs), σs; R = R, K = K)

DCTGaussian(σs::AbstractVector; kwargs...) = DCTGaussian((σs...,); kwargs...)
DCTGaussian(::Type{T}, σs::AbstractVector; kwargs...) where {T} = DCTGaussian(T, (σs...,); kwargs...)

dctgt(σ::AbstractFloat) = typeof(σ)
dctgt(σ::Real) = Float64
dctgt(σs::Tuple) = promote_type(map(dctgt, σs)...)

@inline function dctg(::Type{T}, pre, σs, post, R, K) where T
    kern = ReshapedOneD(pre, DCTGaussian(T, σs[1]; R = R[1], K = K), post)
    (kern, dctg(T, (pre..., post[1]), tail(σs), tail(post), tail(R), K)...)
end
dctg(::Type{T}, pre, σs::Tuple{Real}, ::Tuple{}, R, K) where {T} =
    (ReshapedOneD(pre, DCTGaussian(T, σs[1]; R = R[1], K = K), ()),)

# From a kernel h and a set of moments μ, compute the DCT coefficients where the DCT type is given by D.
# the type of the kernel and of the moments is used to determine the type of the coefficients.
# Due to the symmetry of the DCT only the positive part of the kernel is used.
function DCTcoeff(::Type{D}, h::OffsetVector{T}, μ::NTuple{M, Tuple{Int, T}}; K::Int = 3) where {D <: DCTFilter, T, M}
    @assert checkindex(Bool, eachindex(h), 0) "The kernel ´h´ must be centered"
    firstindex(h) != 0 && DCTcoeff(D, OffsetArray(view(h, 0:lastindex(h)), 0:lastindex(h)), μ; K = K)
    R = length(h)
    W = Diagonal([T(0.5);ones(T, R - 1)]) # weight matrix for DCT symmetry
    C = DCTbasis(T, D, R, K) # DCT basis
    A = DCT_CWCinv(D, C, W) # A =  inv(C' * W * C)
    B = C' * W
    hls = A * B * OffsetArrays.no_offset_view(h) # least square solution without moment constraints
    M == 0 && return D(C, hls) # no moment constraints
    P = zeros(Int, R, M)
    for m = 1:M
        for r = 1:R
            @inbounds P[r, m] = (r - 1) ^ μ[m][1]
        end
    end
    U = B * P
    Uhls = U' * hls # will be U' * hls - 0.5 * μ
    for m in 1:M
        @inbounds Uhls[m] -= 0.5 * μ[m][2]
    end
    Sinv = inv(U' * A * U)
    return D(C, hls - A * U * Sinv * Uhls)
end
function DCTcoeff(::Type{D}, h::AbstractVector{T}, μ::NTuple{M, Tuple{Int, T}}; K::Int = 3) where {D <: DCTFilter, T, M}
    R = 1 + length(h) >> 1
    return DCTcoeff(D, OffsetArray(view(h, R:length(h)), 0:R), μ; K = K)
end

# Given the filter window radius and the approximation order K 
# create the matrix for the DCT/DST basis where the DCT/DST type is given by D.
function DCTbasis(::Type{T}, ::Type{D}, R::Int, K::Int) where {T, D <: DCTFilter}
    w, k0, r0 = DCT_DST_params(T, D, R)
    return [cos(T(2 * π / w) * (k + k0) * (r + r0)) for r in 0:(R - 1), k in 0:(K - 1)]
end
function DSTbasis(::Type{T}, ::Type{D}, R::Int, K::Int) where {T, D <: DSTFilter}
    w, k0, r0 = DCT_DST_params(T, D, R)
    return [sin(T(2 * π / w) * (k + k0) * (r + r0)) for r in 0:(R - 1), k in 0:(K - 1)]
end

# DCT-1 parameters (period length w and phase shifts k0 and r0)
function DCT_DST_params(::Type{T}, ::Type{DCT_1}, R::Int) where T
    w = 2 * R - 2
    k0 = zero(T)
    r0 = zero(T)
    return (w, k0, r0)
end

# Compute (C' * W * C)^-1 for the DCT-1 (C is the DCT basis and W is the weight matrix)
function DCT_CWCinv(::Type{DCT_1}, C::Matrix{T}, W::Diagonal{T}) where T
    Mkinv = T(4 / (2 * size(C, 1) - 2)) * Diagonal([T(0.5);ones(T, size(C, 2) - 1)])
    s = [(-1) ^ k for k in 0:(size(C, 2) - 1)]
    return Mkinv * (I - (s * s' * Mkinv) / T(2 + s' * Mkinv * s))
end

# DCT-3 parameters (period length w and phase shifts k0 and r0)
function DCT_DST_params(::Type{T}, ::Type{DCT_3}, R::Int) where T
    w = 2 * R
    k0 = T(0.5)
    r0 = zero(T)
    return (w, k0, r0)
end

# Compute (C' * W * C)^-1 for the DCT-3 (C is the DCT basis and W is the weight matrix)
function DCT_CWCinv(::Type{DCT_3}, C::Matrix{T}, W::Diagonal{T}) where T
    return Diagonal(T(4 / (2 * size(C, 1))) * ones(T, size(C, 2)))
end

# DCT-5 parameters (period length w and phase shifts k0 and r0)
function DCT_DST_params(::Type{T}, ::Type{DCT_5}, R::Int) where T
    w = 2 * R - 1
    k0 = zero(T)
    r0 = zero(T)
    return (w, k0, r0)
end

# Compute (C' * W * C)^-1 for the DCT-5 (C is the DCT basis and W is the weight matrix)
function DCT_CWCinv(::Type{DCT_5}, C::Matrix{T}, W::Diagonal{T}) where T
    return Diagonal(T(4 / (2 * size(C, 1) - 1)) * [T(0.5);ones(T, size(C, 2) - 1)])
end

# DCT-7 parameters (period length w and phase shifts k0 and r0)
function DCT_DST_params(::Type{T}, ::Type{DCT_7}, R::Int) where T
    w = 2 * R - 1
    k0 = T(0.5)
    r0 = zero(T)
    return (w, k0, r0)
end

# Compute (C' * W * C)^-1 for the DCT-7 (C is the DCT basis and W is the weight matrix)
function DCT_CWCinv(::Type{DCT_7}, C::Matrix{T}, W::Diagonal{T}) where T
    return Diagonal(T(4 / (2 * size(C, 1) - 1)) * ones(T, size(C, 2)))
end

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
