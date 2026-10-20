using ImageFiltering, ImageCore, OffsetArrays, ComputationalResources
using ImageFiltering.KernelFactors: SlidingKernel, SlidingDCT, SlidingDST, SeparableSum, ReshapedOneD,
    DCTGaussian, slidinggaussian, DCTLoG, DCTDoG
using Random, Test

# The same kernel as a plain FIR array, to compare against the sliding algorithm
firkernel(k::SlidingKernel) = OffsetArray(collect(k), axes(k))
firkernel(k::ReshapedOneD{T,N,Npre}) where {T,N,Npre} = ReshapedOneD{N,Npre}(firkernel(k.data))
firkernel(ks::Tuple) = map(firkernel, ks)
firkernel(k) = k

maxabsdiff(a, b) = maximum(x -> sum(abs, channelview([float(x)])), a .- b)
# Round-off tolerance for the (real) element type of an image
rtol(img) = 100 * eps(eltype(eltype(img)))
# Peak signal-to-noise ratio for images with values in [0, 1]
psnr(a, ref) = 10 * log10(length(ref) / sum(abs2, a .- ref))
# The untruncated sampled Gaussian, normalized
function sampledgaussian(σ, M = ceil(Int, 12σ))
    g = OffsetArray([exp(-r^2 / (2σ^2)) for r in -M:M], -M:M)
    return g ./ sum(g)
end
# ... and its derivatives, as correlation kernels (imfilter computes the derivative)
function sampledderivative(σ, order::Integer, M = ceil(Int, 12σ))
    g = sampledgaussian(σ, M)
    order == 0 && return g
    order == 1 && return OffsetArray([r / σ^2 * g[r] for r in -M:M], -M:M)
    return OffsetArray([(r^2 - σ^2) / σ^4 * g[r] for r in -M:M], -M:M)
end
# relative L2 error of a kernel with respect to a reference
l2err(k, ref) = sqrt(sum(r -> abs2((r in axes(k, 1) ? k[r] : 0.0) - ref[r]), axes(ref, 1)) / sum(abs2, ref))

@testset "Sliding DCT/DST" begin
    Random.seed!(1234)
    @testset "kernel" begin
        for σ in (1.0, 2.5, 7.3, 30.0), K in 1:6, kind in (1, 3, 5, 7)
            k = DCTGaussian(σ; K = K, kind = kind)
            N = k.N
            @test axes(k) == (-N:N,)
            @test size(k) == (2N + 1,) && length(k) == 2N + 1
            @test eltype(k) == Float64
            @test all(r -> k[r] ≈ k[-r], 1:N)
            @test_throws BoundsError k[N+1]
            # the sum is preserved; and so is the variance for K ≥ 2
            @test sum(k) ≈ 1
            if K >= 2
                @test sum(r -> r^2 * k[r], -N:N) ≈ σ^2 rtol = 1e-6
            end
            # The DC term is stored separately
            @test (kind ∈ (1, 5)) == !iszero(k.c0)
            @test length(k.c) == (kind ∈ (1, 5) ? K - 1 : K)
        end
        # Accuracy improves with K (relative to the peak of an untruncated sampled Gaussian)
        σ = 10.0
        g = sampledgaussian(σ)
        kerr(k, p) = (r = axes(g, 1); e = [(abs(i) <= k.N ? k[i] : 0.0) - g[i] for i in r]; p == Inf ? maximum(abs, e) / g[0] : sqrt(sum(abs2, e) / sum(abs2, g)))
        errs = [kerr(DCTGaussian(σ; K = K), Inf) for K in 2:6]
        @test issorted(errs; rev = true)
        @test errs[2] < 5e-3      # K = 3, the default
        @test errs[end] < 1e-4    # K = 6
        @test kerr(DCTGaussian(σ), Inf) == errs[2]
        # The default radius minimizes the L2 error
        for σ in (1.5, 3.0, 10.0), K in (2, 3, 5), kind in (5, 7)
            g = sampledgaussian(σ)
            k = DCTGaussian(σ; K = K, kind = kind, emit_warning = false)
            @test all(N -> kerr(DCTGaussian(σ; K = K, N = N, kind = kind), 2) >= kerr(k, 2), max(K, k.N - 5):k.N+5)
        end

        # Element types
        @test eltype(DCTGaussian(3)) == Float64
        @test eltype(DCTGaussian(3.0f0)) == Float32
        @test eltype(DCTGaussian(Float32, 3.0)) == Float32
        @test DCTGaussian(Float32, 3.0) ≈ DCTGaussian(3.0)
        # Tuples
        kx, ky = DCTGaussian((2.0, 5.0))
        @test kx isa ReshapedOneD{Float64,2,0} && ky isa ReshapedOneD{Float64,2,1}
        @test kx.data == DCTGaussian(2.0) && ky.data == DCTGaussian(5.0)
        @test DCTGaussian([2.0, 5.0]) == (kx, ky)
        @test DCTGaussian((2.0, 5.0); N = (9, 20))[2].data.N == 20
        @test DCTGaussian((2.0, 5.0); N = 20)[1].data.N == 20
        @test DCTGaussian((2.0, 5.0); K = 5)[1].data == DCTGaussian(2.0; K = 5)
        @test eltype(DCTGaussian(Float32, (2, 5))[1]) == Float32
        # σ = 0 is the identity
        k0 = DCTGaussian(0)
        @test axes(k0) == (0:0,) && k0[0] == 1
        @test ImageFiltering.iscopy(k0)
        img = rand(5, 6)
        @test imfilter(img, DCTGaussian((0, 0))) == img
        # Warnings and errors
        @test_logs (:warn, "σ is too small for accuracy") DCTGaussian(0.5)
        @test_logs DCTGaussian(0.5; emit_warning = false)
        @test_throws ArgumentError DCTGaussian(-1)
        @test_throws ArgumentError DCTGaussian(3; kind = 2)
        @test_throws ArgumentError DCTGaussian(3; K = 0)
        @test_throws ArgumentError DCTGaussian(3; K = 5, N = 3)

        # Fitting an arbitrary even kernel
        h = centered([1, 4, 6, 4, 1] / 16)
        @test_throws ArgumentError SlidingDCT(OffsetArray(h.parent, 0:4))
        @test_throws ArgumentError SlidingDCT(centered([1.0, 2.0, 3.0, 1.0, 1.0]))
        @test_throws ArgumentError SlidingDCT(h; K = 3, moments = (0, 2, 4, 6))
        # With K = N + 1 the representation is exact, whatever the DCT type
        for kind in (1, 3, 5, 7)
            k = SlidingDCT(h; K = 3, kind = kind)
            @test collect(k) ≈ collect(h)
        end
        k = SlidingDCT(h; K = 2)
        @test sum(k) ≈ 1
        @test sum(r -> r^2 * k[r], -2:2) ≈ sum(r -> r^2 * h[r], -2:2)
        @test eltype(SlidingDCT(centered(Float32[1, 2, 1]); K = 1)) == Float32
        @test sum(SlidingDCT(centered(Float32[1, 2, 1]); K = 1)) ≈ 4
        # A kernel that is itself a sum of cosines is recovered exactly, without moment constraints
        N = 12
        ω = [π * (2k + 1) / (2N + 1) for k in 0:2]
        hc = OffsetArray([0.3cos(ω[1] * r) - 0.2cos(ω[2] * r) + 0.1cos(ω[3] * r) for r in -N:N], -N:N)
        kc = SlidingDCT(hc; K = 3, moments = ())
        @test kc.c ≈ [0.3, -0.2, 0.1]
        @test collect(kc) ≈ collect(hc)
    end

    @testset "agreement with FIR filtering" begin
        borders = ("replicate", "circular", "symmetric", "reflect", Fill(0), Fill(0.25), Inner(), NA())
        # 1d
        x = rand(150)
        for kind in (1, 3, 5, 7), K in (1, 3), σ in (1.5, 6.0)
            k = DCTGaussian(σ; K = K, kind = kind)
            for border in borders
                @test imfilter(x, k, border) ≈ imfilter(x, firkernel(k), border, Algorithm.FIR()) atol = 1e-12
            end
        end
        # n-d with various element types
        imgs = (rand(37, 41), rand(Float32, 64, 20), rand(Gray{N0f8}, 30, 25),
                rand(RGB{N0f8}, 20, 22, 9), rand(RGB{Float32}, 3, 4, 5, 6))
        for img in imgs
            σs = ntuple(d -> 1 + 2d / 3, ndims(img))
            for (kind, K) in ((7, 3), (5, 2))
                ks = DCTGaussian(σs; K = K, kind = kind)
                for border in borders
                    border isa Fill && (border = Fill(zero(eltype(img))))
                    # Inner() requires the image to be larger than the kernel
                    border isa Inner && any(map((n, k) -> n < length(k.data), size(img), ks)) && continue
                    out = imfilter(img, ks, border)
                    ref = imfilter(img, firkernel(ks), border, Algorithm.FIR())
                    @test eltype(out) == eltype(ref)
                    @test axes(out) == axes(ref)
                    @test maxabsdiff(out, ref) < rtol(out)
                end
            end
        end
        # Filtering only some dimensions, and arrays smaller than the kernel
        img = rand(50, 3, 2)
        k = DCTGaussian(4.0)
        k3 = ntuple(d -> ReshapedOneD{3,d-1}(k), 3)
        for ks in (k3[1:1], k3[2:2], k3[1:2], k3)
            for border in borders
                # Inner() requires the image to be larger than the kernel
                border isa Inner && any(kk -> size(img, findfirst(!=(0:0), axes(kk))) < length(k), ks) && continue
                # NA() requires one kernel per dimension
                border isa NA && length(ks) < ndims(img) && continue
                @test maxabsdiff(imfilter(img, ks, border), imfilter(img, firkernel(ks), border, Algorithm.FIR())) < 1e-12
            end
        end
        @test imfilter(Float64[], DCTGaussian(3.0)) == Float64[]
        @test imfilter([2.0], DCTGaussian(3.0)) ≈ [2.0]
        # Offset arrays
        img = OffsetArray(rand(20, 30), -5:14, 3:32)
        ks = DCTGaussian((2.0, 3.0))
        out = imfilter(img, ks, "reflect")
        @test axes(out) == axes(img)
        @test out ≈ imfilter(img, firkernel(ks), "reflect", Algorithm.FIR())
        # Restricted output region and preallocated output
        img = rand(40, 40)
        ref = imfilter(img, firkernel(ks), "symmetric", Algorithm.FIR())
        out = similar(img)
        @test imfilter!(out, img, ks, "symmetric") === out
        @test out ≈ ref
        out = OffsetArray(zeros(10, 12), 11:20, 5:16)
        imfilter!(out, img, ks, "symmetric")
        @test parent(out) ≈ ref[11:20, 5:16]
        # In-place filtering (one and several stages)
        for kk in (ks, ks[1:1], (ks..., ks...))
            img2 = copy(img)
            @test imfilter!(img2, img2, kk, "reflect") === img2
            @test img2 ≈ imfilter(img, firkernel(kk), "reflect", Algorithm.FIR())
        end
        # Filtering along a single dimension
        out = similar(img)
        for border in ("replicate", Pad(:circular), Fill(0.5))
            @test imfilter!(CPU1(Algorithm.DCT()), out, img, ks[2].data, 2, border) === out
            @test out ≈ imfilter(img, firkernel((ks[2],)), border, Algorithm.FIR())
        end
        img2 = copy(img)
        imfilter!(CPU1(Algorithm.DCT()), img2, img2, ks[1].data, 1, "symmetric")
        @test img2 ≈ imfilter(img, firkernel((ks[1],)), "symmetric", Algorithm.FIR())
        @test_throws ArgumentError imfilter!(CPU1(Algorithm.DCT()), out, img, ks[1].data, 3, "symmetric")
        @test_throws DimensionMismatch imfilter!(CPU1(Algorithm.DCT()), zeros(3, 3), img, ks[1].data, 1, "symmetric")
        # Algorithm selection; all algorithms agree
        @test ImageFiltering.filter_algorithm(img, img, ks) == Algorithm.DCT()
        @test ImageFiltering.filter_algorithm(img, img, (ks[1], KernelFactors.gaussian((1.0, 1.0))[2])) != Algorithm.DCT()
        @test imfilter(img, ks, "symmetric", Algorithm.DCT()) ≈ ref
        @test imfilter(CPU1(Algorithm.DCT()), img, ks, Fill(0)) ≈ imfilter(img, firkernel(ks), Fill(0))
        @test imfilter(CPU1(Algorithm.FIR()), img, ks) ≈ imfilter(img, firkernel(ks))
        @test imfilter(img, ks, "replicate", Algorithm.FFT()) ≈ imfilter(img, firkernel(ks))
        # Cascades mixing FIR and SlidingDCT kernels (and multiple DCT kernels along a dimension)
        kf = KernelFactors.gaussian((1.0, 2.0))
        ksd = DCTGaussian((3.0, 1.5))
        mixed = (kf[1], ksd[2], ksd[1], kf[2])
        @test imfilter(img, mixed, "reflect") ≈ imfilter(img, (firkernel(mixed)...,), "reflect", Algorithm.FIR())
        for border in ("replicate", "reflect", Fill(0.3))
            @test imfilter(img, (ksd[1], ksd[1]), border) ≈ imfilter(img, (firkernel(ksd[1]), firkernel(ksd[1])), border, Algorithm.FIR())
            @test imfilter(img, (ksd[2], ksd[1], ksd[2]), border) ≈ imfilter(img, firkernel((ksd[2], ksd[1], ksd[2])), border, Algorithm.FIR())
        end
        # A Fill border must account for the sum of the kernels of previous stages
        kn = SlidingDCT(centered([1.0, 2.0, 3.0, 2.0, 1.0]); K = 3)   # sum(kn) == 9
        kn2 = kernelfactors((kn, kn))
        @test imfilter(img, kn2, Fill(0.7)) ≈ imfilter(img, firkernel(kn2), Fill(0.7), Algorithm.FIR())
        @test imfilter(img, (kn2[2], kn2[1]), Fill(0.7)) ≈ imfilter(img, firkernel((kn2[2], kn2[1])), Fill(0.7), Algorithm.FIR())
        @test imfilter(img, (ksd..., ksd...), "circular") ≈ imfilter(img, firkernel((ksd..., ksd...)), "circular", Algorithm.FIR())
        # Integer images
        imgi = rand(1:100, 30, 30)
        @test imfilter(imgi, ksd) ≈ imfilter(imgi, firkernel(ksd))
        # NaN handling: with NA() missing values are ignored
        imgn = rand(60, 60); imgn[30, 30] = NaN
        @test !any(isnan, imfilter(imgn, ksd, NA()))
        @test imfilter(imgn, ksd, NA()) ≈ imfilter(imgn, firkernel(ksd), NA(), Algorithm.FIR())
        # Otherwise, like for IIR filters, a NaN contaminates the remainder of each line
        # (unlike FIR filtering, which only contaminates the neighborhood)
        out, ref = imfilter(imgn, ksd), imfilter(imgn, firkernel(ksd), Algorithm.FIR())
        @test !isnan(out[1, 1]) && isnan(out[end, end])
        @test !isnan(ref[1, 1]) && !isnan(ref[end, end])
    end

    @testset "properties" begin
        # constant images are unchanged
        img = fill(0.7, 50, 60)
        @test imfilter(img, DCTGaussian((5.0, 12.0))) ≈ img
        img = fill(RGB{N0f8}(0.2, 0.4, 0.6), 30, 30)
        @test imfilter(img, DCTGaussian((3.0, 3.0))) ≈ img
        # Accuracy with respect to an (essentially) exact Gaussian filter: a ±8σ window
        img = rand(200, 200)
        for σ in (2.0, 8.0, 25.0)
            l = 2 * ceil(Int, 8σ) + 1
            ref = imfilter(img, KernelFactors.gaussian((σ, σ), (l, l)), "replicate", Algorithm.FFT())
            p3 = psnr(imfilter(img, DCTGaussian((σ, σ)), "replicate"), ref)
            p5 = psnr(imfilter(img, DCTGaussian((σ, σ); K = 5), "replicate"), ref)
            piir = psnr(imfilter(img, KernelFactors.IIRGaussian((σ, σ)), "replicate"), ref)
            @test p3 > 70
            @test p5 > 95
            @test p3 > piir + 15   # substantially more accurate than IIRGaussian
        end
        # the input is not modified
        img = rand(20, 20); img0 = copy(img)
        imfilter(img, DCTGaussian((2.0, 2.0)))
        @test img == img0
    end

    @testset "numerical stability" begin
        # Long signals: the recurrence must not accumulate significant round-off errors
        x = rand(200_000)
        for (σ, K, kind) in ((3.0, 3, 7), (50.0, 6, 7), (20.0, 4, 5), (20.0, 4, 1), (500.0, 3, 3))
            k = DCTGaussian(σ; K = K, kind = kind)
            out = imfilter(x, k, "reflect")
            ref = imfilter(x, firkernel(k), "reflect", Algorithm.FIR())
            @test maximum(abs, out - ref) < 1e-9
        end
        k32 = DCTGaussian(10.0f0)
        x32 = rand(Float32, 200_000)
        out = imfilter(x32, k32)
        @test eltype(out) == Float32
        @test maximum(abs, out - imfilter(x32, firkernel(DCTGaussian(10.0)))) < 1e-3
    end

    @testset "derivatives" begin
        for σ in (2.0, 7.5), order in (1, 2), kind in (1, 3, 5, 7)
            k = slidinggaussian(σ; order = order, kind = kind)
            N = k.N
            @test k isa (isodd(order) ? SlidingDST : SlidingDCT)
            @test all(r -> k[-r] ≈ (-1)^order * k[r], 0:N)
            # moments: exact on low-degree polynomials
            if order == 1
                @test sum(r -> r * k[r], -N:N) ≈ 1
            else
                @test abs(sum(k)) < 1e-12
                @test sum(r -> r^2 * k[r], -N:N) ≈ 2
            end
        end
        # accuracy with the default parameters, and improvement with K
        for σ in (3.0, 10.0), order in (1, 2)
            ref = sampledderivative(σ, order)
            @test l2err(slidinggaussian(σ; order = order), ref) < 2e-2
            errs = [l2err(slidinggaussian(σ; order = order, K = K), ref) for K in 3:6]
            @test issorted(errs; rev = true)
        end
        @test slidinggaussian(3.0) == DCTGaussian(3.0)
        @test slidinggaussian(3.0; order = 2) isa SlidingDCT{Float64,4}
        @test eltype(slidinggaussian(3.0f0; order = 1)) == Float32
        # derivatives of polynomials, away from the borders
        t = collect(1.0:300.0)
        for σ in (2.0, 10.0)
            k1, k2 = slidinggaussian(σ; order = 1), slidinggaussian(σ; order = 2)
            inner = 6ceil(Int, σ):300-6ceil(Int, σ)
            @test imfilter(3 .* t .- 7, k1)[inner] ≈ fill(3, length(inner))
            @test imfilter(0.5 .* t .^ 2 .+ t, k2)[inner] ≈ fill(1, length(inner))
            @test imfilter(0.5 .* t .^ 2 .+ t, k1)[inner] ≈ t[inner] .+ 1
        end
        # Tuples of kernels: gradients
        ks = slidinggaussian((2.0, 3.0); order = (1, 0))
        @test ks[1].data isa SlidingDST && ks[2].data isa SlidingDCT
        @test ks[2].data == DCTGaussian(3.0)
        @test slidinggaussian((2.0, 3.0); order = (0, 2), K = (3, 5))[2].data isa SlidingDCT{Float64,5}
        @test_throws DimensionMismatch slidinggaussian((2.0, 3.0); order = (1, 0, 0))
        @test_throws ArgumentError slidinggaussian(3.0; order = 3)
        @test_throws ArgumentError slidinggaussian(0; order = 1)
        @test slidinggaussian(0) == DCTGaussian(0)
        # Fitting arbitrary odd kernels
        h = centered([-1.0, -2.0, 0.0, 2.0, 1.0])
        @test_throws ArgumentError SlidingDST(centered([1.0, 2.0, 0.0, 2.0, 1.0]))
        @test collect(SlidingDST(h; K = 2)) ≈ collect(h)   # K = N: exact
        N = 12
        ω = [2π * k / (2N + 1) for k in 1:3]
        hs = OffsetArray([0.3sin(ω[1] * r) - 0.2sin(ω[2] * r) + 0.1sin(ω[3] * r) for r in -N:N], -N:N)
        @test SlidingDST(hs; K = 3, moments = ()).c ≈ [0.3, -0.2, 0.1]
        @test_throws ArgumentError SlidingDST(h; K = 3)   # K must not exceed N

        # Agreement with FIR filtering
        x = rand(150)
        for kind in (1, 5), σ in (1.5, 6.0)
            k = slidinggaussian(σ; order = 1, kind = kind)
            for border in ("replicate", "circular", "symmetric", "reflect", Fill(0), Fill(0.25), Inner())
                @test imfilter(x, k, border) ≈ imfilter(x, firkernel(k), border, Algorithm.FIR()) atol = 1e-12
            end
        end
        for img in (rand(40, 50), rand(Gray{N0f8}, 30, 25), rand(RGB{Float32}, 20, 22, 9))
            σs = ntuple(d -> 1 + d / 2, ndims(img))
            for order in ((1, ntuple(_ -> 0, ndims(img) - 1)...), (0, 2, ntuple(_ -> 0, ndims(img) - 2)...))
                ks = slidinggaussian(σs; order = order)
                for border in ("replicate", "reflect", Fill(zero(eltype(img))), Fill(oneunit(eltype(img))))
                    out = imfilter(img, ks, border)
                    @test maxabsdiff(out, imfilter(img, firkernel(ks), border, Algorithm.FIR())) < rtol(out)
                end
            end
        end
    end

    @testset "SeparableSum" begin
        kx, ky = DCTGaussian((2.0, 3.0))
        dx, dy = slidinggaussian((2.0, 3.0); order = (2, 2))
        L = SeparableSum(((dx, ky), (kx, dy)))
        @test L.weights == (true, true)
        @test ndims(L) == 2
        @test eltype(L) == Float64
        @test axes(L) == (-max(dx.data.N, kx.data.N):max(dx.data.N, kx.data.N), -max(dy.data.N, ky.data.N):max(dy.data.N, ky.data.N))
        @test occursin("SeparableSum of 2", sprint(show, L))
        @test_throws DimensionMismatch SeparableSum(((dx, ky), (kx, dy)), (1,))
        @test_throws DimensionMismatch SeparableSum(((dx, ky), (kx.data,)))
        @test DCTLoG((2.0, 3.0), K0 = 3, K2 = 4).terms == L.terms
        # Filtering is the sum of the filtered terms, for all borders and algorithms
        img = rand(40, 45)
        ref(border) = imfilter(img, firkernel((dx, ky)), border, Algorithm.FIR()) + imfilter(img, firkernel((kx, dy)), border, Algorithm.FIR())
        # (NA() is not meaningful for kernels that sum to zero, such as second derivatives)
        for border in ("replicate", "circular", "symmetric", "reflect", Fill(0.5))
            @test imfilter(img, L, border) ≈ ref(border)
        end
        @test imfilter(img, L) ≈ ref("replicate")
        out = imfilter(img, L, Inner())
        @test axes(out) == map(intersect, axes(imfilter(img, (dx, ky), Inner())), axes(imfilter(img, (kx, dy), Inner())))
        @test out ≈ imfilter(img, firkernel((dx, ky)), Inner(), Algorithm.FIR())[axes(out)...] +
                    imfilter(img, firkernel((kx, dy)), Inner(), Algorithm.FIR())[axes(out)...]
        @test imfilter(Float32, img, L) isa Matrix{Float32}
        @test imfilter(CPU1(Algorithm.DCT()), img, L, "reflect") ≈ ref("reflect")
        @test imfilter(CPU1(Algorithm.FIR()), Float64, img, L, "reflect") ≈ ref("reflect")
        out = similar(img)
        @test imfilter!(out, img, L, "reflect") === out
        @test out ≈ ref("reflect")
        @test imfilter!(CPU1(Algorithm.DCT()), out, img, L, Pad(:reflect)) ≈ ref("reflect")
        img2 = copy(img)
        imfilter!(img2, img2, L, "reflect")
        @test img2 ≈ ref("reflect")
        # Weights
        D = SeparableSum(((kx, ky), (dx, dy)), (2, -0.5))
        @test imfilter(img, D, "reflect") ≈ 2 * imfilter(img, firkernel((kx, ky)), "reflect") - 0.5 * imfilter(img, firkernel((dx, dy)), "reflect")
        @test imfilter(rand(Gray{N0f8}, 20, 20), L) isa Matrix{Gray{Float64}}

        # Laplacian of Gaussian
        img = rand(80, 90)
        for σ in (2.0, 5.0)
            ref = imfilter(img, Kernel.LoG(σ), "reflect")
            @test maximum(abs, imfilter(img, DCTLoG(σ), "reflect") - ref) < 3e-2 * maximum(abs, ref)
            # exact on quadratic images
            q = [0.5i^2 + 2j^2 + i for i in 1:80, j in 1:90]
            m = 6ceil(Int, σ)
            @test imfilter(q, DCTLoG(σ))[m:end-m, m:end-m] ≈ fill(5, 81 - 2m, 91 - 2m)
        end
        img3 = rand(30, 31, 32)
        ref = imfilter(img3, Kernel.LoG((2.0, 2.5, 3.0)), "reflect")
        @test maximum(abs, imfilter(img3, DCTLoG((2.0, 2.5, 3.0)), "reflect") - ref) < 3e-2 * maximum(abs, ref)
        @test length(DCTLoG((2.0, 2.5, 3.0)).terms) == 3
        # Difference of Gaussians, compared with an (essentially) exact one using ±8σ windows
        g(σ) = (l = 2ceil(Int, 8σ) + 1; KernelFactors.gaussian((σ, σ), (l, l)))
        for σ in (2.0, 5.0)
            exact = imfilter(img, g(σ), "reflect", Algorithm.FFT()) - imfilter(img, g(√2 * σ), "reflect", Algorithm.FFT())
            err(D) = maximum(abs, imfilter(img, D, "reflect") - exact)
            @test err(DCTDoG(σ)) < err(Kernel.DoG(σ)) / 2   # more accurate than the (truncated) dense kernel
            @test err(DCTDoG(σ; K = 5)) < 2e-4
        end
        @test DCTDoG((2.0, 3.0)).terms[2] == DCTGaussian((2√2, 3√2))
        @test DCTDoG((2.0, 3.0), (4.0, 5.0)).weights == (true, -1)
    end

    @testset "inference" begin
        img = rand(Float32, 20, 20)
        @inferred imfilter(img, DCTGaussian((2.0, 3.0)))
        @inferred imfilter(rand(RGB{N0f8}, 10, 10), DCTGaussian((2.0, 3.0)))
        @inferred imfilter(img, slidinggaussian((2.0, 3.0); order = (1, 0)))
        @inferred imfilter(img, DCTLoG(2.0))
    end
end
