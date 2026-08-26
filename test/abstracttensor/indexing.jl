using BlockTensorKit
using BlockTensorKit: sprand
using Test
using TensorKit
using LinearAlgebra

V = SumSpace(ℂ^2, ℂ^3, ℂ^2)

for (label, blockt, TT) in (
        ("dense", rand(V ⊗ V ⊗ V), BlockTensorMap),
        ("sparse", sprand(V ⊗ V ⊗ V, 0.5), SparseBlockTensorMap),
    )
    @testset "$label indexing" begin
        # scalar indexing
        @test @inferred(blockt[1]) isa TensorMap
        @test @inferred(blockt[1, 1, 1]) isa TensorMap
        @test @inferred(blockt[CartesianIndex(1, 1, 1)]) isa TensorMap

        # colon indexing
        @test @inferred(blockt[:, :, :]) == blockt
        nnz = nonzero_length(blockt)
        for I in eachindex(blockt)
            if I in nonzero_keys(blockt)
                @test blockt[I] === blockt[I]
            else
                @test norm(blockt[I]) == 0
                @test nonzero_length(blockt) == nnz
            end
        end

        @test size(@inferred(blockt[1, :, 1])) == (1, 3, 1)
        @test size(blockt[1, [1, 3], 1]) == (1, 2, 1)
        blockt3 = @inferred blockt[[1], [1], 1]
        @test blockt3 isa TT
        @test length(blockt3) == 1

        # repeated indices duplicate blocks, as for `AbstractArray`
        b = @inferred blockt[[1, 1, 3], :, :]
        @test size(b) == (3, 3, 3)
        for i in 1:3, j in 1:3
            @test b[1, i, j] == blockt[1, i, j]
            @test b[2, i, j] == blockt[1, i, j]
            @test b[3, i, j] == blockt[3, i, j]
        end
        # a dropped duplicate used to surface as uninitialized memory
        @test norm(b[2, 1, 1]) == norm(blockt[1, 1, 1])
        @test ndims(blockt[:, 1:2, [1, 1]]) == 3

        # logical indexing
        @test @inferred(blockt[[true, false, true], :, 1]) == blockt[[1, 3], :, 1]
        @test blockt[BitVector([true, false, true]), :, :] == blockt[[1, 3], :, :]
        @test size(blockt[falses(3), :, :]) == (0, 3, 3)
        @test_throws BoundsError blockt[[true, false], :, :]
        @test_throws BoundsError blockt[[true, false, true, false], :, :]

        # reversed and stepped ranges
        @test blockt[3:-1:1, :, :] == blockt[[3, 2, 1], :, :]
        @test blockt[1:2:3, :, :] == blockt[[1, 3], :, :]

        # empty slices
        @test size(blockt[Int[], :, :]) == (0, 3, 3)
        @test length(blockt[Int[], :, :]) == 0

        # integer types other than `Int`
        @test blockt[Int8(1), 1:2, 1] == blockt[1, 1:2, 1]
        @test blockt[UInt(1), 1:2, 1] == blockt[1, 1:2, 1]
        @test blockt[Int8(1), Int8(1), Int8(1)] == blockt[1, 1, 1]

        # invalid indexing
        @test_throws ArgumentError blockt[:]
        @test_throws ArgumentError blockt[[1]]
        @test_throws BoundsError blockt[:, :]
        @test_throws BoundsError blockt[4, :, :]
        @test_throws BoundsError @inbounds blockt[4, :, :]

        # slice assignment (index 1 and 3 share their space)
        t2 = copy(blockt)
        t2[[1], :, :] = blockt[[3], :, :]
        for i in 1:3, j in 1:3
            @test t2[1, i, j] == blockt[3, i, j]
        end
    end
end

# single-index slicing of one-dimensional block tensors
for (label, t1) in (("dense", rand(V ← one(V))), ("sparse", sprand(V ← one(V), 0.8)))
    @testset "$label 1-dimensional indexing" begin
        @test t1[:] == t1
        @test norm(t1[1:2])^2 ≈ norm(t1[1])^2 + norm(t1[2])^2
        @test size(t1[[1, 1]]) == (2,)
        @test t1[[1, 1]][2] == t1[1]
    end
end

# the parent array has its own slicing implementation
@testset "parent array slicing" begin
    st = sprand(V ⊗ V ⊗ V, 0.5)
    for inds in ([true, false, true], [1, 3], [1, 1, 3], 1:2)
        a, b = parent(st)[inds, :, :], parent(st[inds, :, :])
        @test a == b
        @test length(a.data) == length(b.data)  # must not densify
    end
end

# `nonzero_*` on the parent array must report stored entries, not every entry
@testset "sparse parent accessors" begin
    st = sprand(V ⊗ V ⊗ V, 0.5)
    A = parent(st)
    stored = length(A.data)
    @test 0 < stored < length(A)
    @test nonzero_length(A) == stored
    @test length(collect(nonzero_keys(A))) == stored
    @test length(collect(nonzero_pairs(A))) == stored
    @test length(collect(nonzero_values(A))) == stored
    @test length(parent(A[1:3, :, :]).data) == stored
end

@testset "sparse slice assignment" begin
    Vh = SumSpace(ℂ^2, ℂ^2, ℂ^2)  # homogeneous, so any index can be assigned to any other
    for _ in 1:10
        t = sprand(Vh ⊗ Vh ⊗ Vh, 0.4)
        src = sprand(Vh ⊗ Vh ⊗ Vh, 0.4)[1:2, :, :]
        expected = Set(I for I in nonzero_keys(t) if I[1] > 2)
        union!(expected, nonzero_keys(src))
        blocks = Dict(I => src[I] for I in nonzero_keys(src))
        for I in nonzero_keys(t)
            I[1] > 2 && (blocks[I] = t[I])
        end
        t[1:2, :, :] = src
        @test Set(nonzero_keys(t)) == expected
        for (I, x) in blocks
            @test t[I] === x
        end
    end

    # structural zeros of the source clear the destination
    t = sprand(Vh ⊗ Vh ⊗ Vh, 1.0)
    @test nonzero_length(t) == 27
    t[1:2, :, :] = spzeros(Float64, Vh ⊗ Vh ⊗ Vh)[1:2, :, :]
    @test nonzero_length(t) == 9
    @test all(I -> I[1] == 3, nonzero_keys(t))
end

@testset "copyto! on the parent array" begin
    Vh = SumSpace(ℂ^2, ℂ^2, ℂ^2)  # homogeneous, so blocks can move between indices
    st = sprand(Vh ⊗ Vh ⊗ Vh, 0.5)
    A = parent(st)

    # into a fresh array from a view: the destination spans the view exactly
    dst = parent(st[1:2, :, :])
    dst[1, 1, 1] = rand(eachspace(st)[1, 1, 1])
    copyto!(dst, view(A, 1:2, :, :))
    @test Set(nonzero_keys(dst)) == Set(I for I in nonzero_keys(A) if I[1] ≤ 2)
    for I in nonzero_keys(dst)
        @test dst[I] === A[I]
    end

    # region-to-region, including a stepped region
    for st_ in (1, 2)
        a, b = parent(sprand(Vh ⊗ Vh ⊗ Vh, 0.5)), parent(sprand(Vh ⊗ Vh ⊗ Vh, 0.5))
        Rsrc = CartesianIndices((1:st_:3, 1:1, 1:1))
        Rdest = CartesianIndices((1:st_:3, 2:2, 3:3))
        before = Dict(I => a[I] for I in nonzero_keys(a))
        copyto!(a, Rdest, b, Rsrc)
        for (Pd, Ps) in zip(Rdest, Rsrc)
            if Ps in nonzero_keys(b)
                @test a[Pd] === b[Ps]
            else
                @test (Pd in keys(before)) == (Pd in nonzero_keys(a))
            end
        end
    end
end

@testset "cat" begin
    st = sprand(V ⊗ V ⊗ V, 0.5)
    c = cat(st, st; dims = 1)
    @test size(c) == (6, 3, 3)
    @test nonzero_length(c) == 2 * nonzero_length(st)
    for I in nonzero_keys(st)
        @test c[I] === st[I]
        @test c[I + CartesianIndex(3, 0, 0)] === st[I]
    end
end

# guard against slicing work proportional to the destination region again
@testset "slicing scales with nnz" begin
    D = 512
    tb = spzeros(Float64, SumSpace(fill(ℂ^1, D)...) ⊗ SumSpace(ℂ^2) ← SumSpace(ℂ^2) ⊗ SumSpace(fill(ℂ^1, D)...))
    for i in 1:D
        tb[i, 1, 1, i] = rand(eachspace(tb)[i, 1, 1, i])
        i < D && (tb[i, 1, 1, i + 1] = rand(eachspace(tb)[i, 1, 1, i + 1]))
    end
    inds = (2:(D - 1), :, :, 2:(D - 1))
    s = tb[inds...]
    @test nonzero_length(s) == 2 * (D - 2) - 1
    @test @allocated(tb[inds...]) < 2_000_000
end

# guard against assignment work proportional to the assigned region
@testset "assignment scales with nnz" begin
    D = 48
    Vb = SumSpace(fill(ℂ^1, D)...)
    tb = spzeros(Float64, Vb ⊗ Vb ← Vb ⊗ Vb)
    for i in 1:D
        tb[i, i, i, i] = rand(eachspace(tb)[i, i, i, i])
    end
    inds = ntuple(_ -> 2:(D - 1), 4)
    src = tb[inds...]
    tb[inds...] = src                     # warm up
    nnz = nonzero_length(tb)
    # the assigned region holds (D - 2)^4 ~ 4.5e6 blocks but only D are stored, so anything
    # proportional to the region takes >100 ms here while this takes microseconds
    @test (@elapsed tb[inds...] = src) < 0.02
    @test nonzero_length(tb) == nnz
end
