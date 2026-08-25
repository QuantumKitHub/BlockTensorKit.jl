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
    @test parent(st)[[true, false, true], :, :] == parent(st[[1, 3], :, :])
    @test parent(st)[[1, 1, 3], :, :] == parent(st[[1, 1, 3], :, :])
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
