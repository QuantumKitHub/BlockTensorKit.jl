using Test, TestExtras
using TensorKit, BlockTensorKit

@testset "CartesianSpace" begin
    using TensorKit, BlockTensorKit
    using Test, TestExtras

    using TensorKit: hassector

    ds = [2, 3, 2]
    d = sum(ds)

    V = SumSpace(Ref(ℝ) .^ ds)
    # TODO: cannot implement this because of how TensorKit does ⊕
    # @test eval(Meta.parse(sprint(show, V))) == V
    # @test eval(Meta.parse(sprint(show, typeof(V)))) == typeof(V)
    @test isa(V, VectorSpace)
    @test isa(V, ElementarySpace)

    @test isa(InnerProductStyle(V), HasInnerProduct)
    @test isa(InnerProductStyle(V), EuclideanInnerProduct)
    @test isa(V, SumSpace)

    @test !isdual(V)
    @test !isdual(V')

    @test @constinferred(hash(V)) == hash(deepcopy(V))
    @test V ==
        @constinferred(dual(V)) ==
        @constinferred(conj(V)) ==
        @constinferred(adjoint(V))
    @test field(V) == ℝ

    @test @constinferred(sectortype(V)) == Trivial
    @test ((@constinferred sectors(V))...,) == (Trivial(),)
    @test length(sectors(V)) == 1
    @test @constinferred(hassector(V, Trivial()))
    @test @constinferred(dim(V)) == d == @constinferred(dim(V, Trivial()))
    @test dim(@constinferred(typeof(V)())) == 0
    @test (sectors(typeof(V)())...,) == ()
    @test @constinferred(axes(V)) == Base.OneTo(d)
    W = @constinferred SumSpace(ℝ^1)
    @test @constinferred(unitspace(V)) == W == unitspace(typeof(V))
    @test @constinferred(⊞(V, V)) == SumSpace(vcat(V.spaces, V.spaces))
    @test @constinferred(⊞(V, unitspace(V))) == SumSpace(vcat(V.spaces, ℝ^1))
    @test @constinferred(⊞(V, V, V, V)) == SumSpace(repeat(V.spaces, 4))
    @test @constinferred(fuse(V, V)) ≅ SumSpace(ℝ^(d^2))
    @test @constinferred(fuse(V, V', V, V')) ≅ SumSpace(ℝ^(d^4))
    @test @constinferred(flip(V)) ≅ V'
    @test flip(V) ≅ V
    @test flip(V) ≾ V
    @test flip(V) ≿ V
    @test V ≺ ⊞(V, V)
    @test !(V ≻ ⊞(V, V))
end

@testset "ComplexSpace" begin
    using TensorKit, BlockTensorKit
    using Test, TestExtras

    using TensorKit: hassector

    ds = [2, 3, 2]
    d = sum(ds)
    V = SumSpace(ComplexSpace.(ds))

    @test isa(V, VectorSpace)
    @test isa(V, ElementarySpace)

    @test isa(InnerProductStyle(V), HasInnerProduct)
    @test isa(InnerProductStyle(V), EuclideanInnerProduct)
    @test isa(V, SumSpace)

    @test !isdual(V)
    @test isdual(V')

    @test @constinferred(hash(V)) == hash(deepcopy(V))
    @test @constinferred(dual(V)) == @constinferred(conj(V)) == @constinferred(adjoint(V))
    @test field(V) == ℂ

    @test @constinferred(sectortype(V)) == Trivial
    @test ((@constinferred sectors(V))...,) == (Trivial(),)
    @test length(sectors(V)) == 1
    @test @constinferred(hassector(V, Trivial()))
    @test @constinferred(dim(V)) == d == @constinferred(dim(V, Trivial()))
    @test dim(@constinferred(typeof(V)())) == 0
    @test (sectors(typeof(V)())...,) == ()
    @test @constinferred(axes(V)) == Base.OneTo(d)
    W = @constinferred SumSpace(ℂ^1)
    @test @constinferred(unitspace(V)) == W == unitspace(typeof(V))
    @test @constinferred(⊞(V, V)) == SumSpace(vcat(V.spaces, V.spaces))
    @test @constinferred(⊞(V, unitspace(V))) == SumSpace(vcat(V.spaces, ℂ^1))
    @test @constinferred(⊞(V, V, V, V)) == SumSpace(repeat(V.spaces, 4))
    @test @constinferred(fuse(V, V)) ≅ SumSpace(ℂ^(d^2))
    @test @constinferred(fuse(V, V', V, V')) ≅ SumSpace(ℂ^(d^4))
    @test @constinferred(flip(V)) ≅ V'
    @test flip(V) ≅ V
    @test flip(V) ≾ V
    @test flip(V) ≿ V
    @test V ≺ ⊞(V, V)
    @test !(V ≻ ⊞(V, V))
end

@testset "GradedSpace" begin
    using TensorKit, BlockTensorKit
    using Test, TestExtras

    using TensorKit: hassector

    V1 = U1Space(0 => 1, 1 => 1)
    V2 = U1Space(0 => 1, 1 => 2)
    V3 = U1Space(0 => 1, 1 => 1)
    d = dim(V1) + dim(V2) + dim(V3)
    V = SumSpace(V1, V2, V3)

    @test isa(V, VectorSpace)
    @test isa(V, ElementarySpace)

    @test isa(InnerProductStyle(V), HasInnerProduct)
    @test isa(InnerProductStyle(V), EuclideanInnerProduct)
    @test isa(V, SumSpace)

    @test !isdual(V)
    @test isdual(V')

    @test @constinferred(hash(V)) == hash(deepcopy(V))
    @test @constinferred(dual(V)) == @constinferred(conj(V)) == @constinferred(adjoint(V))
    @test field(V) == ℂ

    @test @constinferred(sectortype(V)) == sectortype(V1)
    @test ((@constinferred sectors(V))...,) == (U1Irrep(0), U1Irrep(1))
    @test issorted(sectors(V))
    @test length(sectors(V)) == 2
    @test @constinferred(hassector(V, U1Irrep(0)))
    @test !@constinferred(hassector(V, U1Irrep(2)))
    @test @constinferred(dim(V)) ==
        d ==
        @constinferred(dim(V, U1Irrep(0))) + @constinferred(dim(V, U1Irrep(1)))
    @test dim(@constinferred(typeof(V)())) == 0
    @test (sectors(typeof(V)())...,) == ()
    @test @constinferred(axes(V)) == Base.OneTo(d)
    W = @constinferred SumSpace(U1Space(0 => 1))
    @test @constinferred(unitspace(V)) == W == @constinferred(unitspace(typeof(V)))
    @test @constinferred(⊞(V, V)) == SumSpace(vcat(V.spaces, V.spaces))
    @test @constinferred(⊞(V, unitspace(V))) == SumSpace(vcat(V.spaces, unitspace(V1)))
    @test @constinferred(⊞(V, V, V, V)) == SumSpace(repeat(V.spaces, 4))
    @test @constinferred(fuse(V, V)) ≅ SumSpace(U1Space(0 => 9, 1 => 24, 2 => 16))
    @test @constinferred(fuse(V, V', V, V')) ≅
        SumSpace(U1Space(0 => 913, 1 => 600, -1 => 600, 2 => 144, -2 => 144))
    @test @constinferred(flip(V)) ≅ SumSpace(flip.(V.spaces)...)
    @test flip(V) ≅ V
    @test flip(V) ≾ V
    @test flip(V) ≿ V
    @test V ≺ ⊞(V, V)
    @test !(V ≻ ⊞(V, V))
end

@testset "Multifusion" begin
    using TensorKit, BlockTensorKit
    using Test, TestExtras

    using TensorKit: hassector
    using BlockTensorKit: ⊕

    I = IsingBimodule

    C0, C1, D0, D1, M, Mop = I(1, 1, 0), I(1, 1, 1), I(2, 2, 0), I(2, 2, 1), I(1, 2, 0), I(2, 1, 0)

    V1 = Vect[I](C0 => 1, C1 => 1)
    V2 = Vect[I](D0 => 1, D1 => 1)
    V3 = Vect[I](M => 1) # no Mop
    d = dim(V1) + dim(V2) + dim(V3)
    V = SumSpace(V1, V2, V3)

    @test isa(V, VectorSpace)
    @test isa(V, ElementarySpace)

    @test isa(InnerProductStyle(V), HasInnerProduct)
    @test isa(InnerProductStyle(V), EuclideanInnerProduct)
    @test isa(V, SumSpace)

    @test !isdual(V)
    @test isdual(V')

    @test @constinferred(hash(V)) == hash(deepcopy(V))
    @test @constinferred(dual(V)) == @constinferred(conj(V)) == @constinferred(adjoint(V))
    @test field(V) == ℂ

    @test unitspace(V) == unitspace(V1)

    @test @constinferred(sectortype(V)) == sectortype(V1)
    @test ((@constinferred sectors(V))...,) == (C0, C1, M, D0, D1) # sorted order
    @test issorted(sectors(V))
    @test length(sectors(V)) == 5
    @test @constinferred(hassector(V, M))
    @test !@constinferred(hassector(V, Mop))
    @test @constinferred(dim(V)) ==
        d ==
        @constinferred(sum(dim(s) for s in sectors(V)))
    @test dim(@constinferred(typeof(V)())) == 0
    @test (sectors(typeof(V)())...,) == ()

    # (left/right)unitspace tests
    WC = @constinferred SumSpace(Vect[I](C0 => 1))
    WD = @constinferred SumSpace(Vect[I](D0 => 1))
    WM = @constinferred SumSpace(V3)
    WMop = @constinferred SumSpace(Vect[I](Mop => 1))
    for W in [WC, WD]
        @test isunitspace(W)
        @test W == @constinferred(leftunitspace(W)) == @constinferred(rightunitspace(W))
        @test unitspace(typeof(W)) == ⊞(Vect[IsingBimodule]((1, 1, 0) => 1, (2, 2, 0) => 1))
    end

    @test_throws ArgumentError leftunitspace(V)
    @test_throws ArgumentError rightunitspace(V)
    @test leftunitspace(SumSpace(V1, V3)) == WC
    @test rightunitspace(SumSpace(V2, V3)) == WD
    @test leftunitspace(WMop) == WD && rightunitspace(WMop) == WC
    @test leftunitspace(WM) == WC && rightunitspace(WM) == WD
    @test unitspace(WM) == unitspace(WMop) == ⊞(Vect[IsingBimodule]((1, 1, 0) => 1, (2, 2, 0) => 1))

    Wempty = SumSpace(Vect[I]())
    Wzero = zerospace(V)
    @test unitspace(Wempty) == unitspace(Wzero)
    for f in (leftunitspace, rightunitspace)
        @test_throws ArgumentError f(Wempty)
    end

    VC = SumSpace(V1, V1)
    VCM = SumSpace(V1, V3)
    VMD = SumSpace(V2, V3)

    @test @constinferred(⊞(V, V)) == SumSpace(vcat(V.spaces, V.spaces))
    @test @constinferred(⊞(VCM, unitspace(VCM))) == SumSpace(vcat(VCM.spaces, unitspace(VCM).spaces))
    @test @constinferred(⊞(VCM, leftunitspace(VCM))) == SumSpace(vcat(VCM.spaces, leftunitspace(VCM).spaces))
    @test @constinferred(⊞(VMD, rightunitspace(VMD))) == SumSpace(vcat(VMD.spaces, rightunitspace(VMD).spaces))

    @test @constinferred(⊞(V, V, V, V)) == SumSpace(repeat(V.spaces, 4))
    @test @constinferred(fuse(VC, VC)) ≅ SumSpace(Vect[I](C0 => 8, C1 => 8))
    @test @constinferred(fuse(VC, VC', VC, VC')) ≅
        SumSpace(Vect[I](C0 => 128, C1 => 128))
    @test @constinferred(flip(V)) ≅ SumSpace(flip.(V.spaces)...)
    @test flip(V) ≅ V
    @test flip(V) ≾ V
    @test flip(V) ≿ V
    @test V ≺ ⊕(V, V)
    @test !(V ≻ ⊕(V, V))

    # blocksectors tests
    @test issetequal(@constinferred(blocksectors(one(V) ← one(V))), (C0, D0))
    @test issetequal(@constinferred(blocksectors(V ← V)), sectors(V))
    @test @constinferred(blocksectors(one(V))) == [C0, D0]
    for v in [VC, VCM, VMD]
        @test issetequal(@constinferred(blocksectors(v^2)), blocksectors(v ← v))
    end
    for v in [WM, WMop]
        @test isempty(@constinferred(blocksectors(v^2)))
        @test issetequal(@constinferred(blocksectors(v ← v)), blocksectors(v))
    end
end
