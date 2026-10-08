using Test, TestExtras
using TensorKit
using BlockTensorKit
using Random
using Combinatorics
using Adapt

Vtr = (
    SumSpace(ℂ^3),
    SumSpace(ℂ^2, ℂ^2)',
    SumSpace(ℂ^2, ℂ^2, ℂ^1),
    SumSpace(ℂ^2, ℂ^2, ℂ^2),
    SumSpace(ℂ^2, ℂ^3, ℂ^1, ℂ^1)',
)

for V in (Vtr,)
    V1, V2, V3, V4, V5 = V
    @assert V3 * V4 * V2 ≿ V1' * V5' # necessary for leftorth tests
    @assert V3 * V4 ≾ V1' * V2' * V5' # necessary for rightorth tests
end

spacelist = (Vtr,)
scalartypes = (Float64, ComplexF64)
V = first(spacelist)
# @testset "Tensors with symmetry: $(TensorKit.type_repr(sectortype(first(V))))" verbose = true failfast=true for V in
#   spacelist
I = sectortype(first(V))
V1, V2, V3, V4, V5 = V

@testset "Basic tensor properties" begin
    W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
    for T in scalartypes
        t = @inferred sparse(zeros(T, W))
        @test t isa SparseBlockTensorMap
        @test @inferred(hash(t)) == hash(deepcopy(t))
        @test scalartype(t) == T
        @test iszero(norm(t))
        @test W == @inferred codomain(t)
        @test one(W) == @inferred domain(t)
        @test (W ← one(W)) == @inferred space(t)
    end
end

@testset "Constructors" begin
    W1 = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
    W2 = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
    W3 = (codomain(W1), domain(W1))
    W4 = V1

    @testset "sprand($T)" for T in scalartypes
        t1 = @inferred sprand(T, W1, 0.5)
        @test space(t1) == W1
        t2 = @inferred sprand(T, W2, 0.5)
        @test codomain(t2) == W2 && domain(t2) == one(W2)
        t3 = @inferred sprand(T, W3..., 0.5)
        @test codomain(t3) == W3[1] && domain(t3) == W3[2]
        t4 = @inferred sprand(T, W4, 0.5)
        @test codomain(t4) == ProductSpace(W4) && domain(t4) == one(W4)
        @test norm(t1) ≠ 0
        @test norm(t2) ≠ 0
        @test norm(t3) ≠ 0
    end

    @testset "spzeros($T)" for T in scalartypes
        t1 = @inferred spzeros(T, W1)
        @test space(t1) == W1
        t2 = @inferred spzeros(T, W2)
        @test codomain(t2) == W2 && domain(t2) == one(W2)
        t3 = @inferred spzeros(T, W3...)
        @test codomain(t3) == W3[1] && domain(t3) == W3[2]
        t4 = @inferred spzeros(T, W4)
        @test codomain(t4) == ProductSpace(W4) && domain(t4) == one(W4)
        @test norm(t1) == norm(t2) == norm(t3) == norm(t4) == 0
    end
end

@testset "TensorMap conversion" begin
    W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
    for T in scalartypes
        t1 = sprand(T, W, 0.5)
        t2 = sprand(T, W, 0.5)
        t1′ = @inferred convert(TensorMap, t1)
        t2′ = @inferred convert(TensorMap, t2)
        @test norm(t1) ≈ norm(t1′)
        @test norm(t2) ≈ norm(t2′)
        @test inner(t1, t2) ≈ inner(t1′, t2′)
        t1″ = @inferred SparseBlockTensorMap(t1′, W)
        t2″ = @inferred SparseBlockTensorMap(t2′, W)
        @test t1 ≈ t1″
        @test t2 ≈ t2″
    end
end

@testset "TensorMap conversion: sectors and empty blocks" begin
    Vs = (
        Vect[U1Irrep](0 => 2, 1 => 1) ⊞ Vect[U1Irrep](-1 => 1, 2 => 1) ⊞ Vect[U1Irrep](1 => 2),
        Vect[SU2Irrep](0 => 1, 1 // 2 => 2) ⊞ Vect[SU2Irrep](1 => 1) ⊞ Vect[SU2Irrep](1 // 2 => 1, 3 // 2 => 1),
        ℂ^0 ⊞ ℂ^2 ⊞ ℂ^3,
    )
    for V in Vs, W in (V ⊗ V' ← V ⊗ V, V ⊗ V ⊗ V' ← one(V)), T in scalartypes
        t = sprand(T, W, 0.5)
        t′ = @constinferred TensorMap(t)
        @test norm(t) ≈ norm(t′)
        t″ = SparseBlockTensorMap(t′, W)
        @test t″ ≈ t
        @test all(I -> dim(eachspace(t)[I]) > 0, nonzero_keys(t″))
        @test TensorMap(BlockTensorMap(t)) ≈ t′
    end
end

@testset "Adapt" begin
    W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
    t1 = sprand(Float32, W, 0.5)
    for T in (Float64, ComplexF64)
        t2 = @testinferred adapt(Vector{T}, t1)
        @test t2 isa SparseBlockTensorMap
        @test scalartype(t2) == T
        @test storagetype(t2) == Vector{T}
        @test space(t1) == space(t2)
        @test norm(t1) ≈ norm(t2)
    end
end


@testset "Basic linear algebra" begin
    W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
    for T in (Float32, ComplexF64)
        t = rand(T, W)
        @test scalartype(t) == T
        @test space(t) == W
        @test space(t') == W'
        @test dim(t) == dim(space(t))
        @test codomain(t) == codomain(W)
        @test domain(t) == domain(W)
        @test isa(@constinferred(norm(t)), real(T))
        @test norm(t)^2 ≈ dot(t, t)
        α = rand(T)
        @test norm(α * t) ≈ abs(α) * norm(t)
        @test norm(t + t, 2) ≈ 2 * norm(t, 2)
        @test norm(t + t, 1) ≈ 2 * norm(t, 1)
        @test norm(t + t, Inf) ≈ 2 * norm(t, Inf)
        p = 3 * rand(Float64)
        @test norm(t + t, p) ≈ 2 * norm(t, p)
        @test norm(t) ≈ norm(t')

        t2 = rand(T, W)
        β = rand(T)
        @test @constinferred(dot(β * t2, α * t)) ≈ conj(β) * α * conj(dot(t, t2))
        @test dot(t2, t) ≈ conj(dot(t, t2))
        @test dot(t2, t) ≈ conj(dot(t2', t'))
        @test dot(t2, t) ≈ dot(t', t2')

        i1 = @constinferred(isomorphism(storagetype(t), V1 ⊗ V2, V2 ⊗ V1))
        i2 = @constinferred(isomorphism(storagetype(t), V2 ⊗ V1, V1 ⊗ V2))
        @test i1 * i2 == @constinferred(id(storagetype(t), V1 ⊗ V2))
        @test i2 * i1 == @constinferred(id(storagetype(t), V2 ⊗ V1))

        w = @constinferred(isometry(storagetype(t), V1 ⊗ (unitspace(V1) ⊞ unitspace(V1)), V1))
        @test dim(w) == 2 * dim(V1 ← V1)
        @test w' * w == id(storagetype(t), V1)
        @test w * w' == (w * w')^2
    end
end

@testset "norm, ==, tr: test via conversion" begin
    W = V1 ⊗ V2 ← V1 ⊗ V2
    for T in (Float64, ComplexF64)
        t = sprand(T, W, 0.5)
        t′ = convert(TensorMap, t)
        for p in (1, 2, 3, Inf)
            @test norm(t, p) ≈ norm(t′, p)
        end
        @test tr(t) ≈ tr(t′)
        @test t == copy(t)
        ts, td = SparseBlockTensorMap(t), BlockTensorMap(t)
        @test t == ts == td
        @test td == ts
        @test td != 2 * ts
        @test ts != 2 * td
        @test norm(ts, 1) ≈ norm(td, 1)
        @test t != 2 * t
        t2 = copy(t)
        t2[1] = zero(t2[1]) + t2[1]
        @test t2 == t
        @test (t2 + t) != t
        @test t != rand(T, V1 ⊗ V2 ← V1)
        @test iszero(norm(spzeros(T, W)))
        @test spzeros(T, W) == zero(t)
        @test iszero(tr(spzeros(T, W)))
    end
end

@testset "Basic linear algebra: test via conversion" begin
    W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
    for T in (Float32, ComplexF64)
        t = rand(T, W)
        t2 = rand(T, W)
        @test norm(t, 2) ≈ norm(convert(TensorMap, t), 2)
        @test dot(t2, t) ≈ dot(convert(TensorMap, t2), convert(TensorMap, t))
        α = rand(T)
        @test convert(TensorMap, α * t) ≈ α * convert(TensorMap, t)
        @test convert(TensorMap, t + t) ≈ 2 * convert(TensorMap, t)
    end
end

@testset "Real and imaginary parts" begin
    W = V1 ⊗ V2
    for T in (Float64, ComplexF64, ComplexF32)
        t = randn(T, W, W)
        @test real(convert(TensorMap, t)) == convert(TensorMap, @constinferred real(t))
        @test imag(convert(TensorMap, t)) == convert(TensorMap, @constinferred imag(t))
        t′ = @inferred complex(real(t), imag(t))
        @test t ≈ t′ ≈ real(t) + im * imag(t)
    end
end

@testset "Permutations: test via conversion" begin
    W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
    t = rand(ComplexF64, W)
    a = convert(TensorMap, t)
    for k in 0:5
        for p in permutations(1:5)
            p1 = ntuple(n -> p[n], k)
            p2 = ntuple(n -> p[k + n], 5 - k)
            t2 = permute(t, (p1, p2); copy = true)
            a2 = convert(TensorMap, t2)
            @test a2 ≈ permute(a, (p1, p2); copy = true)
            @test convert(TensorMap, transpose(t2)) ≈ transpose(a2)
        end
    end
end

@testset "Full trace: test self-consistency" begin
    t = rand(ComplexF64, V1 ⊗ V2' ⊗ V2 ⊗ V1')
    t2 = permute(t, ((1, 2), (4, 3)))
    s = @constinferred tr(t2)
    @test conj(s) ≈ tr(t2')
    if !isdual(V1)
        t2 = twist!(t2, 1)
    end
    if isdual(V2)
        t2 = twist!(t2, 2)
    end
    ss = tr(t2)
    @tensor s2 = t[a, b, b, a]
    @tensor t3[a, b] := t[a, c, c, b]
    @tensor s3 = t3[a, a]
    @test ss ≈ s2
    @test ss ≈ s3
end

@testset "Partial trace: test self-consistency" begin
    t = rand(ComplexF64, V1 ⊗ V2' ⊗ V3 ⊗ V2 ⊗ V1' ⊗ V3')
    @tensor t2[a, b] := t[c, d, b, d, c, a]
    @tensor t4[a, b, c, d] := t[d, e, b, e, c, a]
    @tensor t5[a, b] := t4[a, b, c, c]
    @test t2 ≈ t5
end

@testset "Trace: test via conversion" begin
    t = rand(ComplexF64, V1 ⊗ V2' ⊗ V3 ⊗ V2 ⊗ V1' ⊗ V3')
    @tensor t2[a, b] := t[c, d, b, d, c, a]
    @tensor t3[a, b] := convert(TensorMap, t)[c, d, b, d, c, a]
    @test t3 ≈ convert(TensorMap, t2)
end

@testset "Trace and contraction" begin
    t1 = rand(ComplexF64, V1 ⊗ V2 ⊗ V3)
    t2 = rand(ComplexF64, V2' ⊗ V4 ⊗ V1')
    t3 = t1 ⊗ t2
    @tensor ta[a, b] := t1[x, y, a] * t2[y, b, x]
    @tensor tb[a, b] := t3[x, y, a, y, b, x]
    @test ta ≈ tb
end

@testset "Tensor contraction: test via conversion" begin
    A1 = randn(ComplexF64, V1' * V2', V3')
    A2 = randn(ComplexF64, V3 * V4, V5)
    rhoL = randn(ComplexF64, V1, V1)
    rhoR = randn(ComplexF64, V5, V5)' # test adjoint tensor
    H = randn(ComplexF64, V2 * V4, V2 * V4)

    @tensor HrA12[a, s1, s2, c] :=
        rhoL[a, a'] * conj(A1[a', t1, b]) * A2[b, t2, c'] * rhoR[c', c] * H[s1, s2, t1, t2]

    @tensor HrA12array[a, s1, s2, c] :=
        convert(TensorMap, rhoL)[a, a'] *
        conj(convert(TensorMap, A1)[a', t1, b]) *
        convert(TensorMap, A2)[b, t2, c'] *
        convert(TensorMap, rhoR)[c', c] *
        convert(TensorMap, H)[s1, s2, t1, t2]

    @test HrA12array ≈ convert(TensorMap, HrA12)
end

@testset "sparse block access" begin
    Vu = Vect[U1Irrep](0 => 2, 1 => 2, -1 => 2) ⊞ Vect[U1Irrep](0 => 3, 1 => 1) ⊞ Vect[U1Irrep](0 => 1, 2 => 1)
    for W in (Vu ⊗ Vu ← Vu, Vu ⊗ Vu' ← Vu ⊗ Vu)
        for t in (sprand(W, 0.3), spzeros(W))
            td = BlockTensorMap(t)
            @test length(TensorKit.blocks(t)) == length(blocksectors(t))
            for (c, b) in TensorKit.blocks(t)
                @test b isa TensorKit.blocktype(t)
                @test @constinferred(block(t, c)) == b == block(td, c)
                @test size.(b.blocks) == size.(block(td, c).blocks)
            end
            TensorKit.foreachblock(t, td) do c, (b, bd)
                @test b == bd
            end
            for (f₁, f₂) in fusiontrees(t)
                @test t[f₁, f₂] == td[f₁, f₂]
            end
            @test t == td
        end
    end
end
