using Test, TestExtras
using TensorKit
using BlockTensorKit
using BlockTensorKit: sprand, nonzero_keys, nonzero_length
using VectorInterface: Zero, One

Vtr = (ℂ^2 ⊞ ℂ^1, (ℂ^1 ⊞ ℂ^2)', ℂ^2 ⊞ ℂ^3, ℂ^2 ⊞ ℂ^2 ⊞ ℂ^1, (ℂ^1 ⊞ ℂ^2)')
F = Vect[FermionParity]
Vf = (
    F(0 => 1, 1 => 1) ⊞ F(0 => 2), F(1 => 1)' ⊞ F(0 => 1, 1 => 2)', F(0 => 2, 1 => 1),
    F(0 => 1) ⊞ F(1 => 1) ⊞ F(0 => 1, 1 => 1), F(0 => 1, 1 => 1)' ⊞ F(1 => 2)',
)

maketensor(sparse, T, W) = sparse ? sprand(T, W, 0.4) : randn(T, W)
dense(t) = convert(TensorMap, t)

# `(p₁..., reverse(p₂)...)` is a cyclic rotation of the planar order of a 3 ← 2 tensor
perms(f) = f === transpose! ? ((2, 3), (1, 4, 5)) : ((3, 1), (5, 2, 4))
transform!(f, C, A, p, α, β) = f === braid! ? f(C, A, p, (1, 2, 3, 4, 5), α, β) : f(C, A, p, α, β)

@testset "$f ($(sectortype(V[1])), C sparse = $Csparse, A sparse = $Asparse, adjoint = $adj)" for V in (Vtr, Vf),
        f in (permute!, braid!, transpose!), Csparse in (false, true), Asparse in (false, true), adj in (false, true)
    T = ComplexF64
    W = V[1] ⊗ V[2] ⊗ V[3] ← V[4] ⊗ V[5]
    A = maketensor(Asparse, T, adj ? W' : W)
    A = adj ? A' : A
    Ad = adj ? dense(A')' : dense(A)
    p = perms(f)
    Wdst = permute(space(A), p)
    for β in (Zero(), zero(T), One(), randn(T))
        α = randn(T)
        C = maketensor(Csparse, T, Wdst)
        Cd = transform!(f, dense(C), Ad, p, α, β)
        @test transform!(f, C, A, p, α, β) === C
        @test dense(C) ≈ Cd
        if iszero(β) && Csparse && Asparse
            @test nonzero_length(C) == nonzero_length(A)
        end
    end
    C = scale!(maketensor(Csparse, T, Wdst), NaN)
    @test dense(transform!(f, C, A, p, One(), Zero())) ≈ transform!(f, zerovector(dense(C)), Ad, p, One(), Zero())
end

@testset "permute of adjoint ($(sectortype(V[1])), sparse = $sparse)" for V in (Vtr, Vf), sparse in (false, true)
    W = V[1] ⊗ V[2] ⊗ V[3] ← V[4] ⊗ V[5]
    t = maketensor(sparse, ComplexF64, W)
    p = ((3, 1), (5, 2, 4))
    @test dense(permute(t', p)) ≈ permute(dense(t)', p)
    C = similar(t, permute(space(t'), p))
    @test dense(permute!(C, t', p)) ≈ permute(dense(t)', p)
end
