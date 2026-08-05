using Test
using TestExtras
using TensorKit
using BlockTensorKit
using Random

Random.seed!(1234)

@testset "BraidingTensor with SumSpace: $(sectortype(V))" for V in (
        ℂ^2, Vect[Z2Irrep](0 => 1, 1 => 1), Vect[FibonacciAnyon](:I => 1, :τ => 1),
    )
    V1, V2 = V, V ⊕ V # V1 != V2 to test adjoint
    S1, S2 = SumSpace(V1), SumSpace(V2)
    τ = @constinferred BraidingTensor(S1, S2)

    @test codomain(τ) == S2 ⊗ S1
    @test domain(τ) == S1 ⊗ S2
    @test TensorMap(τ[1, 1, 1, 1]) ≈ TensorMap(BraidingTensor(V1, V2))

    t = TensorMap(τ)
    @test t' * t ≈ id(domain(t))
    τ_adj = @constinferred BraidingTensor(S1, S2, true)
    @test codomain(τ_adj) == S1 ⊗ S2
    @test domain(τ_adj) == S2 ⊗ S1
    @test TensorMap(τ_adj) ≈ t'

    # agree with TensorKit's `braid`
    Sa, Sb, W = SumSpace(V1, V2), SumSpace(V2), SumSpace(V1)
    x = randn(ComplexF64, Sa ⊗ Sb, W)
    τm = @constinferred BraidingTensor(Sa, Sb)
    @plansor y[-1 -2; -3] := τm[-1 -2; 1 2] * x[1 2; -3]
    @test y ≈ braid(x, ((2, 1), (3,)), (1, 2, 3))
    if !(BraidingStyle(sectortype(V)) isa SymmetricBraiding)
        @test !(y ≈ braid(x, ((2, 1), (3,)), (2, 1, 3)))
    end
end
