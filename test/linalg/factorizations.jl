using Test
using TestExtras
using TensorKit
using BlockTensorKit
using Random
using Combinatorics
using LinearAlgebra
using Adapt
using JLArrays

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
eltypes = (Float64, ComplexF64)

for V in spacelist
    I = sectortype(first(V))
    Istr = TensorKit.type_repr(I)
    println("---------------------------------------")
    println("Factorizations with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "Factorizations with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        W = V1 ⊗ V2

        @testset "QR decomposition" begin
            for T in eltypes,
                    t in (
                        rand(T, W, W), rand(T, W, W)', rand(T, W, V1), rand(T, V1, W)',
                    )

                Q, R = @constinferred qr_full(t)
                @test Q * R ≈ t
                @test isunitary(Q)

                Q, R = @constinferred qr_compact(t)
                @test Q * R ≈ t
                @test isisometric(Q)

                Q, R = @constinferred left_orth(t; alg = :qr)
                @test Q * R ≈ t
                @test isisometric(Q)

                N = @constinferred qr_null(t)
                @test isisometric(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))

                N = @constinferred left_null(t; alg = :qr)
                @test isisometric(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))
            end

            # empty tensor
            for T in eltypes
                t = rand(T, V1 ⊗ V2, zero(V1))

                Q, R = @constinferred qr_full(t)
                @test Q * R ≈ t
                @test isunitary(Q)
                @test dim(R) == dim(t) == 0

                Q, R = @constinferred qr_compact(t)
                @test Q * R ≈ t
                @test isisometric(Q)
                @test dim(Q) == dim(R) == dim(t)

                Q, R = @constinferred left_orth(t; alg = :qr)
                @test Q * R ≈ t
                @test isisometric(Q)
                @test dim(Q) == dim(R) == dim(t)

                N = @constinferred qr_null(t)
                @test isunitary(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))
            end
        end

        @testset "LQ decomposition" begin
            for T in eltypes,
                    t in (
                        rand(T, W, W), rand(T, W, W)', rand(T, W, V1), rand(T, V1, W)',
                    )

                L, Q = @constinferred lq_full(t)
                @test L * Q ≈ t
                @test isunitary(Q)

                L, Q = @constinferred lq_compact(t)
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)

                L, Q = @constinferred right_orth(t; alg = :lq)
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)

                Nᴴ = @constinferred lq_null(t)
                @test isisometric(Nᴴ; side = :right)
                @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))
            end

            for T in eltypes
                # empty tensor
                t = rand(T, zero(V1), V1 ⊗ V2)

                L, Q = @constinferred lq_full(t)
                @test L * Q ≈ t
                @test isunitary(Q)
                @test dim(L) == dim(t) == 0

                L, Q = @constinferred lq_compact(t)
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)
                @test dim(Q) == dim(L) == dim(t)

                L, Q = @constinferred right_orth(t; alg = :lq)
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)
                @test dim(Q) == dim(L) == dim(t)

                Nᴴ = @constinferred lq_null(t)
                @test isunitary(Nᴴ)
                @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))
            end
        end

        @testset "Polar decomposition" begin
            for T in eltypes,
                    t in (
                        rand(T, W, W), rand(T, W, W)', rand(T, W, V1), rand(T, V1, W)',
                    )

                @assert domain(t) ≾ codomain(t)
                w, p = @constinferred left_polar(t)
                @test w * p ≈ t
                @test isisometric(w)
                # @test isposdef(p)

                w, p = @constinferred left_orth(t; alg = :polar)
                @test w * p ≈ t
                @test isisometric(w)
            end

            for T in eltypes,
                    t in (rand(T, W, W), rand(T, W, W)', rand(T, V1, W), rand(T, W, V1)')

                @assert codomain(t) ≾ domain(t)
                p, wᴴ = @constinferred right_polar(t)
                @test p * wᴴ ≈ t
                @test isisometric(wᴴ; side = :right)
                # @test isposdef(p)

                p, wᴴ = @constinferred right_orth(t; alg = :polar)
                @test p * wᴴ ≈ t
                @test isisometric(wᴴ; side = :right)
            end
        end

        @testset "SVD" begin
            for T in eltypes,
                    t in (
                        rand(T, W, W), rand(T, W, W)',
                        rand(T, W, V1), rand(T, V1, W),
                        rand(T, W, V1)', rand(T, V1, W)',
                    )

                u, s, vᴴ = @constinferred svd_full(t)
                @test u * s * vᴴ ≈ t
                @test isunitary(u)
                @test isunitary(vᴴ)

                u, s, vᴴ = @constinferred svd_compact(t)
                @test u * s * vᴴ ≈ t
                @test isisometric(u)
                # @test isposdef(s)
                @test isisometric(vᴴ; side = :right)

                s′ = LinearAlgebra.diag(s)
                for (c, b) in pairs(LinearAlgebra.svdvals(t))
                    @test b ≈ s′[c]
                end

                v, c = @constinferred left_orth(t; alg = :svd)
                @test v * c ≈ t
                @test isisometric(v)

                N = @constinferred left_null(t; alg = :svd)
                @test isisometric(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))

                Nᴴ = @constinferred right_null(t; alg = :svd)
                @test isisometric(Nᴴ; side = :right)
                @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))
            end

            # empty tensor
            for T in eltypes, t in (rand(T, W, zero(V1)), rand(T, zero(V1), W))
                U, S, Vᴴ = @constinferred svd_full(t)
                @test U * S * Vᴴ ≈ t
                @test isunitary(U)
                @test isunitary(Vᴴ)

                U, S, Vᴴ = @constinferred svd_compact(t)
                @test U * S * Vᴴ ≈ t
                @test dim(U) == dim(S) == dim(Vᴴ) == dim(t) == 0
            end
        end

        @testset "truncated SVD" begin
            for T in eltypes,
                    t in (
                        randn(T, W, W), randn(T, W, W)',
                        randn(T, W, V1), randn(T, V1, W),
                        randn(T, W, V1)', randn(T, V1, W)',
                    )

                @constinferred normalize!(t)

                U, S, Vᴴ = @constinferred svd_trunc(t; trunc = notrunc())
                @test U * S * Vᴴ ≈ t
                @test isisometric(U)
                @test isisometric(Vᴴ; side = :right)

                trunc = truncrank(dim(domain(S)) ÷ 2)
                U1, S1, Vᴴ1 = @constinferred svd_trunc(t; trunc)
                @test t * Vᴴ1' ≈ U1 * S1
                @test isisometric(U1)
                @test isisometric(Vᴴ1; side = :right)
                @test dim(domain(S1)) <= trunc.howmany

                λ = minimum(minimum, values(LinearAlgebra.diag(S1)))
                trunc = trunctol(; atol = λ - 10eps(λ))
                U2, S2, Vᴴ2 = @constinferred svd_trunc(t; trunc)
                @test t * Vᴴ2' ≈ U2 * S2
                @test isisometric(U2)
                @test isisometric(Vᴴ2; side = :right)
                @test minimum(minimum, values(LinearAlgebra.diag(S1))) >= λ
                @test U2 ≈ U1
                @test S2 ≈ S1
                @test Vᴴ2 ≈ Vᴴ1

                trunc = truncspace(space(S2, 1))
                U3, S3, Vᴴ3 = @constinferred svd_trunc(t; trunc)
                @test t * Vᴴ3' ≈ U3 * S3
                @test isisometric(U3)
                @test isisometric(Vᴴ3; side = :right)
                @test space(S3, 1) ≾ space(S2, 1)

                trunc = truncerror(; atol = 0.5)
                U4, S4, Vᴴ4 = @constinferred svd_trunc(t; trunc)
                @test t * Vᴴ4' ≈ U4 * S4
                @test isisometric(U4)
                @test isisometric(Vᴴ4; side = :right)
                @test norm(t - U4 * S4 * Vᴴ4) <= 0.5
            end
        end

        @testset "Eigenvalue decomposition" begin
            for T in eltypes,
                    t in (
                        rand(T, V1, V1), rand(T, W, W), rand(T, W, W)',
                    )

                d, v = @constinferred eig_full(t)
                @test t * v ≈ v * d

                d′ = LinearAlgebra.diag(d)
                for (c, b) in pairs(LinearAlgebra.eigvals(t))
                    @test sort(b; by = abs) ≈ sort(d′[c]; by = abs)
                end

                # vdv = v' * v
                # vdv = (vdv + vdv') / 2
                # @test @constinferred isposdef(vdv)
                # t isa DiagonalTensorMap || @test !isposdef(t) # unlikely for non-hermitian map

                d, v = @constinferred eig_trunc(t; trunc = truncrank(dim(domain(t)) ÷ 2))
                @test t * v ≈ v * d
                @test dim(domain(d)) ≤ dim(domain(t)) ÷ 2

                t2 = (t + t')
                D, V = eigen(t2)
                @test isisometric(V)
                D̃, Ṽ = @constinferred eigh_full(t2)
                @test D ≈ D̃
                @test V ≈ Ṽ
                # λ = minimum(
                #     minimum(real(LinearAlgebra.diag(b)))
                #         for (c, b) in blocks(D)
                # )
                # @test cond(Ṽ) ≈ one(real(T))
                # @test isposdef(t2) == isposdef(λ)
                # @test isposdef(t2 - λ * one(t2) + 0.1 * one(t2))
                # @test !isposdef(t2 - λ * one(t2) - 0.1 * one(t2))

                d, v = @constinferred eigh_full(t2)
                @test t2 * v ≈ v * d
                @test isunitary(v)

                # λ = minimum(minimum(real(LinearAlgebra.diag(b))) for (c, b) in blocks(d))
                # @test cond(v) ≈ one(real(T))
                # @test isposdef(t2) == isposdef(λ)
                # @test isposdef(t2 - λ * one(t2) + 0.1 * one(t2))
                # @test !isposdef(t2 - λ * one(t2) - 0.1 * one(t2))

                d, v = @constinferred eigh_trunc(t2; trunc = truncrank(dim(domain(t2)) ÷ 2))
                @test t2 * v ≈ v * d
                @test dim(domain(d)) ≤ dim(domain(t2)) ÷ 2
            end
        end

        # TODO: currently not supported
        # @testset "Condition number and rank" begin
        #     for T in eltypes,
        #             t in (
        #                 rand(T, W, W), rand(T, W, W)',
        #                 rand(T, W, V1), rand(T, V1, W),
        #                 rand(T, W, V1)', rand(T, V1, W)',
        #             )
        #
        #         d1, d2 = dim(codomain(t)), dim(domain(t))
        #         @test rank(t) == min(d1, d2)
        #         M = left_null(t)
        #         @test @constinferred(rank(M)) + rank(t) == d1
        #         Mᴴ = right_null(t)
        #         @test rank(Mᴴ) + rank(t) == d2
        #     end
        #     for T in eltypes
        #         u = unitary(T, V1 ⊗ V2, V1 ⊗ V2)
        #         @test @constinferred(cond(u)) ≈ one(real(T))
        #         @test @constinferred(rank(u)) == dim(V1 ⊗ V2)
        #
        #         t = rand(T, zero(V1), W)
        #         @test rank(t) == 0
        #         t2 = rand(T, zero(V1) * zero(V2), zero(V1) * zero(V2))
        #         @test rank(t2) == 0
        #         @test cond(t2) == 0.0
        #     end
        #     for T in eltypes, t in (rand(T, W, W), rand(T, W, W)')
        #         t += t'
        #         vals = @constinferred LinearAlgebra.eigvals(t)
        #         λmax = maximum(s -> maximum(abs, s), values(vals))
        #         λmin = minimum(s -> minimum(abs, s), values(vals))
        #         @test cond(t) ≈ λmax / λmin
        #     end
        # end
    end
end

# @testset "Tensors with symmetry: $(TensorKit.type_repr(sectortype(first(V))))" verbose = true failfast=true for V in
#   spacelist
# I = sectortype(first(V))
# V1, V2, V3, V4, V5 = V
# W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
# W_empty = V1 ⊗ V2 ← eltype(V1)()
#
# const leftorth_algs = (
#     TensorKit.QR(),
#     TensorKit.QRpos(),
#     TensorKit.QL(),
#     TensorKit.QLpos(),
#     TensorKit.Polar(),
#     TensorKit.SVD(),
#     TensorKit.SDD(),
# )
#
# @testset "leftorth with $alg" for alg in leftorth_algs
#     for T in (Float32, ComplexF64), isadjoint in (false, true)
#         t = isadjoint ? rand(T, W)' : rand(T, W)
#         Q, R = @inferred leftorth(t, ((3, 4, 2), (1, 5)); alg)
#         QdQ = Q' * Q
#         @test QdQ ≈ one(QdQ)
#         @test Q * R ≈ permute(t, ((3, 4, 2), (1, 5)))
#         if alg isa Polar
#             @test isposdef(R)
#             @test domain(R) == codomain(R) == space(t, 1)' ⊗ space(t, 5)'
#         end
#
#         t_empty = isadjoint ? rand(T, W_empty')' : rand(T, W_empty)
#         Q, R = @constinferred leftorth(t_empty; alg)
#         @test Q == t_empty
#         @test dim(Q) == dim(R) == 0
#     end
# end
#
# const leftnull_algs = (TensorKit.QR(), TensorKit.SVD(), TensorKit.SDD())
# @testset "leftnull with $alg" for alg in leftnull_algs
#     for T in (Float32, ComplexF64), isadjoint in (false, true)
#         t = isadjoint ? rand(T, W)' : rand(T, W)
#         N = @constinferred leftnull(t, ((3, 4, 2), (1, 5)); alg = alg)
#         NdN = N' * N
#         @test NdN ≈ one(NdN)
#         @test norm(N' * permute(t, ((3, 4, 2), (1, 5)))) < 100 * eps(norm(t))
#
#         t_empty = isadjoint ? rand(T, W_empty')' : rand(T, W_empty)
#         N = @constinferred leftnull(t_empty; alg = alg)
#         @test N' * N ≈ id(domain(N))
#         @test N * N' ≈ id(codomain(N))
#     end
# end
#
# const rightorth_algs = (
#     TensorKit.RQ(),
#     TensorKit.RQpos(),
#     TensorKit.LQ(),
#     TensorKit.LQpos(),
#     TensorKit.Polar(),
#     TensorKit.SVD(),
#     TensorKit.SDD(),
# )
# @testset "rightorth with $alg" for alg in rightorth_algs
#     for T in (Float32, ComplexF64), isadjoint in (false, true)
#         t = isadjoint ? rand(T, W)' : rand(T, W)
#         L, Q = @constinferred rightorth(t, ((3, 4), (2, 1, 5)); alg = alg)
#         QQd = Q * Q'
#         @test QQd ≈ one(QQd)
#         @test L * Q ≈ permute(t, ((3, 4), (2, 1, 5)))
#         if alg isa Polar
#             @test isposdef(L)
#             @test domain(L) == codomain(L) == space(t, 3) ⊗ space(t, 4)
#         end
#
#         t_empty = isadjoint ? rand(T, W_empty)' : rand(T, W_empty')
#         L, Q = @constinferred rightorth(t_empty; alg = alg)
#         @test Q == t_empty
#         @test dim(Q) == dim(L) == 0
#     end
# end
#
# const rightnull_algs = (TensorKit.LQ(), TensorKit.SVD(), TensorKit.SDD())
# @testset "rightnull with $alg" for alg in rightnull_algs
#     for T in (Float32, ComplexF64), isadjoint in (false, true)
#         t = isadjoint ? rand(T, W)' : rand(T, W)
#         M = @constinferred rightnull(t, ((3, 4), (2, 1, 5)); alg = alg)
#         MMd = M * M'
#         @test MMd ≈ one(MMd)
#         @test norm(permute(t, ((3, 4), (2, 1, 5))) * M') < 100 * eps(norm(t))
#
#         t_empty = isadjoint ? rand(T, W_empty)' : rand(T, W_empty')
#         M = @constinferred rightnull(t_empty; alg)
#         @test M * M' ≈ id(codomain(M))
#         @test M' * M ≈ id(domain(M))
#     end
# end
#
# const svd_algs = (TensorKit.SVD(), TensorKit.SDD())
# @testset "tsvd with $alg" for alg in svd_algs
#     for T in (Float32, ComplexF64), isadjoint in (false, true)
#         t = isadjoint ? rand(T, W)' : rand(T, W)
#         U, S, V = @constinferred tsvd(t, ((3, 4, 2), (1, 5)); alg)
#         UdU = U' * U
#         @test UdU ≈ one(UdU)
#         VVd = V * V'
#         @test VVd ≈ one(VVd)
#         @test U * S * V ≈ permute(t, ((3, 4, 2), (1, 5)))
#
#         t_empty = isadjoint ? rand(T, W_empty')' : rand(T, W_empty)
#         U, S, V = @inferred tsvd(t_empty; alg = alg)
#         @test U == t_empty
#         @test dim(U) == dim(S) == dim(V)
#     end
# end

#         t = Tensor(rand, T, V1 ⊗ V1' ⊗ V2 ⊗ V2')
#         @testset "eig and isposdef" begin
#             D, V = eigen(t, (1, 3), (2, 4))
#             D̃, Ṽ = @constinferred eig(t, (1, 3), (2, 4))
#             @test D ≈ D̃
#             @test V ≈ Ṽ
#             VdV = V' * V
#             VdV = (VdV + VdV') / 2
#             @test isposdef(VdV)
#             t2 = permute(t, (1, 3), (2, 4))
#             @test t2 * V ≈ V * D
#             @test !isposdef(t2) # unlikely for non-hermitian map
#             t2 = (t2 + t2')
#             D, V = eigen(t2)
#             VdV = V' * V
#             @test VdV ≈ one(VdV)
#             D̃, Ṽ = @constinferred eigh(t2)
#             @test D ≈ D̃
#             @test V ≈ Ṽ
#             λ = minimum(minimum(real(LinearAlgebra.diag(b))) for (c, b) in blocks(D))
#             @test isposdef(t2) == isposdef(λ)
#             @test isposdef(t2 - λ * one(t2) + 0.1 * one(t2))
#             @test !isposdef(t2 - λ * one(t2) - 0.1 * one(t2))
#         end
#     end
# end
# @timedtestset "Tensor truncation" begin
#     for T in (Float32, ComplexF64)
#         for p in (1, 2, 3, Inf)
#             # Test both a normal tensor and an adjoint one.
#             ts = (TensorMap(randn, T, V1 ⊗ V2 ⊗ V3, V4 ⊗ V5),
#                   TensorMap(randn, T, V4 ⊗ V5, V1 ⊗ V2 ⊗ V3)')
#             for t in ts
#                 U₀, S₀, V₀, = tsvd(t)
#                 t = rmul!(t, 1 / norm(S₀, p))
#                 U, S, V, ϵ = @constinferred tsvd(t; trunc=truncerr(5e-1), p=p)
#                 # @show p, ϵ
#                 # @show domain(S)
#                 # @test min(space(S,1), space(S₀,1)) != space(S₀,1)
#                 U′, S′, V′, ϵ′ = tsvd(t; trunc=truncerr(nextfloat(ϵ)), p=p)
#                 @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
#                 U′, S′, V′, ϵ′ = tsvd(t; trunc=truncdim(ceil(Int, dim(domain(S)))), p=p)
#                 @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
#                 U′, S′, V′, ϵ′ = tsvd(t; trunc=truncspace(space(S, 1)), p=p)
#                 @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
#                 # results with truncationcutoff cannot be compared because they don't take degeneracy into account, and thus truncate differently
#                 U, S, V, ϵ = tsvd(t; trunc=truncbelow(1 / dim(domain(S₀))), p=p)
#                 # @show p, ϵ
#                 # @show domain(S)
#                 # @test min(space(S,1), space(S₀,1)) != space(S₀,1)
#                 U′, S′, V′, ϵ′ = tsvd(t; trunc=truncspace(space(S, 1)), p=p)
#                 @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
#             end
#         end
#     end
# end

@testset "In-place write-back into block storage" begin
    V1 = SumSpace(ℂ^2, ℂ^2)
    V2 = SumSpace(ℂ^3, ℂ^1)
    D = DiagonalTensorMap(rand(dim(V1)), V1)
    D′ = DiagonalTensorMap(D.data, ℂ^4)
    t = rand(V1 ← V1 ⊗ V2)
    t0 = TensorMap(t)
    @test TensorMap(lmul!(D, t)) ≈ D′ * t0
    t = rand(V1 ⊗ V2 ← V1)
    t0 = TensorMap(t)
    @test TensorMap(rmul!(t, D)) ≈ t0 * D′

    t = rand(V1 ← V1 ⊗ V2)
    b = block(t, first(blocksectors(t)))
    d = BlockTensorKit.copy_dense!(BlockTensorKit.similar_dense(b), b) .* 2
    BlockTensorKit.copy_blocks!(b, d)
    @test BlockTensorKit.copy_dense!(BlockTensorKit.similar_dense(b), b) == d

    tg = adapt(JLVector{Float64}, t)
    c = first(blocksectors(t))
    BlockTensorKit.copy_blocks!(block(t, c), d .* 3)
    BlockTensorKit.copy_blocks!(block(tg, c), JLArray(d .* 3))
    @test adapt(Vector{Float64}, tg) ≈ t
end
