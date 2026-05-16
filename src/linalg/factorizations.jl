using MatrixAlgebraKit
using MatrixAlgebraKit: AbstractAlgorithm, YALAPACK.BlasMat, Algorithm
import MatrixAlgebraKit as MAK

# Type piracy for defining the MAK rules on BlockArrays!
# -----------------------------------------------------
for f in
    [
        :svd_compact, :svd_full, :svd_vals,
        :qr_compact, :qr_full, :qr_null,
        :lq_compact, :lq_full, :lq_null,
        :eig_full, :eig_vals, :eigh_full, :eigh_vals,
        :left_polar, :right_polar,
        :project_hermitian, :project_antihermitian, :project_isometric,
    ]
    f! = Symbol(f, :!)
    @eval MAK.default_algorithm(::typeof($f!), ::Type{T}; kwargs...) where {T <: AbstractBlockTensorMap} =
        MAK.default_algorithm($f!, eltype(T); kwargs...)
end

for f! in (
        :qr_compact!, :qr_full!, :lq_compact!, :lq_full!,
        :eig_full!, :eigh_full!, :svd_compact!, :svd_full!,
        :left_polar!, :right_polar!,
    )
    @eval function MAK.$f!(t::AbstractBlockTensorMap, F, alg::AbstractAlgorithm)
        TensorKit.foreachblock(t, F...) do _, (tblock, Fblocks...)
            Fblocks′ = MAK.$f!(copy_dense!(similar_dense(tblock), tblock), alg)
            # deal with the case where the output is not in-place
            for (b′, b) in zip(Fblocks′, Fblocks)
                b === b′ || copyto!(b, b′)
            end
            return nothing
        end
        return F
    end
end

# Handle these separately because single output instead of tuple
for f! in (
        :qr_null!, :lq_null!,
        :svd_vals!, :eig_vals!, :eigh_vals!,
        :project_hermitian!, :project_antihermitian!, :project_isometric!,
    )
    @eval function MAK.$f!(t::AbstractBlockTensorMap, N, alg::AbstractAlgorithm)
        TensorKit.foreachblock(t, N) do _, (tblock, Nblock)
            Nblock′ = MAK.$f!(copy_dense!(similar_dense(tblock), tblock), alg)
            # deal with the case where the output is not the same as the input
            Nblock === Nblock′ || copyto!(Nblock, Nblock′)
            return nothing
        end
        return N
    end
end

# Default algorithm
# -----------------
# Needed for disambiguation
for f in [
        :lq_full, :lq_compact, :lq_null,
        :qr_full, :qr_compact, :qr_null,
        :schur_full, :schur_vals,
        :eig_full, :eig_vals, :eig_trunc, :eig_trunc_no_error,
        :eigh_full, :eigh_vals, :eigh_trunc, :eigh_trunc_no_error,
        :svd_full, :svd_compact, :svd_trunc, :svd_trunc_no_error, :svd_vals,
        :left_polar, :right_polar,
        :left_orth, :right_orth, :left_null, :right_null,
        :project_hermitian, :project_antihermitian, :project_isometric,
    ]
    f! = Symbol(f, :!)
    @eval MAK.$f!(t::AbstractBlockTensorMap, alg::DefaultAlgorithm) =
        MAK.$f!(t, MAK.select_algorithm(MAK.$f!, t, nothing; alg.kwargs...))
    @eval MAK.$f!(t::AbstractBlockTensorMap, out, alg::DefaultAlgorithm) =
        MAK.$f!(t, out, MAK.select_algorithm(MAK.$f!, t, nothing; alg.kwargs...))
end

# specializations until fixes in base package
function MAK.is_left_isometric(A::BlockMatrix; atol::Real = 0, rtol::Real = MAK.defaulttol(A), norm = LinearAlgebra.norm)
    P = A' * A
    nP = norm(P) # isapprox would use `rtol * max(norm(P), norm(I))`
    for I in MAK.diagind(P)
        P[I] -= 1
    end
    return norm(P) <= max(atol, rtol * nP) # assume that the norm of I is `sqrt(n)`
end
function MAK.is_right_isometric(A::BlockMatrix; atol::Real = 0, rtol::Real = MAK.defaulttol(A), norm = LinearAlgebra.norm)
    P = A * A'
    nP = norm(P) # isapprox would use `rtol * max(norm(P), norm(I))`
    for I in MAK.diagind(P)
        P[I] -= 1
    end
    return norm(P) <= max(atol, rtol * nP) # assume that the norm of I is `sqrt(n)`
end

# Make sure sparse blocktensormaps have dense outputs
function MAK.initialize_output(::typeof(qr_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = ⊕(fuse(codomain(t)))
    Q = dense_similar(t, codomain(t) ← V_Q)
    R = dense_similar(t, V_Q ← domain(t))
    return Q, R
end
function MAK.initialize_output(::typeof(qr_compact!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    Q = dense_similar(t, codomain(t) ← V_Q)
    R = dense_similar(t, V_Q ← domain(t))
    return Q, R
end
function MAK.initialize_output(::typeof(qr_null!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(codomain(t)), V_Q)
    N = dense_similar(t, codomain(t) ← V_N)
    return N
end

function MAK.initialize_output(::typeof(lq_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = ⊕(fuse(domain(t)))
    L = dense_similar(t, codomain(t) ← V_Q)
    Q = dense_similar(t, V_Q ← domain(t))
    return L, Q
end
function MAK.initialize_output(::typeof(lq_compact!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    L = dense_similar(t, codomain(t) ← V_Q)
    Q = dense_similar(t, V_Q ← domain(t))
    return L, Q
end
function MAK.initialize_output(::typeof(lq_null!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(domain(t)), V_Q)
    N = dense_similar(t, V_N ← domain(t))
    return N
end

function MAK.initialize_output(::typeof(left_polar!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    W = dense_similar(t, space(t))
    P = dense_similar(t, domain(t) ← domain(t))
    return W, P
end

function MAK.initialize_output(::typeof(right_polar!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    P = dense_similar(t, codomain(t) ← codomain(t))
    Wᴴ = dense_similar(t, space(t))
    return P, Wᴴ
end

function MAK.initialize_output(::typeof(eigh_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_D = ⊕(fuse(domain(t)))
    T = real(scalartype(t))
    D = TK.similar_diagonal(t, T, V_D)
    V = dense_similar(t, codomain(t) ← V_D)
    return D, V
end

function MAK.initialize_output(::typeof(eig_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_D = ⊕(fuse(domain(t)))
    Tc = complex(scalartype(t))
    D = TK.similar_diagonal(t, Tc, V_D)
    V = dense_similar(t, Tc, codomain(t) ← V_D)
    return D, V
end

function MAK.initialize_output(::typeof(svd_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_cod = ⊕(fuse(codomain(t)))
    V_dom = ⊕(fuse(domain(t)))
    U = dense_similar(t, codomain(t) ← V_cod)
    S = similar(t, real(scalartype(t)), V_cod ← V_dom)
    Vᴴ = dense_similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end
function MAK.initialize_output(::typeof(svd_compact!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_cod = V_dom = infimum(fuse(codomain(t)), fuse(domain(t)))
    U = dense_similar(t, codomain(t) ← V_cod)
    S = TK.similar_diagonal(t, real(scalartype(t)), V_cod)
    Vᴴ = dense_similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end

# Disambiguate Diagonal implementations
# -------------------------------------
# these shouldn't ever happen as blocktensors aren't diagonal
for f! in (
        :eig_full!, :eig_vals!, :eigh_full!, :eigh_vals!,
        :lq_compact!, :lq_full!, :qr_compact!, :qr_full!,
        :svd_full!, :svd_compact!, :svd_vals!,
    )
    @eval MAK.initialize_output(::typeof($f!), t::AbstractBlockTensorMap, ::DiagonalAlgorithm) =
        error("Blocktensors are incompatible with diagonal algorithm")
    @eval MAK.$f!(::AbstractBlockTensorMap, x, ::DiagonalAlgorithm) =
        error("Blocktensors are incompatible with diagonal algorithm")
end

function TensorKit.Factorizations.truncate_domain!(tdst::AbstractBlockTensorMap, tsrc::AbstractBlockTensorMap, inds)
    TensorKit.foreachblock(tdst, tsrc) do c, (dst_block, src_block)
        I = get(inds, c, nothing)
        dst_dense = copy_dense!(similar_dense(dst_block), dst_block)
        src_dense = copy_dense!(similar_dense(src_block), src_block)
        @assert !isnothing(I)
        @views dst_dense .= src_dense[:, I]
        # deal with the case where the output is not in-place
        dst_dense === dst_block || copyto!(dst_block, dst_dense)
        return nothing
    end
    return tdst
end
function TensorKit.Factorizations.truncate_codomain!(tdst::AbstractBlockTensorMap, tsrc::AbstractBlockTensorMap, inds)
    TensorKit.foreachblock(tdst, tsrc) do c, (dst_block, src_block)
        I = get(inds, c, nothing)
        dst_dense = copy_dense!(similar_dense(dst_block), dst_block)
        src_dense = copy_dense!(similar_dense(src_block), src_block)
        @assert !isnothing(I)
        @views dst_dense .= src_dense[I, :]
        # deal with the case where the output is not in-place
        dst_dense === dst_block || copyto!(dst_block, dst_dense)
        return nothing
    end
    return tdst
end
