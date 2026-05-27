# TensorOperations
# ----------------
function TO.tensoradd_type(
        TC, A::AbstractBlockTensorMap, ::Index2Tuple{N₁, N₂}, ::Bool
    ) where {N₁, N₂}
    S = spacetype(A)
    M = TK.similarstoragetype(A, TK.promote_permute(TC, sectortype(S)))
    return if issparse(A)
        sparseblocktensormaptype(S, N₁, N₂, M)
    else
        blocktensormaptype(S, N₁, N₂, M)
    end
end
function TO.tensoradd_type(TC, A::AdjointBlockTensorMap, pA::Index2Tuple, conjA::Bool)
    return TO.tensoradd_type(TC, A', adjointtensorindices(A, pA), !conjA)
end

function TO.tensorscalar(t::AbstractBlockTensorMap{T, S, 0, 0}) where {T, S}
    return nonzero_length(t) == 0 ? zero(T) : TO.tensorscalar(only(nonzero_values(t)))
end

# tensoralloc_contract
# --------------------
for TTA in (:AbstractTensorMap, :AbstractBlockTensorMap), TTB in (:AbstractTensorMap, :AbstractBlockTensorMap)
    TTA == TTB == :AbstractTensorMap && continue
    @eval function TO.tensorcontract_type(
            TC,
            A::$TTA, ::Index2Tuple, ::Bool,
            B::$TTB, ::Index2Tuple, ::Bool,
            ::Index2Tuple{N₁, N₂},
        ) where {N₁, N₂}
        S = TK.check_spacetype(A, B)
        TC′ = TK.promote_permute(TC, sectortype(S))
        M = TK.promote_storagetype(TK.similarstoragetype(A, TC′), TK.similarstoragetype(B, TC′))
        return if issparse(A) && issparse(B)
            sparseblocktensormaptype(S, N₁, N₂, M)
        else
            blocktensormaptype(S, N₁, N₂, M)
        end
    end
end

function similarblocktype(::Type{A}, ::Type{TT}) where {A, TT}
    return Core.Compiler.return_type(similar, Tuple{A, Type{TT}, NTuple{numind(TT), Int}})
end

function TO.tensoralloc(
        ::Type{BT}, structure::TensorMapSumSpace, istemp::Val, allocator = TO.DefaultAllocator()
    ) where {BT <: AbstractBlockTensorMap}
    C = BT(undef_blocks, structure)
    issparse(C) && return C # don't fill up sparse blocks
    blockallocator(V) = TO.tensoralloc(eltype(C), V, istemp, allocator)
    map!(blockallocator, parent(C), eachspace(C))
    return C
end

# tensorfree!
# -----------
function TO.tensorfree!(t::BlockTensorMap, allocator = TO.DefaultAllocator())
    foreach(Base.Fix2(TO.tensorfree!, allocator), parent(t))
    return nothing
end
function TO.tensorfree!(t::SparseBlockTensorMap, allocator = TO.DefaultAllocator())
    foreach(Base.Fix2(TO.tensorfree!, allocator), nonzero_values(t))
    return nothing
end

function TK.trace_permute!(
        tdst::AbstractBlockTensorMap,
        tsrc::AbstractBlockTensorMap,
        (p₁, p₂)::Index2Tuple,
        (q₁, q₂)::Index2Tuple,
        α::Number, β::Number,
        backend::AbstractBackend = TO.DefaultBackend(),
    )
    # some input checks
    TK.check_spacetype(tdst, tsrc)
    if !(BraidingStyle(sectortype(tdst)) isa SymmetricBraiding)
        throw(
            SectorMismatch(
                "only tensors with symmetric braiding rules can be contracted; try `@planar` instead",
            ),
        )
    end
    (N₃ = length(q₁)) == length(q₂) ||
        throw(IndexError("number of trace indices does not match"))

    @boundscheck begin
        space(tdst) == TK.select(space(tsrc), (p₁, p₂)) ||
            throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
                    tdst = $(codomain(tdst))←$(domain(tdst)), p₁ = $(p₁), p₂ = $(p₂)"))
        all(i -> space(tsrc, q₁[i]) == dual(space(tsrc, q₂[i])), 1:N₃) ||
            throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
                    q₁ = $(q₁), q₂ = $(q₂)"))
    end

    scale!(tdst, β)
    @inbounds for (Isrc, vsrc) in nonzero_pairs(tsrc)
        TT.getindices(Isrc.I, q₁) == TT.getindices(Isrc.I, q₂) || continue
        Idst = CartesianIndex(TT.getindices(Isrc.I, (p₁..., p₂...)))
        tdst[Idst] = TensorKit.trace_permute!(
            tdst[Idst], vsrc, (p₁, p₂), (q₁, q₂), α, One(), backend
        )
    end
    return tdst
end

# PlanarOperations
# ----------------

function TK.BraidingTensor(
        V1::SumSpace{S}, V2::SumSpace{S}, adjoint::Bool = false
    ) where {S}
    T = BraidingStyle(sectortype(S)) isa SymmetricBraiding ? Float64 : ComplexF64
    return TK.BraidingTensor{T, S}(V1, V2, adjoint)
end

function TK.BraidingTensor{T, S}(
        V1::SumSpace{S}, V2::SumSpace{S}, adjoint::Bool = false
    ) where {T, S}
    τtype = BraidingTensor{T, S}
    tdst = SparseBlockTensorMap{τtype}(undef, V2 ⊗ V1, V1 ⊗ V2)
    Vs = eachspace(tdst)
    @inbounds for I in CartesianIndices(tdst)
        if I[1] == I[4] && I[2] == I[3]
            V = Vs[I]
            tdst[I] = TK.BraidingTensor{T, S}(V[2], V[1], adjoint)
        end
    end
    return tdst
end
