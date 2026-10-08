# TensorOperations
# ----------------

# tensoradd!
# ----------
# `TensorKit` is free to reorganize its own index manipulation kernels, and `TO.tensoradd!` does
# not necessarily route through `permute!` or `add_transform!`. Intercepting the public
# `TO.tensoradd!` entry point instead keeps the blockwise implementations reachable regardless.
function TO.tensoradd!(
        C::BlockTensorMap, A::BlockTensorMap, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number, backend, allocator
    )
    Cdata = parent(C)
    Adata = permutedims(StridedView(parent(A)), (pA[1]..., pA[2]...))
    @inbounds for I in eachindex(Cdata, Adata)
        Cdata[I] = TO.tensoradd!(Cdata[I], Adata[I], pA, conjA, α, β, backend, allocator)
    end
    return C
end
function TO.tensoradd!(
        C::AbstractBlockTensorMap, A::AbstractBlockTensorMap, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number, backend, allocator
    )
    scale!(C, β)
    p_lin = (pA[1]..., pA[2]...)
    @inbounds for (I, v) in nonzero_pairs(A)
        I′ = CartesianIndex(TT.getindices(I.I, p_lin))
        C[I′] = TO.tensoradd!(
            getindex!(C, I′, allocator), v, pA, conjA, α, One(), backend, allocator
        )
    end
    return C
end
# adjoints are absorbed into the conjugation flag and the permutation
function TO.tensoradd!(
        C::AbstractBlockTensorMap, A::AdjointBlockTensorMap, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number, backend, allocator
    )
    return TO.tensoradd!(C, A', adjointtensorindices(A, pA), !conjA, α, β, backend, allocator)
end
# a block tensor holding a single block is interchangeable with that block
function TO.tensoradd!(
        C::TensorMap, A::BlockTensorMap, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number, backend, allocator
    )
    return TO.tensoradd!(C, only(A), pA, conjA, α, β, backend, allocator)
end
function TO.tensoradd!(
        C::BlockTensorMap, A::TensorMap, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number, backend, allocator
    )
    TO.tensoradd!(only(C), A, pA, conjA, α, β, backend, allocator)
    return C
end

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

# sparse results allocate exactly the blocks that will be written, so `tensorfree!` is consistent
const BlockOrAdjoint = Union{AbstractBlockTensorMap, AdjointBlockTensorMap}

_tensoralloc(ttype, structure, keys, istemp::Val, allocator) =
    TO.tensoralloc(ttype, structure, istemp, allocator)
function _tensoralloc(
        ttype::Type{<:SparseBlockTensorMap}, structure, keys, istemp::Val, allocator
    )
    C = ttype(undef_blocks, structure)
    Vs = eachspace(C)
    for I in keys
        haskey(C, I) || (C[I] = TO.tensoralloc(eltype(C), Vs[I], istemp, allocator))
    end
    return C
end

function contract_keys(A, pA::Index2Tuple, B, pB::Index2Tuple, pAB::Index2Tuple)
    OB = NTuple{length(pB[2]), Int}
    openB = Dict{NTuple{length(pB[1]), Int}, Vector{OB}}()
    for IB in nonzero_keys(B)
        push!(get!(Vector{OB}, openB, TT.getindices(IB.I, pB[1])), TT.getindices(IB.I, pB[2]))
    end
    p = TO.linearize(pAB)
    keys = Set{CartesianIndex{length(p)}}()
    for IA in nonzero_keys(A)
        oA = TT.getindices(IA.I, pA[1])
        for oB in get(openB, TT.getindices(IA.I, pA[2]), ())
            push!(keys, CartesianIndex(TT.getindices((oA..., oB...), p)))
        end
    end
    return keys
end

function TO.tensoralloc_add(
        TC, A::BlockOrAdjoint, pA::Index2Tuple, conjA::Bool,
        istemp::Val = Val(false), allocator = TO.DefaultAllocator()
    )
    ttype = TO.tensoradd_type(TC, A, pA, conjA)
    structure = TO.tensoradd_structure(A, pA, conjA)
    p = TO.linearize(pA)
    keys = (CartesianIndex(TT.getindices(I.I, p)) for I in nonzero_keys(A))
    return _tensoralloc(ttype, structure, keys, istemp, allocator)
end

function TO.tensoralloc_contract(
        TC, A::BlockOrAdjoint, pA::Index2Tuple, conjA::Bool,
        B::BlockOrAdjoint, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple, istemp::Val = Val(false), allocator = TO.DefaultAllocator()
    )
    ttype = TO.tensorcontract_type(TC, A, pA, conjA, B, pB, conjB, pAB)
    structure = TO.tensorcontract_structure(A, pA, conjA, B, pB, conjB, pAB)
    keys = contract_keys(A, pA, B, pB, pAB)
    return _tensoralloc(ttype, structure, keys, istemp, allocator)
end

# contract directly into `C` when it is a valid BLAS destination, otherwise go through a
# temporary holding only the product blocks instead of a copy of all blocks of `C`
function TO.tensorcontract!(
        C::SparseBlockTensorMap,
        A::BlockOrAdjoint, pA::Index2Tuple, conjA::Bool,
        B::BlockOrAdjoint, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple, α::Number, β::Number, backend, allocator
    )
    ipAB = TO.oindABinC(pAB, pA, pB)
    if TO.isblasdestination(C, ipAB) || TO.isblasdestination(C, reverse(ipAB))
        foreach(I -> getindex!(C, I, allocator), contract_keys(A, pA, B, pB, pAB))
        return _tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, backend, allocator)
    end
    N₁ = length(pA[1])
    pAB′ = (ntuple(identity, N₁), ntuple(i -> N₁ + i, length(pB[2])))
    AB = TO.tensoralloc_contract(
        scalartype(C), A, pA, conjA, B, pB, conjB, pAB′, Val(true), allocator
    )
    _tensorcontract!(AB, A, pA, conjA, B, pB, conjB, pAB′, One(), Zero(), backend, allocator)
    TO.tensoradd!(C, AB, pAB, false, α, β, backend, allocator)
    TO.tensorfree!(AB, allocator)
    return C
end
function _tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, backend, allocator)
    return @invoke TO.tensorcontract!(
        C::AbstractTensorMap,
        A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
        B::AbstractTensorMap, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple, α::Number, β::Number, backend::Any, allocator::Any
    )
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
    τtype = TK.braidingtensortype(S, T)
    return τtype(V1, V2, adjoint)
end
function TK.BraidingTensor{T, S, A}(
        V1::SumSpace{S}, V2::SumSpace{S}, adjoint::Bool = false
    ) where {T, S, A}
    τtype = BraidingTensor{T, S, A}
    cod, dom = adjoint ? (V1 ⊗ V2, V2 ⊗ V1) : (V2 ⊗ V1, V1 ⊗ V2)
    tdst = SparseBlockTensorMap{τtype}(undef, cod, dom)
    Vs = eachspace(tdst)
    @inbounds for I in CartesianIndices(tdst)
        if I[1] == I[4] && I[2] == I[3]
            V = Vs[I]
            tdst[I] = adjoint ? τtype(V[1], V[2], true) : τtype(V[2], V[1], false)
        end
    end
    return tdst
end

TK.braidingtensortype(::Type{SumSpace{S}}, ::Type{TorA}) where {S <: IndexSpace, TorA} =
    TK.braidingtensortype(S, TorA)

# TODO: remove once proper promotion rules in TensorKit are in place
TK.BraidingTensor{T, S, A}(V1::S, V2::SumSpace{S}, adjoint::Bool = false) where {T, S, A} =
    BraidingTensor{T, S, A}(promote(V1, V2)..., adjoint)
TK.BraidingTensor{T, S, A}(V1::SumSpace{S}, V2::S, adjoint::Bool = false) where {T, S, A} =
    BraidingTensor{T, S, A}(promote(V1, V2)..., adjoint)
