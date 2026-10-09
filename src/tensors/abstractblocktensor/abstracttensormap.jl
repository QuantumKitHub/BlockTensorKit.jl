# AbstractTensorMap Interface
# ---------------------------
# TODO: do we really want this:
# note: this goes along with the specializations of Base.similar above...
function TensorKit.tensormaptype(
        ::Type{SumSpace{S}}, N₁::Int, N₂::Int, TorA::Type
    ) where {S}
    return blocktensormaptype(S, N₁, N₂, TorA)
end

eachspace(t::AbstractBlockTensorMap) = SumSpaceIndices(space(t))

@inline function Base.setindex!(
        t::AbstractBlockTensorMap, v::AbstractBlockArray, f₁::FusionTree, f₂::FusionTree
    )
    for I in eachindex(t)
        b = view(v, Block(I.I))
        isempty(b) && continue
        x = issparse(t) ? getindex!(t, I) : t[I]
        x[f₁, f₂] = b
    end
    return t
end
@inline function Base.setindex!(
        t::AbstractBlockTensorMap, v::AbstractArray, f₁::FusionTree, f₂::FusionTree
    )
    spaces = (codomain(t)..., domain(t)...)
    uncoupleds = (f₁.uncoupled..., f₂.uncoupled...)
    bsz = map(spaces, uncoupleds) do V, uncoupled
        return dim.(V, Ref(uncoupled))
    end
    v′ = BlockedArray(v, bsz...)
    return setindex!(t, v′, f₁, f₂)
end

# Blocks and subblocks
# --------------------
_entrycache(f, t::AbstractBlockTensorMap) =
    issparse(t) ? Dict(I => f(x) for (I, x) in nonzero_pairs(t)) : map(f, parent(t))

_blockcache(x) = x
_blockcache(x::AbstractTensorMap) = (b = TK.blocks(x); b isa TK.BlockIterator ? b : x)
_cachedblock(x, c::Sector) = block(x, c)
_cachedblock(b::TK.BlockIterator, c::Sector) = b[c]

_subblockcache(x) = x
_subblockcache(x::AbstractTensorMap) = sectortype(x) === Trivial ? x : TK.subblocks(x)
_cachedsubblock(x, f) = subblock(x, f)
_cachedsubblock(s::TK.SubblockIterator, f) = s[f]

TensorKit.blocks(t::AbstractBlockTensorMap) =
    TK.BlockIterator(t, (blocksectors(t), _entrycache(_blockcache, t)))
TensorKit.block(t::AbstractBlockTensorMap, c::Sector) = _block(t, parent(t), c)

TensorKit.subblocks(t::AbstractBlockTensorMap) =
    TK.SubblockIterator(t, (TK.fusiontrees(t), _entrycache(_subblockcache, t)))
TensorKit.subblock(t::AbstractBlockTensorMap, f::Tuple{FusionTree, FusionTree}) = _subblock(t, parent(t), f)

Base.length(iter::TK.BlockIterator{<:AbstractBlockTensorMap}) = length(first(iter.structure))
Base.isdone(iter::TK.BlockIterator{<:AbstractBlockTensorMap}, state...) =
    Base.isdone(first(iter.structure), state...)
function Base.iterate(iter::TK.BlockIterator{<:AbstractBlockTensorMap}, state...)
    next = iterate(first(iter.structure), state...)
    isnothing(next) && return next
    c, newstate = next
    return c => iter[c], newstate
end
Base.getindex(iter::TK.BlockIterator{<:AbstractBlockTensorMap}, c::Sector) =
    _block(iter.t, last(iter.structure), c)

Base.length(iter::TK.SubblockIterator{<:AbstractBlockTensorMap}) = length(first(iter.structure))
Base.isdone(iter::TK.SubblockIterator{<:AbstractBlockTensorMap}, i::Int = 1) = i > length(iter)
function Base.iterate(iter::TK.SubblockIterator{<:AbstractBlockTensorMap}, i::Int = 1)
    i > length(iter) && return nothing
    f = TK.gettokenvalue(first(iter.structure), i)
    return f => iter[f], i + 1
end
Base.getindex(iter::TK.SubblockIterator{<:AbstractBlockTensorMap}, i::Int) =
    iter[TK.gettokenvalue(first(iter.structure), i)]
Base.getindex(iter::TK.SubblockIterator{<:AbstractBlockTensorMap}, f::Tuple{FusionTree, FusionTree}) =
    _subblock(iter.t, last(iter.structure), f)

function _block(t::AbstractBlockTensorMap, entries, c::Sector)
    sectortype(t) == typeof(c) || throw(SectorMismatch())
    rowdims = _blockdims(codomain(t), c)
    coldims = _blockdims(domain(t), c)
    allblocks = Matrix{_blockeltype(typeof(t))}(undef, length(rowdims), length(coldims))
    for (k, I) in enumerate(eachindex(IndexCartesian(), t))
        allblocks[k] = if !issparse(t) || haskey(entries, I)
            _cachedblock(entries[I], c)
        else
            i, j = Tuple(CartesianIndices(allblocks)[k])
            _zeroblock(t, I, rowdims[i], coldims[j], c)
        end
    end
    return mortar(allblocks, rowdims, coldims)
end

function _subblock(t::AbstractBlockTensorMap, entries, (f₁, f₂)::Tuple{FusionTree, FusionTree})
    sectortype(t) === sectortype(f₁) === sectortype(f₂) ||
        throw(SectorMismatch("Not a valid sectortype for this tensor"))
    numout(t) == length(f₁) && numin(t) == length(f₂) ||
        throw(DimensionMismatch("Invalid number of fusiontree legs for this tensor"))
    W = eachspace(t)
    subblocks = map(eachindex(IndexCartesian(), t)) do I
        V = W[I]
        sz = (dims(codomain(V), f₁.uncoupled)..., dims(domain(V), f₂.uncoupled)...)
        if prod(sz) == 0 || (issparse(t) && !haskey(entries, I))
            return sreshape(StridedView(zerovector!(similar(storagetype(t), prod(sz)))), sz)
        else
            return _cachedsubblock(entries[I], (f₁, f₂))
        end
    end
    return mortar(subblocks)
end

function _blockdims(P::ProductSumSpace{S, N}, c::Sector) where {S, N}
    return vec(
        map(CartesianIndices(map(length, P.spaces))) do I
            return blockdim(ProductSpace{S, N}(map(getindex, P.spaces, Tuple(I))), c)
        end
    )
end

function _zeroblock(t::AbstractBlockTensorMap, I, d₁::Int, d₂::Int, c::Sector)
    eltype(t) <: TensorMap || return block(zerovector!(similar(eltype(t), eachspace(t)[I])), c)
    data = zerovector!(similar(storagetype(t), d₁ * d₂))
    return reshape(view(data, 1:(d₁ * d₂)), (d₁, d₂))
end

# TODO: this might get fixed once new tensormap is implemented
TensorKit.blocksectors(t::AbstractBlockTensorMap) = blocksectors(space(t))
TensorKit.hasblock(t::AbstractBlockTensorMap, c::Sector) = c in blocksectors(t)

Base.@assume_effects :foldable function _blockeltype(::Type{TT}) where {TT <: AbstractBlockTensorMap}
    T = scalartype(TT)
    B = TK.blocktype(eltype(TT))
    return B <: AbstractMatrix{T} ? B : AbstractMatrix{T} # safeguard against type-instability
end
function TensorKit.blocktype(::Type{TT}) where {TT <: AbstractBlockTensorMap}
    BS = NTuple{2, BlockedOneTo{Int, Vector{Int}}}
    return BlockMatrix{scalartype(TT), Matrix{_blockeltype(TT)}, BS}
end

function TensorKit.foreachblock(f, t::AbstractBlockTensorMap, ts...; scheduler = nothing)
    tensors = (t, ts...)
    iters = map(_blockcache, tensors)
    foreach(union(blocksectors.(tensors)...)) do c
        return f(c, map(Base.Fix2(_cachedblock, c), iters))
    end
    return nothing
end

TK.storagetype(::Type{TT}) where {TT <: AbstractBlockTensorMap} = storagetype(eltype(TT))
