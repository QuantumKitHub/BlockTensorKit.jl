# AbstractArray Interface
# -----------------------
# mostly pass everything through to the parent array, but with additional features for slicing
Base.eltype(t::AbstractBlockTensorMap) = eltype(typeof(t))

Base.ndims(t::AbstractBlockTensorMap) = numind(t)
Base.size(t::AbstractBlockTensorMap) = size(eachspace(t))
Base.size(t::AbstractBlockTensorMap, i::Int) = size(t)[i]
Base.length(t::AbstractBlockTensorMap) = prod(size(t))
Base.axes(t::AbstractBlockTensorMap) = map(Base.OneTo, size(t))
Base.axes(t::AbstractBlockTensorMap, i::Int) = Base.OneTo(i ≤ ndims(t) ? size(t, i) : 1)

Base.first(t::AbstractBlockTensorMap) = first(parent(t))
Base.last(t::AbstractBlockTensorMap) = last(parent(t))
Base.firstindex(t::AbstractBlockTensorMap) = 1
Base.firstindex(t::AbstractBlockTensorMap, i::Int) = 1
Base.lastindex(t::AbstractBlockTensorMap) = length(t)
Base.lastindex(t::AbstractBlockTensorMap, i::Int) = size(t, i)

Base.IndexStyle(::AbstractBlockTensorMap) = IndexCartesian()
Base.CartesianIndices(t::AbstractBlockTensorMap) = CartesianIndices(size(t))
Base.LinearIndices(t::AbstractBlockTensorMap) = LinearIndices(size(t))
Base.eachindex(t::AbstractBlockTensorMap) = eachindex(IndexStyle(t), t)
Base.eachindex(::IndexCartesian, t::AbstractBlockTensorMap) = CartesianIndices(t)
Base.eachindex(::IndexLinear, t::AbstractBlockTensorMap) = Base.OneTo(length(t))

Base.keys(l::Base.IndexStyle, t::AbstractBlockTensorMap) = keys(l, parent(t))
Base.haskey(t::AbstractBlockTensorMap, args...) = haskey(parent(t), args...)

Base.only(t::AbstractBlockTensorMap) = only(parent(t))
Base.isempty(t::AbstractBlockTensorMap) = isempty(parent(t))

# index checking
@inline function Base.checkbounds(t::AbstractBlockTensorMap, I...)
    checkbounds(Bool, t, I...) || Base.throw_boundserror(t, I)
    return nothing
end
@inline function Base.checkbounds(::Type{Bool}, t::AbstractBlockTensorMap, I...)
    return Base.checkbounds_indices(Bool, axes(t), I)
end
@inline function Base.checkbounds(::Type{Bool}, t::AbstractBlockTensorMap, i)
    return Base.checkindex(Bool, eachindex(IndexLinear(), t), i)
end

# TODO: make this also have Bool as first argument
function checkspaces(t::AbstractBlockTensorMap, v::AbstractTensorMap, I...)
    space(v) == eachspace(t)[I...] || throw(
        SpaceMismatch(
            "inserting a tensor of space $(space(v)) at index $I into a tensor of space $(eachspace(t)[I...])",
        ),
    )
    return nothing
end
function checkspaces(t::AbstractBlockTensorMap, v::AbstractBlockTensorMap, I...)
    V_slice = eachspace(t)[I...]
    return if V_slice isa SumSpaceIndices
        space(v) == space(V_slice) || throw(
            SpaceMismatch(
                "inserting a tensor of space $(space(v)) at index $I into a tensor of space $(space(V_slice))",
            ),
        )
    else
        space(only(v)) == V_slice || throw(
            SpaceMismatch(
                "inserting a tensor of space $(space(only(v))) at index $I into a tensor of space $(V_slice)",
            ),
        )
    end
end
function checkspaces(t::AbstractBlockTensorMap)
    iter = SumSpaceIndices(space(t))
    for I in eachindex(iter)
        iter[I] == space(t[I]) || throw(
            SpaceMismatch(
                "index $I has space $(iter[I]) but tensor has space $(space(t[I]))"
            ),
        )
    end
    return nothing
end

# scalar indexing is dispatched through:
@propagate_inbounds Base.getindex(t::AbstractBlockTensorMap, I::Vararg{Int, N}) where {N} =
    getindex(parent(t), I...)
@propagate_inbounds Base.getindex(t::AbstractBlockTensorMap, I::CartesianIndex{N}) where {N} =
    getindex(parent(t), I)
@propagate_inbounds getindex!(t::AbstractBlockTensorMap, I::Vararg{Int, N}) where {N} =
    getindex!(parent(t), I...)
@propagate_inbounds getindex!(t::AbstractBlockTensorMap, I::CartesianIndex{N}) where {N} =
    getindex!(parent(t), I)

# slicing getindex needs to correctly allocate output blocktensor:
const SliceIndex = Union{Strided.SliceIndex, AbstractVector{<:Union{Integer, Bool}}}

Base.@propagate_inbounds function Base.getindex(
        t::AbstractBlockTensorMap, indices::Vararg{SliceIndex}
    )
    V = space(eachspace(t)[indices...])
    tdst = similar(t, V)
    length(tdst) == 0 && return tdst

    # prevent discarding of singleton dimensions
    indices′ = map(indices) do ind
        return ind isa Int ? (ind:ind) : ind
    end
    Rsrc = CartesianIndices(t)[indices′...]
    Rdst = CartesianIndices(tdst)

    for (I, v) in nonzero_pairs(t)
        j = findfirst(==(I), Rsrc)
        isnothing(j) && continue
        tdst[Rdst[j]] = v
    end
    return tdst
end

# disambiguate:
@propagate_inbounds function Base.getindex(
        t::AbstractBlockTensorMap, indices::Vararg{Strided.SliceIndex}
    )
    V = space(eachspace(t)[indices...])
    tdst = similar(t, V)
    length(tdst) == 0 && return tdst

    # prevent discarding of singleton dimensions
    indices′ = map(indices) do ind
        return ind isa Int ? (ind:ind) : ind
    end
    Rsrc = CartesianIndices(t)[indices′...]
    Rdst = CartesianIndices(tdst)

    for (I, v) in nonzero_pairs(t)
        j = findfirst(==(I), Rsrc)
        isnothing(j) && continue
        tdst[Rdst[j]] = v
    end
    return tdst
end

# TODO: check if this fallback is fair
@propagate_inbounds Base.setindex!(t::AbstractBlockTensorMap, v::AbstractTensorMap, args...) = (
    setindex!(parent(t), v, args...); t
)

# ambiguity fix
function Base.setindex!(::AbstractBlockTensorMap, ::AbstractTensorMap, sectors::Tuple{I, Vararg{I}}) where {I <: Sector}
    error("invalid indexing for blocktensormap")
end
function Base.setindex!(::AbstractBlockTensorMap, ::AbstractTensorMap, ::FusionTree, ::FusionTree)
    error("invalid indexing for blocktensormap")
end

# setindex verifies structure is correct
@inline function Base.setindex!(
        t::AbstractBlockTensorMap, v::AbstractTensorMap, indices::Vararg{SliceIndex}
    )
    @boundscheck begin
        checkbounds(t, indices...)
        checkspaces(t, v, indices...)
    end
    @inbounds parent(t)[indices...] = v
    return t
end
# setindex with blocktensor needs to correctly slice-assign
@inline function Base.setindex!(
        t::AbstractBlockTensorMap, v::AbstractBlockTensorMap, indices::Vararg{SliceIndex}
    )
    @boundscheck begin
        checkbounds(t, indices...)
        checkspaces(t, v, indices...)
    end

    @inbounds copyto!(view(parent(t), indices...), parent(v))
    return t
end

# disambiguate
@inline function Base.setindex!(
        t::AbstractBlockTensorMap, v::AbstractTensorMap, indices::Vararg{Strided.SliceIndex}
    )
    @boundscheck begin
        checkbounds(t, indices...)
        checkspaces(t, v, indices...)
    end
    @inbounds parent(t)[indices...] = v
    return t
end
# disambiguate
@inline function Base.setindex!(
        t::AbstractBlockTensorMap, v::AbstractBlockTensorMap, indices::Vararg{Strided.SliceIndex},
    )
    @boundscheck begin
        checkbounds(t, indices...)
        checkspaces(t, v, indices...)
    end

    @inbounds copyto!(view(parent(t), indices...), parent(v))
    return t
end

@inline function Base.get(t::AbstractBlockTensorMap, key, default)
    @boundscheck checkbounds(t, key)
    return @inbounds get(parent(t), key, default)
end

function Base.copy(t::AbstractBlockTensorMap)
    tdst = if eltype(t) <: AdjointTensorMap
        similar(t, Base.promote_op(copy, eltype(t)), space(t))
    else
        similar(t)
    end
    return copy!(tdst, t)
end
function Base.copy!(tdst::AbstractBlockTensorMap, tsrc::AbstractBlockTensorMap)
    space(tdst) == space(tsrc) || throw(SpaceMismatch("$(space(tdst)) ≠ $(space(tsrc))"))
    @inbounds for (key, value) in nonzero_pairs(tsrc)
        tdst[key] = copy!(tdst[key], value)
    end
    return tdst
end
function Base.copyto!(
        tdst::AbstractBlockTensorMap, Rdest::CartesianIndices,
        tsrc::AbstractBlockTensorMap, Rsrc::CartesianIndices,
    )
    copyto!(parent(tdst), Rdest, parent(tsrc), Rsrc)
    return tdst
end

# generic implementation for AbstractTensorMap with Sumspace -> returns `BlockTensorMap`
# function Base.similar(
#     ::AbstractTensorMap, ::Type{TorA}, P::TensorMapSumSpace{S}
# ) where {TorA<:TensorKit.MatOrNumber,S}
#     N₁ = length(codomain(P))
#     N₂ = length(domain(P))
#     TT = blocktensormaptype(S, N₁, N₂, TorA)
#     return TT(undef, codomain(P), domain(P))
# end
# disambiguate
# function Base.similar(
#     t::TensorKit.AdjointTensorMap, T::Type{TorA}, P::TensorMapSumSpace{S}
# ) where {TorA<:TensorKit.MatOrNumber,S}
#     @invoke Base.similar(t::TensorKit.AdjointTensorMap, T::Type{TorA}, P::TensorMapSpace)
# end

Base.similar(t::AbstractBlockTensorMap) = similar(t, eltype(t), space(t))
Base.similar(t::AbstractBlockTensorMap, P::TensorMapSumSpace) = similar(t, eltype(t), P)

# make sure tensormap specializations are not used for sumspaces:
function Base.similar(
        t::AbstractTensorMap, ::Type{TorA}, P::TensorMapSumSpace{S}
    ) where {S, TorA}
    return issparse(t) ? sparse_similar(t, TorA, P) : dense_similar(t, TorA, P)
end

dense_similar(t::AbstractTensorMap, P::TensorMapSumSpace) = dense_similar(t, TK.similarstoragetype(t), P)
function dense_similar(t::AbstractTensorMap, ::Type{TorA}, P::TensorMapSumSpace) where {TorA}
    TT = similar_tensormaptype(t, TorA, P)
    return BlockTensorMap{TT}(undef, P)
end
sparse_similar(t::AbstractTensorMap, P::TensorMapSumSpace) = sparse_similar(t, TK.similarstoragetype(t), P)
function sparse_similar(t::AbstractTensorMap, ::Type{TorA}, P::TensorMapSumSpace) where {TorA}
    TT = similar_tensormaptype(t, TorA, P)
    return SparseBlockTensorMap{TT}(undef, P)
end

function similar_tensormaptype(
        ::AbstractTensorMap, T::Type{<:AbstractTensorMap}, P::TensorMapSumSpace{S}
    ) where {S}
    if isconcretetype(T)
        return tensormaptype(S, numout(P), numin(P), storagetype(T))
    else
        return AbstractTensorMap{scalartype(T), S, numout(P), numin(P)}
    end
end
function similar_tensormaptype(
        ::AbstractTensorMap, T::Type{<:AbstractVector}, P::TensorMapSumSpace{S}
    ) where {S}
    return tensormaptype(S, numout(P), numin(P), T)
end
function similar_tensormaptype(
        ::AbstractTensorMap, T::Type{<:Number}, P::TensorMapSumSpace{S}
    ) where {S}
    return tensormaptype(S, numout(P), numin(P), T)
end

function similar_tensormaptype(
        t::AbstractBlockTensorMap, T::Type{<:AbstractTensorMap}, P::TensorMapSumSpace{S}
    ) where {S}
    if eltype(t) === T && typeof(space(t)) === typeof(P)
        return T
    elseif isconcretetype(T) || T isa Union
        return tensormaptype(S, numout(P), numin(P), storagetype(T))
    else
        return AbstractTensorMap{scalartype(T), S, numout(P), numin(P)}
    end
end
function similar_tensormaptype(
        t::AbstractBlockTensorMap, M::Type{<:AbstractVector}, P::TensorMapSumSpace{S}
    ) where {S}
    if isconcretetype(eltype(t))
        return tensormaptype(S, numout(P), numin(P), M)
    else
        return AbstractTensorMap{scalartype(M), S, numout(P), numin(P)}
    end
end
function similar_tensormaptype(
        t::AbstractBlockTensorMap, T::Type{<:Number}, P::TensorMapSumSpace{S}
    ) where {S}
    if isconcretetype(eltype(t))
        M = TensorKit.similarstoragetype(t, T)
        return tensormaptype(S, numout(P), numin(P), M)
    else
        return AbstractTensorMap{T, S, numout(P), numin(P)}
    end
end

# implementation in type domain
function Base.similar(::Type{T}, P::TensorMapSumSpace) where {T <: AbstractBlockTensorMap}
    return T(undef, P)
end

# Cat
# ---
Base.eltypeof(t::AbstractBlockTensorMap) = eltype(t)

@inline function Base._cat_t(
        dims, ::Type{T}, ts::AbstractBlockTensorMap...
    ) where {T <: AbstractTensorMap}
    catdims = Base.dims2cat(dims)
    V = space(Base._cat(dims, eachspace.(ts)...))
    A = similar(ts[1], TK.storagetype(ts[1]), V)
    shape = size(A)
    if count(!iszero, catdims)::Int > 1
        zerovector!(A)
    end
    return Base.__cat(A, shape, catdims, ts...)
end

Base._copy_or_fill!(A, inds, x::AbstractBlockTensorMap) = (A[inds...] = x)

# WHY DOES BASE NOT DEFAULT TO AXES
Base.cat_indices(A::AbstractBlockTensorMap, d) = axes(A, d)
Base.cat_size(A::AbstractBlockTensorMap, d) = size(A, d)
