# SparseBlockTensorMap parent array
# ---------------------------------
struct SparseTensorArray{S, N₁, N₂, T <: AbstractTensorMap{<:Any, S, N₁, N₂}, N} <:
    AbstractArray{T, N}
    data::Dict{CartesianIndex{N}, T}
    space::TensorMapSumSpace{S, N₁, N₂}
    function SparseTensorArray{S, N₁, N₂, T, N}(
            data::Dict{CartesianIndex{N}, T}, space::TensorMapSumSpace{S, N₁, N₂}
        ) where {S, N₁, N₂, T, N}
        N₁ + N₂ == N || throw(
            TypeError(
                :SparseTensorArray,
                SparseTensorArray{S, N₁, N₂, T, N₁ + N₂},
                SparseTensorArray{S, N₁, N₂, T, N},
            ),
        )
        return new{S, N₁, N₂, T, N}(data, space)
    end
end

function SparseTensorArray{S, N₁, N₂, T, N}(
        ::UndefInitializer, space::TensorMapSumSpace{S, N₁, N₂}
    ) where {S, N₁, N₂, T <: AbstractTensorMap{<:Any, S, N₁, N₂}, N}
    return SparseTensorArray{S, N₁, N₂, T, N}(Dict{CartesianIndex{N}, T}(), space)
end

function SparseTensorArray(
        data::Dict{CartesianIndex{N}, T}, space::TensorMapSumSpace{S, N₁, N₂}
    ) where {S, N₁, N₂, T, N}
    return SparseTensorArray{S, N₁, N₂, T, N}(data, space)
end

Base.pairs(A::SparseTensorArray) = pairs(A.data)
Base.keys(A::SparseTensorArray) = keys(A.data)
Base.values(A::SparseTensorArray) = values(A.data)

nonzero_keys(A::SparseTensorArray) = keys(A.data)
nonzero_values(A::SparseTensorArray) = values(A.data)
nonzero_pairs(A::SparseTensorArray) = pairs(A.data)
nonzero_length(A::SparseTensorArray) = length(A.data)

TensorKit.space(A::SparseTensorArray) = A.space
TensorKit.codomain(A::SparseTensorArray) = codomain(space(A))
TensorKit.domain(A::SparseTensorArray) = domain(space(A))

TensorKit.numout(A::SparseTensorArray) = numout(eltype(A))
TensorKit.numout(::Type{T}) where {T <: SparseTensorArray} = numout(eltype(A))
TensorKit.numin(A::SparseTensorArray) = numin(eltype(A))
TensorKit.numin(::Type{T}) where {T <: SparseTensorArray} = numin(eltype(A))

# AbstractArray interface
# -----------------------
Base.size(A::SparseTensorArray) = ntuple(Base.Fix1(size, A), ndims(A))
function Base.size(A::SparseTensorArray, i::Int)
    1 ≤ i ≤ ndims(A) || throw(ArgumentError("Invalid number of dimensions"))
    return i <= numout(A) ? length(codomain(A)[i]) : length(domain(A)[i - numout(A)])
end

@propagate_inbounds function Base.getindex(
        A::SparseTensorArray{S, N₁, N₂, T, N}, I::Vararg{Int, N}
    ) where {S, N₁, N₂, T, N}
    @boundscheck checkbounds(A, I...)
    return @inbounds get(A.data, CartesianIndex(I)) do
        return fill!(similar(T, eachspace(A)[I...]), zero(scalartype(T)))
    end
end
@propagate_inbounds function getindex!(
        A::SparseTensorArray{S, N₁, N₂, T, N}, I::CartesianIndex{N}
    ) where {S, N₁, N₂, T, N}
    @boundscheck checkbounds(A, I)
    return @inbounds get!(A.data, I) do
        return fill!(similar(T, eachspace(A)[I]), zero(scalartype(T)))
    end
end
@propagate_inbounds function getindex!(
        A::SparseTensorArray{S, N₁, N₂, T, N}, I::Vararg{Int, N}
    ) where {S, N₁, N₂, T, N}
    return getindex!(A, CartesianIndex(I))
end
@propagate_inbounds function Base.setindex!(
        A::SparseTensorArray{S, N₁, N₂, T, N}, v, I::Vararg{Int, N}
    ) where {S, N₁, N₂, T, N}
    @boundscheck begin
        checkbounds(A, I...)
        checkspaces(A, v, I...)
    end
    @inbounds A.data[CartesianIndex(I)] = v # implicit converter
    return A
end

function Base.delete!(A::SparseTensorArray, I::Vararg{Int, N}) where {N}
    return delete!(A.data, CartesianIndex(I))
end
Base.delete!(A::SparseTensorArray, I::CartesianIndex) = delete!(A.data, I)
Base.empty!(A::SparseTensorArray) = empty!(A.data)
function Base.haskey(A::SparseTensorArray, I::Vararg{Int, N}) where {N}
    return haskey(A.data, CartesianIndex(I))
end
Base.haskey(A::SparseTensorArray, I::CartesianIndex) = haskey(A.data, I)

function Base.similar(
        ::SparseTensorArray, ::Type{T}, spaces::TensorMapSumSpace{S, N₁, N₂}
    ) where {S, N₁, N₂, T <: AbstractTensorMap{<:Any, S, N₁, N₂}}
    N = N₁ + N₂
    return SparseTensorArray{S, N₁, N₂, T, N}(Dict{CartesianIndex{N}, T}(), spaces)
end

_undropped(inds::Tuple) = map(I -> I isa Base.ScalarIndex ? (I:I) : I, inds)

# clear the entries of `A` selected by `inds` that `v` does not store
function _deletemissing!(A::SparseTensorArray, inds::Tuple, v)
    # sweep whichever of the selected region and the stored entries is smaller
    if length(v) ≤ nonzero_length(A)
        for I in eachindex(IndexCartesian(), v)
            haskey(v, I) || delete!(A, CartesianIndex(Base.reindex(inds, I.I)))
        end
    else
        maps = map(_invert_index, size(A), inds)
        for J in collect(nonzero_keys(A))
            rs = map(_dstrange, maps, J.I)
            any(isempty, rs) && continue
            any(P -> haskey(v, CartesianIndex(P)), Iterators.product(rs...)) && continue
            delete!(A, J)
        end
    end
    return A
end

# the destination spans exactly the viewed region, so everything not copied is dropped
Base.@propagate_inbounds function Base.copyto!(
        t::SparseTensorArray, v::SubArray{T, N, A}
    ) where {T, N, A <: SparseTensorArray}
    empty!(t)
    return _copyslice!(t, parent(v), _undropped(Base.parentindices(v)))
end

Base.@propagate_inbounds function Base.copyto!(
        t::SubArray{T, N, A}, v::SparseTensorArray
    ) where {T, N, A <: SparseTensorArray}
    inds = _undropped(Base.parentindices(t))
    _deletemissing!(parent(t), inds, v)
    for (I, x) in nonzero_pairs(v)
        parent(t)[Base.reindex(inds, I.I)...] = x
    end
    return t
end

Base.@propagate_inbounds function Base.copyto!(
        dest::SparseTensorArray, Rdest::CartesianIndices,
        src::SparseTensorArray, Rsrc::CartesianIndices,
    )
    isempty(Rdest) && return dest
    if size(Rdest) != size(Rsrc)
        throw(
            ArgumentError(
                "source and destination must have same size (got $(size(Rsrc)) and $(size(Rdest)))",
            ),
        )
    end
    @boundscheck begin
        checkbounds(dest, first(Rdest))
        checkbounds(dest, last(Rdest))
        checkbounds(src, first(Rsrc))
        checkbounds(src, last(Rsrc))
    end
    maps = map(_invert_index, size(src), Rsrc.indices)
    for (I, x) in nonzero_pairs(src)
        rs = map(_dstrange, maps, I.I)
        any(isempty, rs) && continue
        for P in Iterators.product(rs...)
            dest[Rdest[P...]] = x
        end
    end
    return dest
end

# non-scalar indexing
# -------------------
# specialisations to have non-scalar indexing behave as expected
function Base._unsafe_getindex(
        ::IndexCartesian,
        t::SparseTensorArray{S, N₁, N₂, T, N}, I::Vararg{Union{Real, AbstractArray}, N},
    ) where {S, N₁, N₂, T, N}
    dest = similar(t, eltype(t), space(eachspace(t)[I...]))
    return _copyslice!(dest, t, Base.to_indices(t, I))
end

# Space checking
# --------------
eachspace(A::SparseTensorArray) = SumSpaceIndices(A.space)
function checkspaces(A::SparseTensorArray, v::AbstractTensorMap, I...)
    return space(v) == eachspace(A)[I...] || throw(
        SpaceMismatch(
            "inserting a tensor of space $(space(v)) at $(I) into a SparseTensorArray with space $(eachspace(A))",
        ),
    )
    return nothing
end
