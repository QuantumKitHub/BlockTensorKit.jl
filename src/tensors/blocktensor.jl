"""
    struct BlockTensorMap{TT<:AbstractTensorMap{E,S,N₁,N₂}} <: AbstractTensorMap{E,S,N₁,N₂}

Dense `BlockTensorMap` type that stores tensors of type `TT` in a dense array.
"""
struct BlockTensorMap{TT <: AbstractTensorMap, E, S, N₁, N₂, N} <:
    AbstractBlockTensorMap{E, S, N₁, N₂}
    data::Array{TT, N}
    space::TensorMapSumSpace{S, N₁, N₂}

    # uninitialized constructor
    function BlockTensorMap{TT, E, S, N₁, N₂, N}(
            ::UndefBlocksInitializer, space::TensorMapSumSpace{S, N₁, N₂}
        ) where {E, S, N₁, N₂, N, TT <: AbstractTensorMap{E, S, N₁, N₂}}
        @assert N₁ + N₂ == N "BlockTensorMap: data has wrong number of dimensions"
        data = Array{TT, N}(undef, size(SumSpaceIndices(space)))
        return new{TT, E, S, N₁, N₂, N}(data, space)
    end

    # constructor from data
    function BlockTensorMap{TT, E, S, N₁, N₂, N}(
            data::Array{TT, N}, space::TensorMapSumSpace{S, N₁, N₂}
        ) where {E, S, N₁, N₂, N, TT <: AbstractTensorMap{E, S, N₁, N₂}}
        @assert N₁ + N₂ == N "BlockTensorMap: data has wrong number of dimensions"
        return new{TT, E, S, N₁, N₂, N}(data, space)
    end
end

function BlockTensorMap{TT, E, S, N₁, N₂, N}(
        ::UndefInitializer, space::TensorMapSumSpace{S, N₁, N₂}
    ) where {TT, E, S, N₁, N₂, N}
    tdst = BlockTensorMap{TT, E, S, N₁, N₂, N}(undef_blocks, space)
    tdst.data .= similar.(TT, SumSpaceIndices(space))
    return tdst
end
function BlockTensorMap{TT, E, S, N₁, N₂, N}(
        ::UndefInitializer, space::TensorMapSumSpace{S, N₁, N₂}
    ) where {TT′, TT <: AdjointTensorMap{<:Any, <:Any, <:Any, <:Any, TT′}, E, S, N₁, N₂, N}
    tdst = BlockTensorMap{TT, E, S, N₁, N₂, N}(undef_blocks, space)
    tdst.data .= adjoint.(similar.(TT′, adjoint.(SumSpaceIndices(space))))
    return tdst
end

# uninitialized constructor
function BlockTensorMap{TT}(
        u::Union{UndefBlocksInitializer, UndefInitializer}, space::TensorMapSumSpace{S, N₁, N₂}
    ) where {E, S, N₁, N₂, TT <: AbstractTensorMap{E, S, N₁, N₂}}
    N = N₁ + N₂
    return BlockTensorMap{TT, E, S, N₁, N₂, N}(u, space)
end

# constructor from data
function BlockTensorMap{TT}(
        data::Array{TT, N}, space::TensorMapSumSpace{S, N₁, N₂}
    ) where {E, S, N₁, N₂, N, TT <: AbstractTensorMap{E, S, N₁, N₂}}
    @assert N₁ + N₂ == N "BlockTensorMap: data has wrong number of dimensions"
    return BlockTensorMap{TT, E, S, N₁, N₂, N}(data, space)
end

function blocktensormaptype(::Type{S}, N₁::Int, N₂::Int, ::Type{T}) where {S, T}
    TT = tensormaptype(S, N₁, N₂, T)
    return BlockTensorMap{TT}
end
function blocktensormaptype(::Type{SumSpace{S}}, N₁::Int, N₂::Int, ::Type{T}) where {S, T}
    TT = tensormaptype(S, N₁, N₂, T)
    return BlockTensorMap{TT}
end

# Constructors
# ------------
function BlockTensorMap{TT}(
        data::Union{Array{TT}, UndefInitializer, UndefBlocksInitializer},
        codom::ProductSumSpace{S, N₁}, dom::ProductSumSpace{S, N₂},
    ) where {TT, S, N₁, N₂}
    return BlockTensorMap{TT}(data, codom ← dom)
end

function BlockTensorMap(
        f::Union{UndefInitializer, UndefBlocksInitializer}, space::TensorMapSumSpace{S, N₁, N₂}
    ) where {S, N₁, N₂}
    TT = tensormaptype(S, N₁, N₂, Float64)
    return BlockTensorMap{TT}(f, space)
end
function BlockTensorMap(
        f::Union{UndefInitializer, UndefBlocksInitializer},
        codom::ProductSumSpace, dom::ProductSumSpace,
    )
    return BlockTensorMap(f, codom ← dom)
end

function BlockTensorMap(
        data::Array{TT}, space::TensorMapSumSpace{S, N₁, N₂}
    ) where {S, N₁, N₂, TT <: AbstractTensorMap{<:Any, S, N₁, N₂}}
    return BlockTensorMap{TT}(data, space)
end
function BlockTensorMap(
        data::Array{<:AbstractTensorMap}, codom::ProductSumSpace, dom::ProductSumSpace
    )
    return BlockTensorMap(data, codom ← dom)
end

# AbstractBlockTensorMap -> BlockTensorMap
function BlockTensorMap(t::AbstractBlockTensorMap)
    t isa BlockTensorMap && return t # TODO: should this copy?
    tdst = BlockTensorMap{eltype(t)}(undef_blocks, space(t))
    for I in eachindex(t)
        tdst[I] = t[I]
    end
    return tdst
end

# AbstractTensorMap -> BlockTensorMap
function BlockTensorMap(t::AbstractTensorMap, space::TensorMapSumSpace)
    TT = tensormaptype(spacetype(t), numout(t), numin(t), storagetype(t))
    tdst = BlockTensorMap{TT}(undef, space)
    @inbounds for (f₁, f₂) in fusiontrees(t)
        dst = tdst[f₁, f₂]
        src = t[f₁, f₂]
        for block_index in Iterators.product(blockaxes(dst)...)
            indices = getindex.(axes(dst), block_index)
            dst[block_index...] .= @view src[indices...]
        end
    end
    return tdst
end

# Convenience constructors
# ------------------------
for (fname, felt) in ((:zeros, :zero), (:ones, :one))
    @eval begin
        function Base.$fname(::Type{TorA}, V::TensorMapSumSpace) where {TorA}
            TT = blocktensormaptype(spacetype(V), numout(V), numin(V), TorA)
            t = TT(undef, V)
            fill!(t, $felt(scalartype(t)))
            return t
        end
    end
end

for randfun in (:rand, :randn, :randexp)
    randfun! = Symbol(randfun, :!)
    @eval begin
        function Random.$randfun(
                rng::Random.AbstractRNG, ::Type{TorA}, V::TensorMapSumSpace
            ) where {TorA}
            TT = blocktensormaptype(spacetype(V), numout(V), numin(V), TorA)
            t = TT(undef, V)
            Random.$randfun!(rng, t)
            return t
        end

        function Random.$randfun!(rng::Random.AbstractRNG, t::BlockTensorMap)
            foreach(b -> Random.$randfun!(rng, b), parent(t))
            return t
        end
    end
end

# Properties
# ----------
Base.eltype(::Type{<:BlockTensorMap{TT}}) where {TT} = TT
Base.parent(t::BlockTensorMap) = t.data

# handle this separately because the storagetype of `AbstractTensorMap` is
# *always* Vector no matter the actual data storage type
TK.storagetype(t::BlockTensorMap{AbstractTensorMap{E, S, N₁, N₂}}) where {E, S, N₁, N₂} =
    foldl(TK.promote_storagetype, values(t.data))

function Base.copyto!(
        dest::BlockTensorMap, Rdest::CartesianIndices,
        src::BlockTensorMap, Rsrc::CartesianIndices,
    )
    copyto!(parent(dest), Rdest, parent(src), Rsrc)
    return dest
end

TK.space(t::BlockTensorMap) = t.space
VI.scalartype(::Type{<:BlockTensorMap{TT}}) where {TT} = scalartype(TT)

issparse(::BlockTensorMap) = false

# Utility
# -------
Base.delete!(t::BlockTensorMap, I...) = (zerovector!(getindex(t, I...)); t)

# Show
# ----
function Base.showarg(io::IO, t::BlockTensorMap, toplevel::Bool)
    !toplevel && print(io, "::")
    print(io, TK.type_repr(typeof(t)))
    return nothing
end

function TK.type_repr(::Type{BlockTensorMap{T, E, S, N₁, N₂, N}}) where {T, E, S, N₁, N₂, N}
    return "BlockTensorMap{" * TK.type_repr(T) * ", …}"
end

# Converters
# ----------
function Base.promote_rule(
        ::Type{<:BlockTensorMap{TT₁}}, ::Type{<:BlockTensorMap{TT₂}}
    ) where {TT₁, TT₂}
    TT = promote_type(TT₁, TT₂)
    return BlockTensorMap{TT}
end

function Base.convert(::Type{BlockTensorMap}, t::AbstractTensorMap)
    S = spacetype(t)
    data = fill(t, ntuple(Returns(1), numind(t)))
    tdst = BlockTensorMap(
        data,
        convert(ProductSumSpace{S, numout(t)}, codomain(t)),
        convert(ProductSumSpace{S, numin(t)}, domain(t)),
    )
    return tdst
end

# Utility
# -------
Base.haskey(t::BlockTensorMap, I::CartesianIndex) = checkbounds(Bool, t.data, I)
Base.haskey(t::BlockTensorMap, i::Int) = checkbounds(Bool, t.data, i)
