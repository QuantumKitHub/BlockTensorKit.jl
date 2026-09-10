"""
    struct SumSpace{S<:ElementarySpace} <: ElementarySpace

A (lazy) direct sum of elementary vector spaces of type `S`.
"""
struct SumSpace{S <: ElementarySpace} <: ElementarySpace
    spaces::Vector{S}
    dual::Bool
    function SumSpace{S}(spaces::Vector{S}, dual::Bool) where {S}
        allequal(==(dual) ∘ isdual, spaces) || throw(ArgumentError("Invalid mix of dual flags"))
        return new{S}(spaces, dual)
    end
end

function SumSpace(V::S, spaces::S...; dual::Bool = isdual(V)) where {S <: ElementarySpace}
    return SumSpace(collect((V, spaces...)); dual)
end
function SumSpace(spaces::Vector{S}; dual::Bool = isempty(spaces) ? false : isdual(first(spaces))) where {S}
    return SumSpace{S}(spaces, dual)
end
SumSpace{S}(; dual::Bool = false) where {S} = SumSpace{S}(S[], dual)

# Convenience aliases
const ProductSumSpace{S, N} = ProductSpace{SumSpace{S}, N}
const TensorSumSpace{S} = TensorSpace{SumSpace{S}}
const TensorMapSumSpace{S, N₁, N₂} = TensorMapSpace{SumSpace{S}, N₁, N₂}

# unicode name
"""
    V1 ⊞ V2...
    boxplus(V1::ElementarySpace, V2::ElementarySpace...)

Create a lazy representation of the direct sum of the supplied vector spaces, which retains the order.
See also [`SumSpace`](@ref).
"""
function ⊞ end
const boxplus = ⊞

⊞(V₁::VectorSpace, V₂::VectorSpace) = ⊞(promote(V₁, V₂)...)
⊞(V::Vararg{VectorSpace}) = reduce(⊞, V)

⊞(V::ElementarySpace) = V isa SumSpace ? V : SumSpace(V)
function (V₁::S ⊞ V₂::S) where {S <: ElementarySpace}
    return if isdual(V₁) == isdual(V₂)
        SumSpace(V₁, V₂)
    else
        throw(SpaceMismatch("Direct sum of a vector space and its dual does not exist"))
    end
end
function (V₁::SumSpace{S} ⊞ V₂::SumSpace{S}) where {S}
    V = SumSpace(vcat(V₁.spaces, V₂.spaces))
    allequal(isdual, V.spaces) ||
        throw(SpaceMismatch("Direct sum of a vector space and its dual does not exist"))
    return V
end

# AbstractArray behavior
# ----------------------
Base.size(S::SumSpace) = size(S.spaces)
Base.length(S::SumSpace) = length(S.spaces)

Base.getindex(S::SumSpace, i::Int) = S.spaces[i]
Base.getindex(S::SumSpace, i) = SumSpace(S.spaces[i])
Base.setindex!(S::SumSpace, args...) = setindex!(S.spaces, args...)

Base.iterate(S::SumSpace, args...) = iterate(S.spaces, args...)

Base.lastindex(S::SumSpace) = lastindex(S.spaces)
Base.firstindex(S::SumSpace) = firstindex(S.spaces)

Base.eltype(V::SumSpace) = eltype(typeof(V))
Base.eltype(::Type{SumSpace{S}}) where {S} = S

Base.axes(S::SumSpace) = Base.OneTo(dim(S))
Base.axes(S::SumSpace, n::Int) = axes(S.spaces, n)
function Base.axes(S::SumSpace, c::Sector)
    offset = 0
    a = []
    for s in S.spaces
        a = push!(a, axes(s, c) .+ offset)
        offset += dim(s)
    end
    return collect(flatten(a))
end

Base.hash(S::SumSpace, h::UInt) = hash(S.spaces, h)
Base.:(==)(S1::SumSpace, S2::SumSpace) = S1.spaces == S2.spaces
@inline Base.isassigned(S::SumSpace, i::Int) = isassigned(S.spaces, i)

# VectorSpace behavior
# --------------------
TensorKit.InnerProductStyle(S::Type{<:SumSpace}) = InnerProductStyle(eltype(S))
TensorKit.spacetype(S::Type{<:SumSpace}) = eltype(S)
TensorKit.spacetype(::Type{<:ProductSumSpace{S}}) where {S} = S
TensorKit.spacetype(::Type{<:TensorMapSumSpace{S}}) where {S} = S
TensorKit.sectortype(S::Type{<:SumSpace}) = sectortype(eltype(S))
TensorKit.field(::Type{SumSpace{S}}) where {S} = field(S)

"""
    sumspacetype(::Union{S,Type{S}}) where {S<:ElementarySpace}

Return the type of a `SumSpace` with elements of type `S`.
"""
sumspacetype(::Type{S}) where {S <: ElementarySpace} = SumSpace{S}

TensorKit.dims(S::SumSpace) = map(dim, S.spaces)
TensorKit.dim(S::SumSpace, n::Int) = dim(S.spaces[n])
TensorKit.dim(S::SumSpace) = sum(dims(S))

TensorKit.isdual(S::SumSpace) = S.dual
TensorKit.dual(S::SumSpace{CartesianSpace}) = SumSpace(map(dual, S.spaces); dual = false)
TensorKit.dual(S::SumSpace) = SumSpace(map(dual, S.spaces); dual = !isdual(S))
Base.conj(S::SumSpace) = dual(S)
TensorKit.flip(S::SumSpace{CartesianSpace}) = SumSpace(map(flip, S.spaces); dual = false)
TensorKit.flip(S::SumSpace) = SumSpace(map(flip, S.spaces); dual = isdual(S))

function TensorKit.hassector(S::SumSpace, s::Sector)
    return mapreduce(v -> hassector(v, s), |, S.spaces; init = false)
end
function TensorKit.hassector(S::SumSpace, ::Trivial)
    return mapreduce(v -> hassector(v, Trivial()), |, S.spaces; init = false)
end

TensorKit.sectors(S::SumSpace) = TensorKit._sectors(S, sectortype(S))
function TensorKit._sectors(V::SumSpace, ::Type{Trivial})
    return OneOrNoneIterator(dim(V) != 0, Trivial())
end
function TensorKit._sectors(S::SumSpace, ::Type{I}) where {I}
    s = Set{I}()
    for v in S.spaces
        union!(s, sectors(v))
    end
    # sorted for a canonical order, consistent with `GradedSpace` and `blocksectors`
    return sort!(collect(s))
end

TensorKit.dim(S::SumSpace, sector::Sector) = sum(v -> dim(v, sector), S.spaces; init = 0)
# ambiguity fix:
TensorKit.dim(S::SumSpace, ::Trivial) = sum(v -> dim(v, Trivial()), S.spaces; init = 0)

# TODO: find a better name for this function

TensorKit.compose(V, W) = TensorKit.compose(promote(V, W)...)
# bit of a hack to make spacechecks happy?
Base.:(==)(V::SumSpace{S}, W::S) where {S} = ==(promote(V, W)...)
Base.:(==)(V::S, W::SumSpace{S}) where {S} = ==(promote(V, W)...)
Base.:(==)(V::ProductSumSpace{S}, W::ProductSpace{S}) where {S <: ElementarySpace} = ==(promote(V, W)...)
Base.:(==)(V::ProductSpace{S}, W::ProductSumSpace{S}) where {S <: ElementarySpace} = ==(promote(V, W)...)
function Base.:(==)(V::TensorMapSumSpace{S}, W::TensorMapSpace{S}) where {S <: IndexSpace}
    return ==(promote(V, W)...)
end
function Base.:(==)(V::TensorMapSpace{S}, W::TensorMapSumSpace{S}) where {S <: IndexSpace}
    return ==(promote(V, W)...)
end
# disambiguate
function Base.:(==)(V::TensorMapSumSpace{S}, W::TensorMapSumSpace{S}) where {S <: IndexSpace}
    return @invoke ==(V::HomSpace, W::HomSpace)
end


TensorKit.infimum(V::S, W::S) where {S <: SumSpace} = infimum(⊕(V), ⊕(W))
TensorKit.supremum(V::S, W::S) where {S <: SumSpace} = supremum(⊕(V), ⊕(W))
TensorKit.ominus(V::S, W::S) where {S <: SumSpace} = ominus(⊕(V), ⊕(W))

TensorKit.oplus(V::SumSpace{S}) where {S} = reduce(⊕, V.spaces; init = isdual(V) ? zerospace(S)' : zerospace(S))
TensorKit.oplus(V1::SumSpace{S}, V2::SumSpace{S}...) where {S} = mapreduce(⊕, ⊕, (V1, V2...))

function TensorKit.fuse(V1::S, V2::S) where {S <: SumSpace}
    return SumSpace(vec([fuse(v1, v2) for (v1, v2) in Base.product(V1.spaces, V2.spaces)]))
end

TensorKit.unitspace(S::Type{<:SumSpace}) = SumSpace(TensorKit.unitspace(eltype(S)))
TensorKit.zerospace(::Type{SumSpace{S}}) where {S} = SumSpace{S}()

function TensorKit.leftunitspace(S::SumSpace)
    return SumSpace(leftunitspace(oplus(S)))
end
function TensorKit.rightunitspace(S::SumSpace)
    return SumSpace(rightunitspace(oplus(S)))
end
TensorKit.isunitspace(S::SumSpace) = !isempty(S) && all(isunitspace, S.spaces)

# Promotion and conversion
# ------------------------
Base.promote_rule(::Type{S}, ::Type{SumSpace{S}}) where {S <: ElementarySpace} = SumSpace{S}
function Base.promote_rule(
        ::Type{S1}, ::Type{<:ProductSpace{S2}}
    ) where {S1 <: ElementarySpace, S2 <: ElementarySpace}
    return ProductSpace{promote_type(S1, S2)}
end
function Base.promote_rule(
        ::Type{<:ProductSpace{S1}}, ::Type{<:ProductSpace{S2}}
    ) where {S1 <: ElementarySpace, S2 <: ElementarySpace}
    return ProductSpace{promote_type(S1, S2)}
end
function Base.promote_rule(
        ::Type{<:TensorMapSumSpace{S}}, ::Type{<:TensorMapSpace{S}}
    ) where {S}
    return TensorMapSumSpace{S}
end

Base.convert(::Type{I}, S::SumSpace{I}) where {I <: ElementarySpace} = ⊕(S)
Base.convert(::Type{SumSpace{S}}, V::S) where {S <: ElementarySpace} = SumSpace(V)
function Base.convert(::Type{<:ProductSumSpace{S, N}}, V::ProductSpace{S, N}) where {S, N}
    return ProductSumSpace{S, N}(SumSpace.(V.spaces)...)
end
function Base.convert(::Type{<:ProductSumSpace{S}}, V::ProductSpace{S, N}) where {S, N}
    return ProductSumSpace{S, N}(SumSpace.(V.spaces)...)
end
function Base.convert(::Type{<:ProductSpace{S, N}}, V::ProductSumSpace{S, N}) where {S, N}
    return ProductSpace{S, N}(map(⊕, V.spaces)...)
end
function Base.convert(
        ::Type{<:TensorMapSumSpace{S}}, V::TensorMapSpace{S, N₁, N₂}
    ) where {S, N₁, N₂}
    return convert(ProductSumSpace{S, N₁}, codomain(V)) ←
        convert(ProductSumSpace{S, N₂}, domain(V))
end
function Base.convert(
        ::Type{<:TensorMapSpace{S, N₁, N₂}}, V::TensorMapSumSpace{S, N₁, N₂}
    ) where {S, N₁, N₂}
    return convert(ProductSpace{S, N₁}, codomain(V)) ← convert(ProductSpace{S, N₂}, domain(V))
end

# Show
# ----
# adapted from Base.show_vector
# https://github.com/JuliaLang/julia/blob/9af96508e9715e22154fc7b5a7283ad41d23765a/base/arrayshow.jl#L514
const SUMSPACE_SHOW_LIMIT = Ref(5)
function Base.show(io::IO, V::SumSpace)
    if length(V) == 1
        print(io, "⊞(")
        show(io, V[1])
        print(io, ")")
        return nothing
    end

    limited = get(io, :limited, true)
    ioc = IOContext(io, :compact => true)
    if limited && length(V) > SUMSPACE_SHOW_LIMIT[]
        ax = axes(V.spaces, 1)
        f, l = first(ax), last(ax)
        h = SUMSPACE_SHOW_LIMIT[] ÷ 2
        Base.show_delim_array(ioc, V.spaces, "(", " ⊞", "", false, f, f + h)
        print(io, " ⊞ ⋯ ⊞ ")
        Base.show_delim_array(ioc, V.spaces, "", " ⊞", ")", false, l - h, l)
    else
        Base.show_delim_array(ioc, V.spaces, "(", " ⊞", ")", false)
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", V::SumSpace)
    # print small summary, e.g.: l-element SumSpace(Vect[I](…)) of dim d
    l = length(V.spaces)
    d = dim(V)
    print(io, l, "-element ⊞(::", TK.type_repr(eltype(V)), "…)")
    isdual(V) && print(io, "'")
    print(io, " of dim ", d)

    compact = get(io, :compact, false)::Bool
    (iszero(d) || compact) && return nothing

    # print detailed space information - hijack Base.Vector printing
    print(io, ":\n")
    print_data = V.spaces
    ioc = IOContext(io, :typeinfo => eltype(print_data))
    Base.print_matrix(ioc, print_data)

    return nothing
end

# TensorMapSumSpace
# -----------------
# function TensorKit.fusionblockstructure(
#     W::TensorMapSumSpace{S,N₁,N₂}, cachestyle::TensorKit.CacheStyle
# ) where {S,N₁,N₂}
#     return TensorKit.fusionblockstructure(convert(TensorMapSpace{S,N₁,N₂}, W), cachestyle)
# end
