# Conversion
# ----------

function _copy_subblocks!(tdst, tsrc)
    N₁, N₂ = numout(tsrc), numin(tsrc)
    offsets = ntuple(i -> _sumspace_offsets(i <= N₁ ? codomain(tsrc)[i] : domain(tsrc)[i - N₁]), N₁ + N₂)
    # a single `SubblockIterator` holds the structure: `tdst[f₁, f₂]` would look it up in TensorKit's locked cache every time
    dstblocks = subblocks(tdst)
    for (k, v) in nonzero_pairs(tsrc), ((f₁, f₂), src) in subblocks(v)
        ranges = map(offsets, (f₁.uncoupled..., f₂.uncoupled...), Tuple(k)) do o, c, kᵢ
            return (o[c][kᵢ] + 1):o[c][kᵢ + 1]
        end
        copy!(dstblocks[(f₁, f₂)][ranges...], src)
    end
    return tdst
end

# for every sector `c` of `V`, the cumulative dimensions of `c` over the summands of `V`
function _sumspace_offsets(V)
    return SectorDict{sectortype(V), Vector{Int}}(c => cumsum(vcat(0, [dim(Vᵢ, c) for Vᵢ in V])) for c in sectors(V))
end

function Base.convert(::Type{TensorMap}, t::AbstractBlockTensorMap)
    S = spacetype(t)
    N₁, N₂ = numout(t), numin(t)
    cod = ProductSpace{S, N₁}(oplus.(codomain(t).spaces))
    dom = ProductSpace{S, N₂}(oplus.(domain(t).spaces))
    tdst = TensorKit.TensorMapWithStorage{scalartype(t), storagetype(t)}(undef, cod, dom)

    issparse(t) && zerovector!(tdst)
    _copy_subblocks!(tdst, t)
    return tdst
end

function Base.convert(::Type{TT}, t::AbstractBlockTensorMap) where {TT <: TensorKit.TensorMap}
    S = spacetype(t)
    N₁, N₂ = numout(t), numin(t)
    cod = ProductSpace{S, N₁}(oplus.(codomain(t).spaces))
    dom = ProductSpace{S, N₂}(oplus.(domain(t).spaces))
    tdst = TT(undef, cod ← dom)
    issparse(t) && zerovector!(tdst)

    _copy_subblocks!(tdst, t)
    return tdst
end

function Base.convert(::Type{TT}, t::AbstractTensorMap) where {TT <: AbstractBlockTensorMap}
    t isa TT && return t
    if t isa AbstractBlockTensorMap
        tdst = similar(TT, space(t))
        for (I, v) in nonzero_pairs(t)
            tdst[I] = v
        end
    else
        S = spacetype(t)
        tdst = TT(
            undef,
            convert(ProductSumSpace{S, numout(t)}, codomain(t)),
            convert(ProductSumSpace{S, numin(t)}, domain(t)),
        )
        tdst[1] = t
    end
    return tdst
end

TensorKit.TensorMap(t::AbstractBlockTensorMap) = convert(TensorMap, t)
