# Conversion
# ----------

function _subblock_ranges(offsets, (f₁, f₂), I::CartesianIndex)
    return map(offsets, (f₁.uncoupled..., f₂.uncoupled...), Tuple(I)) do o, c, kᵢ
        return (o[c][kᵢ] + 1):o[c][kᵢ + 1]
    end
end

function _subblock_pairs(t::AbstractTensorMap)
    sectortype(t) === Trivial || return subblocks(t)
    f = TK.trivial_fusiontree(t)
    return (f => subblock(t, f),)
end

function _copy_subblocks!(tdst::TensorMap, tsrc::AbstractBlockTensorMap)
    offsets = map(_sumspace_offsets, eachspace(tsrc).sumspaces)
    dst = Base.Fix1(_cachedsubblock, _subblockcache(tdst))
    for (I, v) in nonzero_pairs(tsrc), (f, src) in _subblock_pairs(v)
        copy!(view(dst(f), _subblock_ranges(offsets, f, I)...), src)
    end
    return tdst
end

function _copy_subblocks!(tdst::AbstractBlockTensorMap, tsrc::AbstractTensorMap)
    offsets = map(_sumspace_offsets, eachspace(tdst).sumspaces)
    src = Base.Fix1(_cachedsubblock, _subblockcache(tsrc))
    for (I, V) in pairs(eachspace(tdst))
        v = similar(eltype(tdst), V)
        vblocks = _subblock_pairs(v)
        issparse(tdst) && all(isempty ∘ last, vblocks) && continue
        for (f, dst) in vblocks
            copy!(dst, view(src(f), _subblock_ranges(offsets, f, I)...))
        end
        @inbounds tdst[I] = v
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
