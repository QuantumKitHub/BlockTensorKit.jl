const BlockTensorOrAdjoint = Union{AbstractBlockTensorMap, AdjointBlockTensorMap}

# permute!, braid!, tensoradd!, permute_twist!
# --------------------------------------------
# these all unwrap adjoints and then call `_braid!`

# `t'` itself isn't an `AdjointTensorMap`, but its blocks are, so
# unwrap just `tsrc` due to `twistmask` referring to the legs of `tdst`
@propagate_inbounds function _braid_block!(
        tdst, tsrc, p, conjsrc, levels, twistmask, α, β, backend, allocator
    )
    tsrc′, p′, levels′, conjsrc′ = TK._unwrap_source(tsrc, p, levels, conjsrc)
    return TK._braid!(tdst, tsrc′, p′, conjsrc′, levels′, twistmask, α, β, backend, allocator)
end

@propagate_inbounds function TK._braid!(
        tdst::BlockTensorMap, tsrc::BlockTensorMap, p::Index2Tuple, conjsrc::Bool,
        levels::IndexTuple, twistmask::Tuple{Vararg{Bool}}, α, β, backend, allocator
    )
    @boundscheck TK.spacecheck_transform(TK.permute, space(tdst), space(tsrc), p, conjsrc)
    dstdata = parent(tdst)
    srcdata = permutedims(StridedView(parent(tsrc)), (p[1]..., p[2]...))
    @inbounds for I in eachindex(dstdata, srcdata)
        dstdata[I] = _braid_block!(
            dstdata[I], srcdata[I], p, conjsrc, levels, twistmask, α, β, backend, allocator
        )
    end
    return tdst
end
@propagate_inbounds function TK._braid!(
        tdst::AbstractBlockTensorMap, tsrc::AbstractBlockTensorMap, p::Index2Tuple, conjsrc::Bool,
        levels::IndexTuple, twistmask::Tuple{Vararg{Bool}}, α, β, backend, allocator
    )
    @boundscheck TK.spacecheck_transform(TK.permute, space(tdst), space(tsrc), p, conjsrc)
    p_lin = (p[1]..., p[2]...)
    _scale_untouched!(tdst, tsrc, β, p_lin)
    @inbounds for (I, v) in nonzero_pairs(tsrc)
        I′ = CartesianIndex(TT.getindices(I.I, p_lin))
        _addblock!(tdst, I′, β, allocator) do c, γ
            _braid_block!(c, v, p, conjsrc, levels, twistmask, α, γ, backend, allocator)
        end
    end
    return tdst
end
# a block tensor holding a single block is interchangeable with that block
@propagate_inbounds function TK._braid!(
        tdst::TensorMap, tsrc::BlockTensorMap, p::Index2Tuple, conjsrc::Bool,
        levels::IndexTuple, twistmask::Tuple{Vararg{Bool}}, α, β, backend, allocator
    )
    return _braid_block!(tdst, only(tsrc), p, conjsrc, levels, twistmask, α, β, backend, allocator)
end
@propagate_inbounds function TK._braid!(
        tdst::BlockTensorMap, tsrc::TensorMap, p::Index2Tuple, conjsrc::Bool,
        levels::IndexTuple, twistmask::Tuple{Vararg{Bool}}, α, β, backend, allocator
    )
    TK._braid!(only(tdst), tsrc, p, conjsrc, levels, twistmask, α, β, backend, allocator)
    return tdst
end

# transpose!
# ----------
function TK.transpose!(
        tdst::BlockTensorMap, tsrc::BlockTensorMap, p::Index2Tuple,
        α::Number, β::Number, backend::AbstractBackend, allocator
    )
    @boundscheck TK.spacecheck_transform(TK.transpose, space(tdst), space(tsrc), p)

    dstdata = parent(tdst)
    srcdata = permutedims(StridedView(parent(tsrc)), (p[1]..., p[2]...))

    @inbounds for I in eachindex(dstdata, srcdata)
        dstdata[I] = TK.transpose!(dstdata[I], srcdata[I], p, α, β, backend, allocator)
    end
    return tdst
end
function TK.transpose!(
        tdst::AbstractBlockTensorMap, tsrc::BlockTensorOrAdjoint,
        p::Index2Tuple, α::Number, β::Number, backend::AbstractBackend, allocator
    )
    @boundscheck TK.spacecheck_transform(TK.transpose, space(tdst), space(tsrc), p)
    p_lin = (p[1]..., p[2]...)
    _scale_untouched!(tdst, tsrc, β, p_lin)
    @inbounds for (I, v) in nonzero_pairs(tsrc)
        I′ = CartesianIndex(TT.getindices(I.I, p_lin))
        _addblock!(tdst, I′, β, allocator) do c, γ
            TK.transpose!(c, v, p, α, γ, backend, allocator)
        end
    end
    return tdst
end
function TK.transpose!(
        tdst::TensorMap, tsrc::BlockTensorMap, p::Index2Tuple,
        α::Number, β::Number, backend::AbstractBackend, allocator
    )
    return TK.transpose!(tdst, only(tsrc), p, α, β, backend, allocator)
end
function TK.transpose!(
        tdst::BlockTensorMap, tsrc::TensorMap, p::Index2Tuple,
        α::Number, β::Number, backend::AbstractBackend, allocator
    )
    TK.transpose!(only(tdst), tsrc, p, α, β, backend, allocator)
    return tdst
end

Base.@constprop :aggressive function TK.insertleftunit(
        t::AbstractBlockTensorMap, i::Int = numind(t) + 1; kwargs...
    )
    W = TK.insertleftunit(space(t), i; kwargs...)
    tdst = similar(t, W)
    for (I, v) in nonzero_pairs(t)
        I′ = CartesianIndex(TT.insertafter(I.I, i - 1, (1,)))
        tdst[I′] = TK.insertleftunit(v, i; kwargs...)
    end
    return tdst
end

Base.@constprop :aggressive function TK.insertrightunit(
        t::AbstractBlockTensorMap, i::Int = numind(t) + 1; kwargs...
    )
    W = TK.insertrightunit(space(t), i; kwargs...)
    tdst = similar(t, W)
    for (I, v) in nonzero_pairs(t)
        I′ = CartesianIndex(TT.insertafter(I.I, i, (1,)))
        tdst[I′] = TK.insertrightunit(v, i; kwargs...)
    end
    return tdst
end

Base.@constprop :aggressive function TK.removeunit(
        t::AbstractBlockTensorMap, i::Int; kwargs...
    )
    W = TK.removeunit(space(t), i)
    tdst = similar(t, W)
    for (I, v) in nonzero_pairs(t)
        I′ = CartesianIndex(TT.deleteat(I.I, i))
        tdst[I′] = TK.removeunit(v, i)
    end
    return tdst
end

function TK.twist!(t::AbstractBlockTensorMap, is; inv::Bool = false)
    foreach(x -> twist!(x, is; inv), nonzero_values(t))
    return t
end
