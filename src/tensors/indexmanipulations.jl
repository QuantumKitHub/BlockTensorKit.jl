@propagate_inbounds function TK.add_transform!(
        tdst::BlockTensorMap, tsrc::BlockTensorMap, p::Index2Tuple, transformer,
        α::Number, β::Number, backend, allocator
    )
    @boundscheck TK.spacecheck_transform(permute, tdst, tsrc, p)

    dstdata = parent(tdst)
    srcdata = permutedims(StridedView(parent(tsrc)), (p[1]..., p[2]...))

    @inbounds for I in eachindex(dstdata, srcdata)
        dstdata[I] = TK.add_transform!(
            dstdata[I], srcdata[I], p, transformer, α, β, backend, allocator
        )
    end
    return tdst
end
@propagate_inbounds function TK.add_transform!(
        tdst::AbstractBlockTensorMap, tsrc::AbstractBlockTensorMap,
        p::Index2Tuple, transformer, α::Number, β::Number, backend, allocator
    )
    @boundscheck TK.spacecheck_transform(permute, tdst, tsrc, p)
    scale!(tdst, β)
    p_lin = (p[1]..., p[2]...)
    @inbounds for (I, v) in nonzero_pairs(tsrc)
        I′ = CartesianIndex(TT.getindices(I.I, p_lin))
        tdst[I′] = TK.add_transform!(
            tdst[I′], v, p_lin, transformer, α, One(), backend, allocator
        )
    end
    return tdst
end
function TK.add_transform!(
        tdst::AbstractBlockTensorMap, tsrc::AdjointTensorMap{T, S, N₁, N₂, TT},
        p::Index2Tuple, transformer, α::Number, β::Number, backend, allocator
    ) where {T, S, N₁, N₂, TT <: AbstractBlockTensorMap}
    @boundscheck TK.spacecheck_transform(permute, tdst, tsrc, p)
    scale!(tdst, β)
    p_lin = (p[1]..., p[2]...)
    @inbounds for (I, v) in nonzero_pairs(tsrc)
        I′ = CartesianIndex(TT.getindices(I.I, p_lin))
        tdst[I′] = TK.add_transform!(
            tdst[I′], v, p, transformer, α, One(), backend, allocator
        )
    end
    return tdst
end
function TK.add_transform!(
        tdst::TensorMap, tsrc::BlockTensorMap, p::Index2Tuple, transformer,
        α::Number, β::Number, backend, allocator
    )
    return TK.add_transform!(
        tdst, only(tsrc), p, transformer, α, β, backend, allocator
    )
end
function TK.add_transform!(
        tdst::BlockTensorMap, tsrc::TensorMap, p::Index2Tuple, transformer,
        α::Number, β::Number, backend, allocator
    )
    return TK.add_transform!(
        only(tdst), tsrc, p, transformer, α, β, backend, allocator
    )
end

# we need to capture the other functions earlier to enjoy the fast transformers...
for f in (:permute, :transpose)
    f! = Symbol(f, :!)
    @eval begin
        function TK.$f!(
                tdst::BlockTensorMap, tsrc::BlockTensorMap, p::Index2Tuple,
                α::Number, β::Number, backend::AbstractBackend, allocator
            )
            @boundscheck TK.spacecheck_transform(TK.$f, tdst, tsrc, p)

            dstdata = parent(tdst)
            srcdata = permutedims(StridedView(parent(tsrc)), (p[1]..., p[2]...))

            @inbounds for I in eachindex(dstdata, srcdata)
                dstdata[I] = TK.$f!(dstdata[I], srcdata[I], p, α, β, backend, allocator)
            end
            return tdst
        end
        function TK.$f!(
                tdst::AbstractBlockTensorMap, tsrc::AbstractBlockTensorMap,
                p::Index2Tuple, α::Number, β::Number, backend::AbstractBackend, allocator
            )
            @boundscheck TK.spacecheck_transform(TK.$f, tdst, tsrc, p)
            scale!(tdst, β)
            p_lin = (p[1]..., p[2]...)
            @inbounds for (I, v) in nonzero_pairs(tsrc)
                I′ = CartesianIndex(TT.getindices(I.I, p_lin))
                tdst[I′] = TK.$f!(tdst[I′], v, p, α, One(), backend, allocator)
            end
            return tdst
        end
        function TK.$f!(
                tdst::AbstractBlockTensorMap, tsrc::AdjointTensorMap{T, S, N₁, N₂, TT},
                p::Index2Tuple, α::Number, β::Number, backend::AbstractBackend, allocator
            ) where {T, S, N₁, N₂, TT <: AbstractBlockTensorMap}
            @boundscheck TK.spacecheck_transform(TK.$f, tdst, tsrc, p)
            scale!(tdst, β)
            p_lin = (p[1]..., p[2]...)
            @inbounds for (I, v) in nonzero_pairs(tsrc)
                I′ = CartesianIndex(TT.getindices(I.I, p))
                tdst[I′] = TK.$f!(tdst[I′], v, (p₁, p₂), α, One(), backend, allocator)
            end
            return tdst
        end
        function TK.$f!(
                tdst::TensorMap, tsrc::BlockTensorMap, p::Index2Tuple,
                α::Number, β::Number, backend::AbstractBackend, allocator
            )
            return TK.$f!(tdst, only(tsrc), p, α, β, backend, allocator)
        end
        function TK.$f!(
                tdst::BlockTensorMap, tsrc::TensorMap, p::Index2Tuple,
                α::Number, β::Number, backend::AbstractBackend, allocator
            )
            TK.$f!(only(tdst), tsrc, p, α, β, backend, allocator)
            return tdst
        end
    end
end

@propagate_inbounds function TK.braid!(
        tdst::BlockTensorMap, tsrc::BlockTensorMap, p::Index2Tuple, levels::IndexTuple,
        α::Number, β::Number, backend::AbstractBackend, allocator
    )
    @boundscheck TK.spacecheck_transform(braid, tdst, tsrc, p, levels)

    dstdata = parent(tdst)
    srcdata = permutedims(StridedView(parent(tsrc)), (p[1]..., p[2]...))

    @inbounds for I in eachindex(dstdata, srcdata)
        dstdata[I] = TK.braid!(
            dstdata[I], srcdata[I], p, levels, α, β, backend, allocator
        )
    end
    return tdst
end
@propagate_inbounds function TK.braid!(
        tdst::AbstractBlockTensorMap, tsrc::AbstractBlockTensorMap,
        p::Index2Tuple, levels::IndexTuple,
        α::Number, β::Number, backend::AbstractBackend, allocator
    )
    @boundscheck TK.spacecheck_transform(braid, tdst, tsrc, p, levels)
    scale!(tdst, β)
    p_lin = (p[1]..., p[2]...)
    @inbounds for (I, v) in nonzero_pairs(tsrc)
        I′ = CartesianIndex(TT.getindices(I.I, p_lin))
        tdst[I′] = TK.braid!(tdst[I′], v, p, levels, α, One(), backend, allocator)
    end
    return tdst
end
function TK.braid!(
        tdst::AbstractBlockTensorMap, tsrc::AdjointTensorMap{T, S, N₁, N₂, TT},
        p::Index2Tuple, levels::IndexTuple, α::Number, β::Number, backend::AbstractBackend, allocator
    ) where {T, S, N₁, N₂, TT <: AbstractBlockTensorMap}
    @boundscheck TK.spacecheck_transform(braid, tdst, tsrc, p, levels)
    scale!(tdst, β)
    p_lin = (p[1]..., p[2]...)
    @inbounds for (I, v) in nonzero_pairs(tsrc)
        I′ = CartesianIndex(TT.getindices(I.I, p_lin))
        tdst[I′] = TK.braid!(tdst[I′], v, p, levels, α, One(), backend, allocator)
    end
    return tdst
end
function TK.braid!(
        tdst::TensorMap, tsrc::BlockTensorMap,
        p::Index2Tuple, levels::IndexTuple,
        α::Number, β::Number, backend::AbstractBackend, allocator
    )
    return TK.add_braid!(tdst, only(tsrc), p, levels, α, β, backend, allocator)
end
function TK.braid!(
        tdst::BlockTensorMap, tsrc::TensorMap,
        p::Index2Tuple, levels::IndexTuple,
        α::Number, β::Number, backend::AbstractBackend, allocator
    )
    TK.braid!(only(tdst), tsrc, p, levels, α, β, backend, allocator)
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
