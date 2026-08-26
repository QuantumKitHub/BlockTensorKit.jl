# Slice indices
# -------------
# inverting index maps, to copy sliced blocks in `O(nnz)` instead of `O(nnz * length(dst))`

const SliceIndex = Union{Strided.SliceIndex, AbstractVector{<:Integer}}

_key_tuple(I::CartesianIndex) = I.I
_key_tuple(i::Integer) = (Int(i),)

"""
    _invert_index(n::Int, ind) -> m

Invert an index into a dimension of length `n`, such that `_dstrange(m, i)` yields all
destination coordinates selecting source coordinate `i`.
"""
_invert_index(::Int, i::Integer) = Int(i)
_invert_index(::Int, r::AbstractUnitRange{Int}) = r
function _invert_index(n::Int, ind)
    ptr = zeros(Int, n + 1)
    for i in ind
        1 ≤ i ≤ n || throw(BoundsError(Base.OneTo(n), i))
        ptr[i + 1] += 1
    end
    cumsum!(ptr, ptr)
    dsts = Vector{Int}(undef, length(ind))
    pos = copy(ptr)
    for (j, i) in enumerate(ind)
        dsts[pos[i] += 1] = j
    end
    return ptr, dsts
end

_dstrange(m::Int, i::Int) = i == m ? (1:1) : (1:0)
function _dstrange(m::AbstractUnitRange{Int}, i::Int)
    j = i - first(m) + 1
    return 1 ≤ j ≤ length(m) ? (j:j) : (1:0)
end
_dstrange((ptr, dsts)::Tuple{Vector{Int}, Vector{Int}}, i::Int) =
    view(dsts, (ptr[i] + 1):ptr[i + 1])

"""
    _copyslice!(tdst, tsrc, inds::NTuple{N,Any}) -> tdst

Copy the nonzero blocks of `tsrc` selected by `inds` into `tdst`, where `inds` holds one
normalized index per dimension of `tsrc`.
"""
function _copyslice!(tdst, tsrc, inds::NTuple{N, Any}) where {N}
    maps = map(_invert_index, size(tsrc), inds)
    for (I, v) in nonzero_pairs(tsrc)
        rs = map(_dstrange, maps, _key_tuple(I))
        any(isempty, rs) && continue
        for J in Iterators.product(rs...)
            tdst[J...] = v
        end
    end
    return tdst
end
