## `AbstractBlockTensorMap`

The main type of this package is `BlockTensorMap`, which is a generalization of `AbstractTensorMap` to the case where the tensor is a concatenation of other tensors.
The design philosophy is to have the same interface as `AbstractTensorMap`, with the additional ability to query the individual tensors that make up the block tensor, as `AbstractArray{AbstractTensorMap}`.

### Type hierarchy

The type hierarchy of the `AbstractBlockTensorMap` consists of one abstract and two concrete subtypes of `AbstractBlockTensorMap`:

```julia
BlockTensorMap <: AbstractBlockTensorMap <: AbstractTensorMap
SparseBlockTensorMap <: AbstractBlockTensorMap <: AbstractTensorMap
```

In particular, these structures hold the structural information as a `HomSpace` of `SumSpace`s, as defined in [`SumSpaces`](@ref sec_sumspaces), as well as the individual tensors that make up the block tensor.
For `BlockTensorMap`, the list of tensors is dense, thus they are stored in an `Array{AbstractTensorMap,N}`, where `N` is the total number of indices of a tensor.
For `SparseBlockTensorMap`, this is not the case, and the list of tensors is stored in a `Dict{CartesianIndex{N},AbstractTensorMap}`.

The elementary constructors for these types are:

```julia
BlockTensorMap{TT}(undef, space::TensorMapSumSpace)
SparseBlockTensorMap{TT}(undef, space::TensorMapSumSpace)
```

where `TT<:AbstractTensorMap` is the type of the individual tensors, and `space` is the `TensorMapSumSpace` that defines the structure of the block tensor.

!!! note

    In rare cases, `undef_blocks` can also be used, which won't allocate the component tensors.
    In these cases it is left up to the user to not access elements before they are allocated.

Similarly, they can be initialized from a list of tensors:

```julia
BlockTensorMap{TT}(tensors::AbstractArray{AbstractTensorMap,N}, space::TensorMapSumSpace)
SparseBlockTensorMap{TT}(tensors::Dict{CartesianIndex{N},AbstractTensorMap}, space::TensorMapSumSpace)
```

Typically though, the most convenient way of obtaining a block tensor is by using one of `zeros`, `rand` or `randn`, as well as their sparse counterparts `spzeros` or `sprand`.

```@repl blocktensors
using TensorKit, BlockTensorKit
V = ℂ^1 ⊞ ℂ^2;
W = V * V → V;
t = rand(W)
eltype(t)
s = sprand(W, 0.5)
eltype(s)
```

!!! note

    In analogy to `TensorKit`, most of the functionality that requires a `space` object can equally well be called in terms of `codomain(space), domain(space)`, if that is more convenient.

### Indexing

For indexing operators, `AbstractBlockTensorMap` behaves like an `AbstractArray{AbstractTensorMap}`, and the individual tensors can be accessed via the `getindex` and `setindex!` functions.
In particular, the `getindex` function returns a `TT` object, and the `setindex!` function expects a `TT` object.
Both linear and cartesian indexing styles are supported.

```@repl blocktensors
t[1] isa eltype(t)
t[1] == t[1, 1, 1]
t[2] = 3 * t[2]
s[1] isa eltype(t)
s[1] == s[1, 1, 1]
s[1] += 2 * s[1]
```

Slicing operations are also supported, and the `AbstractBlockTensorMap` can be sliced in the same way as an `AbstractArray{AbstractTensorMap}`.
There is however one elementary difference: as the slices still contain tensors with the same amount of legs, there can be no reduction in the number of dimensions.
In particular, in contrast to `AbstractArray`, scalar dimensions are not discarded, and as a result, linear index slicing is not allowed.
Repeated indices duplicate the selected tensors, as they would for an `AbstractArray`.

```@repl blocktensors
ndims(t[1, 1, :]) == 3
ndims(t[:, 1:2, [1, 1]]) == 3
t[1:2] # error
```

### VectorInterface.jl

As part of the `TensorKit` interface, `AbstractBlockTensorMap` also implements `VectorInterface`.
This means that you can efficiently add, scale, and compute the inner product of `AbstractBlockTensorMap` objects.

```@repl blocktensors
t1, t2 = rand!(similar(t)), rand!(similar(t));
add(t1, t2, rand())
scale(t1, rand())
inner(t1, t2)
```

For further in-place and possibly-in-place methods, see [`VectorInterface.jl`](https://github.com/Jutho/VectorInterface.jl)

### TensorOperations.jl

The `TensorOperations.jl` interface is also implemented for `AbstractBlockTensorMap`.
In particular, the `AbstractBlockTensorMap` can be contracted with other `AbstractBlockTensorMap` objects, as well as with `AbstractTensorMap` objects.
In order for that mix to work, the `AbstractTensorMap` objects are automatically converted to `AbstractBlockTensorMap` objects with a single tensor, i.e. the sum spaces will be a sum of one space.
As a consequence, as soon as one of the input tensors is blocked, the output tensor will also be blocked, even though its size might be trivial.
In these cases, `only` can be used to retrieve the single element in the `BlockTensorMap`.

```@repl blocktensors
@tensor t3[a; b] := t[a; c d] * conj(t[b; c d])
@tensor t4[a; b] := t[1, :, :][a; c d] * conj(t[1, :, :][b; c d]) # blocktensor * blocktensor = blocktensor
t4 isa AbstractBlockTensorMap
only(t4) isa eltype(t4)
@tensor t5[a; b] := t[1][a; c d] * conj(t[1:1, 1:1, 1:1][b; c d]) # tensor * blocktensor = blocktensor
t5 isa AbstractBlockTensorMap
only(t5) isa eltype(t5)
```

### Factorizations

Currently, there is only rudimentary support for factorizations of `AbstractBlockTensorMap` objects.
In particular, the implementations are not yet optimized for performance, and the factorizations are typically carried out by mapping to a dense tensor, and then performing the factorization on that tensor.

!!! note

    Most factorizations do not retain the additional imposed block structure. In particular, constructions of orthogonal bases will typically mix up the subspaces, and as such the resulting vector spaces will be `SumSpace`s of a single term.
