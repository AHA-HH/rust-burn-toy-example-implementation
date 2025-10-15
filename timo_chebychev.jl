

module Chebychev

using LinearAlgebra

import SIMD: Vec

"""Return the n Chebychev points on the interval [-1, 1]."""
function cheb_points(n::Int64)
    return [cos((j * pi / (n - 1))) for j in 0:n-1]
end

""" Return the n Chebychev weights for barycentric interpolation."""
function cheb_weights(n::Int64)
    weights = zeros(n)
    weights[1] = weights[n] = 0.5
    weights[2:n-1] .= 1.0
    return weights .* (-1.0) .^ (1:n)
end


function evaluate1d_impl(eval_points, values, cheb_points, weights)

    npoints = length(eval_points)
    n = length(values)

    @assert typeof(eval_points) <: AbstractVector
    @assert typeof(values) <: AbstractVector
    @assert typeof(cheb_points) <: AbstractVector
    @assert typeof(weights) <: AbstractVector
    @assert length(cheb_points) == n
    @assert length(weights) == n

    S = eltype(values)
    R = eltype(values)

    if S <: Vec
        R = eltype(S)
    end

    # If the elements of `values` contain a SIMD vector, we want the actual
    # underlying real type.

    @assert R <: Real
    @assert eltype(cheb_points) == R
    @assert eltype(weights) == R
    @assert eltype(eval_points) == R



    numerator = zeros(S, npoints)
    denominator = zeros(S, npoints)
    require_exact = zeros(Int64, npoints)

    for (index, cheb_point) in enumerate(cheb_points)
        @inbounds for i in 1:npoints
            diff = eval_points[i] - cheb_point
            diff_inv = inv(diff)
            if iszero(diff)
                require_exact[i] = index
            end
            numerator[i] += values[index] * (weights[index] * diff_inv)
            denominator[i] += weights[index] * diff_inv
        end
    end

    result = numerator ./ denominator

    @inbounds for index in 1:npoints
        if require_exact[index] > 0
            result[index] = values[require_exact[index]]
        end
    end

    result

end


"""Evaluate a 1d Chebychev interpolant."""
function evaluate1d(eval_points, values)

    @assert typeof(eval_points) <: AbstractVector
    @assert typeof(values) <: AbstractVector
    @assert length(eval_points) > 0
    @assert length(values) > 0

    n = length(values)

    # Now build the SIMD vector

    cheb_points = Chebychev.cheb_points(n)
    weights = Chebychev.cheb_weights(n)
    return evaluate1d_impl(eval_points, values, cheb_points, weights)


end



"""
   evaluate2d(eval_x, eval_y, values)

Evaluate aq 2d chebychev interpolant on a 2d tensor grid.

# Arguments
- `eval_x::AbstractVector{S}`: The x-coordinates where the interpolant is to be evaluated.
- `eval_y::AbstractVector{S}`: The y-coordinates where the interpolant is to be evaluated.
- `values::AbstractMatrix{S}`: The values of the interpolant at the Chebychev points. The first axis is the x dimension.
   The second axis is the y dimension.
"""
function evaluate2d_tensor(eval_x, eval_y, values)

    # The first axis is the x-dimension
    # So m is the number of Chebychev points in x direction
    m = size(values, 1)
    # n is the number of Chebychev points in the y direction.
    n = size(values, 2)

    npx = length(eval_x)
    npy = length(eval_y)

    S = eltype(values)

    cheb_points_m = Chebychev.cheb_points(m)
    cheb_points_n = Chebychev.cheb_points(n)
    weights_m = Chebychev.cheb_weights(m)
    weights_n = Chebychev.cheb_weights(n)

    res1 = zeros(S, m, npy)

    # For each x evaluate across the y direction. 
    for j in 1:m
        res1[j, :] = Chebychev.evaluate1d_impl(eval_y, values[j, :], cheb_points_n, weights_n)
    end

    res2 = zeros(S, npx, npy)

    # For each y point now evaluate across the x direction.

    for j in 1:npy
        res2[:, j] = Chebychev.evaluate1d_impl(eval_x, res1[:, j], cheb_points_m, weights_m)
    end

    return res2
end

"""
   evaluate3d(eval_x, eval_y, values)

# Evaluate a 3d chebychev interpolant on a 3d tensor grid.

# Arguments
# - `eval_x::AbstractVector{S}`: The x-coordinates where the interpolant is to be evaluated.
# - `eval_y::AbstractVector{S}`: The y-coordinates where the interpolant is to be evaluated.
# - `eval_z::AbstractVector{S}`: The z-coordinates where the interpolant is to be evaluated.
# - `values::AbstractArray{S, 3}`: The values of the interpolant at the Chebychev points. The first axis is the x dimension.
#    The second axis is the y dimension. The third is the z dimension.
# """
function evaluate3d_tensor_impl(eval_x, eval_y, eval_z, values)

    # The first axis is the z-dimension
    # So m is the number of Chebychev points in x direction
    m = size(values, 1)
    # n is the number of Chebychev points in the y direction.
    n = size(values, 2)
    # p is the number of Chebychev points in the z direction.
    p = size(values, 3)

    npx = length(eval_x)
    npy = length(eval_y)
    npz = length(eval_z)

    S = eltype(values)


    cheb_points_m = Chebychev.cheb_points(m)
    cheb_points_n = Chebychev.cheb_points(n)
    weights_m = Chebychev.cheb_weights(m)
    weights_n = Chebychev.cheb_weights(n)
    cheb_points_p = Chebychev.cheb_points(p)
    weights_p = Chebychev.cheb_weights(p)

    res1 = zeros(S, m, n, npz)

    # For each x and y evaluate across the z direction. 
    for i in 1:m
        for j in 1:n
            res1[i, j, :] = Chebychev.evaluate1d_impl(eval_z, values[i, j, :], cheb_points_p, weights_p)
        end
    end

    res2 = zeros(S, m, npy, npz)

    # Now evaluate across y

    for i in 1:m
        for j in 1:npz
            res2[i, :, j] = Chebychev.evaluate1d_impl(eval_y, res1[i, :, j], cheb_points_n, weights_n)
        end
    end

    # We have now done the y, z plane for each x point. Now evaluate for each of those points along
    # the x direction.

    res3 = zeros(S, npx, npy, npz)

    for i in 1:npy
        for j in 1:npz
            res3[:, i, j] = Chebychev.evaluate1d_impl(eval_x, res2[:, i, j], cheb_points_m, weights_m)
        end
    end

    return res3
end

function evaluate3d_tensor_1x(eval_x::AbstractVector{S}, eval_y::AbstractVector{S}, eval_z::AbstractVector{S}, values::AbstractArray{S,3}) where {S<:Real}

    return evaluate3d_tensor_impl(eval_x, eval_y, eval_z, values)

end


function evaluate3d_tensor_2x(eval_x::AbstractVector{S}, eval_y::AbstractVector{S}, eval_z::AbstractVector{S}, values1::AbstractArray{S,3}, values2::AbstractArray{S,3}) where {S<:Real}

    @assert size(values1) == size(values2)
    simd_values = zeros(Vec{2,S}, size(values1, 1), size(values1, 2), size(values1, 3))
    for i in axes(values1, 1)
        for j in axes(values1, 2)
            for k in axes(values1, 3)
                simd_values[i, j, k] = Vec{2,S}((values1[i, j, k], values2[i, j, k]))
            end
        end
    end

    simd_result = evaluate3d_tensor_impl(eval_x, eval_y, eval_z, simd_values)

    return (getindex.(simd_result, 1), getindex.(simd_result, 2))

end

function evaluate3d_tensor_4x(eval_x::AbstractVector{S}, eval_y::AbstractVector{S}, eval_z::AbstractVector{S}, values1::AbstractArray{S,3}, values2::AbstractArray{S,3}, value3::AbstractArray{S,3}, value4::AbstractArray{S,3}) where {S<:Real}

    @assert size(values1) == size(values2) == size(value3) == size(value4)
    simd_values = zeros(Vec{4,S}, size(values1, 1), size(values1, 2), size(values1, 3))
    for i in axes(values1, 1)
        for j in axes(values1, 2)
            for k in axes(values1, 3)
                simd_values[i, j, k] = Vec{4,S}((values1[i, j, k], values2[i, j, k], values3[i, j, k], values4[i, j, k]))
            end
        end
    end

    simd_result = evaluate3d_tensor_impl(eval_x, eval_y, eval_z, simd_values)


    return (getindex.(simd_result, 1), getindex.(simd_result, 2), getindex.(simd_result, 3), getindex.(simd_result, 4))

end

function evaluate3d_tensor_8x(eval_x::AbstractVector{S}, eval_y::AbstractVector{S}, eval_z::AbstractVector{S},
    values1::AbstractArray{S,3}, values2::AbstractArray{S,3}, value3::AbstractArray{S,3}, value4::AbstractArray{S,3}, value5::AbstractArray{S,3}, value6::AbstractArray{S,3}, value7::AbstractArray{S,3}, value8::AbstractArray{S,3}) where {S<:Real}

    @assert size(values1) == size(values2) == size(value3) == size(value4) == size(value5) == size(value6) == size(value7) == size(value8)
    simd_values = zeros(Vec{8,S}, size(values1, 1), size(values1, 2), size(values1, 3))
    for i in axes(values1, 1)
        for j in axes(values1, 2)
            for k in axes(values1, 3)
                simd_values[i, j, k] = Vec{8,S}((values1[i, j, k], values2[i, j, k], values3[i, j, k], values4[i, j, k], values5[i, j, k], values6[i, j, k], values7[i, j, k], values8[i, j, k]))
            end
        end
    end

    simd_result = evaluate3d_tensor_impl(eval_x, eval_y, eval_z, simd_values)
    return (getindex.(simd_result, 1), getindex.(simd_result, 2), getindex.(simd_result, 3), getindex.(simd_result, 4), getindex.(simd_result, 5), getindex.(simd_result, 6), getindex.(simd_result, 7), getindex.(simd_result, 8))

end

function evaluate3d_tensor_1x(eval_x::AbstractVector{S}, eval_y::AbstractVector{S}, eval_z::AbstractVector{S}, values::AbstractArray{S,3}) where {S<:Complex}

    return Complex.(evaluate3d_tensor_1x(eval_x, eval_y, eval_z, real(values)), evaluate3d_tensor_1x(eval_x, eval_y, eval_z, imag(values)))

end

function evaluate3d_tensor_2x(eval_x::AbstractVector{S}, eval_y::AbstractVector{S}, eval_z::AbstractVector{S}, values1::AbstractArray{S,3}, values2::AbstractArray{S,3}) where {S<:Complex}

    return (Complex.(evaluate3d_tensor_2x(eval_x, eval_y, eval_z, real(values1), real(values2))), Complex.(evaluate3d_tensor_2x(eval_x, eval_y, eval_z, imag(values1), imag(values2))))

end

function evaluate3d_tensor_4x(eval_x::AbstractVector{S}, eval_y::AbstractVector{S}, eval_z::AbstractVector{S}, values1::AbstractArray{S,3}, values2::AbstractArray{S,3}, value3::AbstractArray{S,3}, value4::AbstractArray{S,3}) where {S<:Complex}

    return (Complex.(evaluate3d_tensor_4x(eval_x, eval_y, eval_z, real(values1), real(values2), real(value3), real(value4))), Complex.(evaluate3d_tensor_4x(eval_x, eval_y, eval_z, imag(values1), imag(values2), imag(value3), imag(value4))))

end

function evaluate3d_tensor_8x(eval_x::AbstractVector{S}, eval_y::AbstractVector{S}, eval_z::AbstractVector{S},
    values1::AbstractArray{S,3}, values2::AbstractArray{S,3}, value3::AbstractArray{S,3}, value4::AbstractArray{S,3}, value5::AbstractArray{S,3}, value6::AbstractArray{S,3}, value7::AbstractArray{S,3}, value8::AbstractArray{S,3}) where {S<:Complex}

    return (Complex.(evaluate3d_tensor_8x(eval_x, eval_y, eval_z, real(values1), real(values2), real(value3), real(value4), real(value5), real(value6), real(value7), real(value8))), Complex.(evaluate3d_tensor_8x(eval_x, eval_y, eval_z, imag(values1), imag(values2), imag(value3), imag(value4), imag(value5), imag(value6), imag(value7), imag(value8))))

end



function eval_1d_single_point_impl(x, values, cheb_points, weights)

    n = length(values)

    S = eltype(values)

    numerator::S = 0.0
    denominator::S = 0.0
    exact_index::Int64 = 0

    @inbounds for (index, cheb_point) in enumerate(cheb_points)
        diff = x - cheb_point
        if iszero(diff)
            exact_index = index
        end
        diff_inv = inv(diff)
        numerator += values[index] * (weights[index] * diff_inv)
        denominator += weights[index] * diff_inv
    end

    result = numerator / denominator

    if exact_index > 0
        result = values[exact_index]
    end

    return result
end

function evaluate3d_single_point_impl(x, y, z, values, cheb_points_x, cheb_points_y, cheb_points_z, weights_x, weights_y, weights_z)

    # We now do the 3d evaluation.

    # The first axis is the z-dimension
    # So m is the number of Chebychev points in x direction
    m = size(values, 1)
    # n is the number of Chebychev points in the y direction.
    n = size(values, 2)
    # p is the number of Chebychev points in the z direction.
    p = size(values, 3)

    S = typeof(x)

    res1 = zeros(S, m, n)

    # Evaluate for z
    for i in 1:m
        for j in 1:n
            res1[i, j] = Chebychev.eval_1d_single_point_impl(z, values[i, j, :], cheb_points_z, weights_z)
        end
    end

    res2 = zeros(S, m)

    # For each z point now evaluate across all x along the y direction.

    for i in 1:m
        res2[i] = Chebychev.eval_1d_single_point_impl(y, res1[i, :], cheb_points_y, weights_y)
    end

    # Finally, we evaluate for z.

    Chebychev.eval_1d_single_point_impl(x, res2, cheb_points_x, weights_x)

end

function evaluate3d(eval_points::AbstractArray{S,2}, values::AbstractArray{S,3})::Vector{S} where {S<:Real}

    # The eval_points should have first dimension 3

    @assert size(eval_points, 1) == 3

    # The values should have 3 dimensions.
    @assert ndims(values) == 3

    m = size(values, 1)
    n = size(values, 2)
    p = size(values, 3)

    # Create Chebychev points and values for each dimension.

    cheb_points_x = Chebychev.cheb_points(m)
    cheb_points_y = Chebychev.cheb_points(n)
    cheb_points_z = Chebychev.cheb_points(p)
    weights_x = Chebychev.cheb_weights(m)
    weights_y = Chebychev.cheb_weights(n)
    weights_z = Chebychev.cheb_weights(p)

    result = zeros(S, size(eval_points, 2))

    for index in eachindex(result)
        (x, y, z) = eval_points[:, index]
        result[index] = Chebychev.evaluate3d_single_point_impl(x, y, z, values, cheb_points_x, cheb_points_y, cheb_points_z, weights_x, weights_y, weights_z)
    end


    result

end

end
