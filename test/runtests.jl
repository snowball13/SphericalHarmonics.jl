# Testing

using Test
using BlockArrays
using BlockBandedMatrices
using SparseArrays
using StaticArrays
using LinearAlgebra
using Revise
using ApproxFun
using SphericalHarmonics

import SphericalHarmonics: jacobiz, evaluate, laplacianoperator, resizedata!,
                            gradientoperator, divergenceoperator,
                            unitcrossproductoperator, getSHspace,
                            gettangentspace, getopindex, getnormalisingconstant,
                            jacobix, jacobiy, jacobiz



# Setup
rhoval(S::SphericalHarmonicsSpace, z) = sqrt(1 - z^2)
isindomain(FAM::SphericalHarmonicsFamily, pt) = pt[1]^2 + pt[2]^2 + pt[3]^2 ≈ 1.0
T = ComplexF64; R = Float64
FAM = SphericalHarmonicsFamily(T, R)
S = FAM()
ST = gettangentspace(S)
z = 0.3; y = 0.45; x = sqrt(1 - y^2 - z^2); pt = [x; y; z]; isindomain(FAM, pt)
ϕ = acos(z); θ = atan(y / x)




function laplaciantest_phi(S::SphericalHarmonicsSpace, inds, pt)
    """ Returns (1/ρ)∂ϕ(ρ∂ϕ(f)) evaluated at the point pt

    Assumes f = x^a * y^b * z^c  where inds = [a;b;c]
    """

    x, y, z = pt
    a, b, c = inds
    ret = (a+b)^2 * z^4 / (1 - z^2)
    ret -= ((a+b) * (c + 1) + c * (a+b+2)) * z^2
    ret += c * (c-1) * (1 - z^2)
    ret *= x^a * y^b * z^(c-2)
end
function laplaciantest_theta(S::SphericalHarmonicsSpace, inds, pt)
    """ Returns (1/ρ^2)∂θ²(f) evaluated at the point pt

    Assumes f = x^a * y^b * z^c  where inds = [a;b;c]
    """

    x, y, z = pt
    a, b, c = inds
    ret = b * (b-1) * x^4
    ret -= (2a * b + a + b) * x^2 * y^2
    ret += a * (a-1) * y^4
    ret *= x^(a-2) * y^(b-2) * z^c / (1 - z^2)
end
function laplaciantest(S::SphericalHarmonicsSpace, inds, pt)
    """ Returns Δ_s(f) evaluated at the point pt

    Assumes f = x^a * y^b * z^c  where inds = [a;b;c]
    """

    @assert (length(inds) == 3 && inds[1] > 1 && inds[2] > 1 && inds[3] > 1) "Indicies need to be ≥ 2"
    ret = laplaciantest_phi(S, inds, pt)
    ret += laplaciantest_theta(S, inds, pt)
    ret
end


#---

@testset "Evaluation" begin
    FAM = SphericalHarmonicsFamily(T, R)
    S = FAM()
    N = 20
    resizedata!(S, N+1)
    # Real coeffs
    for l = 0:N, m = -l:l
        absm = abs(m)
        # P = Fun(FAM.P(Float64(absm), Float64(absm)), [zeros(l-absm); 1]); P.coefficients
        P = Fun(Jacobi(absm, absm), [zeros(l-absm); 1]); P.coefficients
        cfs = zeros(getopindex(S, l, l)); cfs[getopindex(S, l, m)] = 1; cfs
        sh = Fun(S, cfs)
        shactual = getnormalisingconstant(R, S, l, m) * P(pt[3]) * rhoval(S, pt[3])^absm * exp(im * m * θ)
        @test sh(pt) ≈ shactual
    end
    # Imaginary coeffs
    for l = 0:N, m = -l:l
        absm = abs(m)
        # P = Fun(FAM.P(Float64(absm), Float64(absm)), [zeros(l-absm); 1]); P.coefficients
        P = Fun(Jacobi(absm, absm), [zeros(l-absm); 1]); P.coefficients
        cfs = zeros(T, getopindex(S, l, l)); cfs[getopindex(S, l, m)] = 1im; cfs
        sh = Fun(S, cfs)
        shactual = im * getnormalisingconstant(R, S, l, m) * P(pt[3]) * rhoval(S, pt[3])^absm * exp(im * m * θ)
        @test sh(pt) ≈ shactual
    end
end

@testset "Transform" begin
    FAM = SphericalHarmonicsFamily(T, R)
    S = FAM()

    # specified function
    N = 20
    n = 2(N+1)^2
    f = (x,y,z) -> exp(x)*y*z
    pts = points(S, n)
    vals = [f(p...) for p in pts]
    cfs = transform(S, vals)
    F = Fun(S, cfs)
    @test F(pt) ≈ f(pt...)

    # random polynomial
    inds = [4; 5; 10]
    N = sum(inds)
    n = 2(N+1)^2
    f = (x,y,z) -> x^inds[1] * y^inds[2] * z^inds[3]
    pts = points(S, n)
    vals = [f(p...) for p in pts]
    cfs = transform(S, vals)
    F = Fun(S, cfs)
    @test F(pt) ≈ f(pt...)
    # @test vals ≈ itransform(S, cfs)
end

@testset "Declaring a function" begin
    FAM = SphericalHarmonicsFamily(T, R)
    S = FAM()
    f = (x,y,z) -> exp(x)*y*z
    N = 20
    n = 2(N+1)^2
    F = Fun(f, S, n) # NOTE we need to give the number of coeffs we want (x2)
    @test F(pt) ≈ f(pt...)
end

@testset "Jacobi matrices" begin
    # TODO make the jacobi mat methods
    FAM = SphericalHarmonicsFamily(T, R)
    S = FAM()
    # N is chosen so that it is at least one bigger than deg f, so the Jacobi
    # mats work
    N = 20
    f = (x,y,z) -> exp(x)*y*z
    F = Fun(f, S, 2(N+1)^2)
    cfs = F.coefficients
    @test length(cfs) == (N+1)^2
    Jx = jacobix(S, N)
    Jy = jacobiy(S, N)
    Jz = jacobiz(S, N)
    @test Fun(S, Jx * cfs)(pt) ≈ pt[1] * f(pt...)
    @test Fun(S, Jy * cfs)(pt) ≈ pt[2] * f(pt...)
    @test Fun(S, Jz * cfs)(pt) ≈ pt[3] * f(pt...)
end

@testset "laplaceoperator" begin
    FAM = SphericalHarmonicsFamily(T, R)
    S = FAM()
    inds = [4; 5; 10] # randomly chosen, each ints ≥ 2
    f = (x,y,z) -> x^inds[1] * y^inds[2] * z^inds[3]
    N = sum(inds)
    F = Fun(f, S, 2(N+1)^2)
    Δs = laplacianoperator(S, N)
    cfs = Δs * F.coefficients
    @test Fun(S, cfs)(pt) ≈ laplaciantest(S, inds, pt)
end

# @testset "Operator Clenshaw" begin
#
# end

# These involve VSHs (tangent space)
# @testset "divergenceoperator" begin
#
# end
#
# @testset "gradientoperator" begin
#
# end
#
# @testset "unitcrossproductoperator" begin
#
# end
