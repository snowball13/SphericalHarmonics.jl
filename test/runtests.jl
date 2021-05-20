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
                            jacobix, jacobiy, jacobiz,
                            getopindex, getlm, getlmkind, convertcoeffsveclength



# Setup
rhoval(S::SphericalHarmonicsSpace, z) = sqrt(1 - z^2)
isindomain(FAM::SphericalHarmonicsFamily, pt) = pt[1]^2 + pt[2]^2 + pt[3]^2 ≈ 1.0
T = ComplexF64; R = Float64
FAM = SphericalHarmonicsFamily(T, R)
S = FAM()
ST = gettangentspace(S)
z = 0.3; y = 0.45; x = sqrt(1 - y^2 - z^2); pt = [x; y; z]; isindomain(FAM, pt)
ϕ = acos(z); θ = atan(y / x)
N = 20
resizedata!(S, N)
resizedata!(ST, N)

l = 21
    aa = [ST.A[l+1][1]; ST.A[l+1][2]; ST.A[l+1][3]]
    dd = [ST.DT[l+1][1] zeros(2(2l+3), 2(2l+1)) ST.DT[l+1][3]]
    real.(dd * aa) ≈ I



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
function phivec(ϕ::Real, θ::Real)
    """ Returns basis vector ϕ̲ at the point (ϕ, θ)
    """
    [cos(θ)*cos(ϕ); sin(θ)*cos(ϕ); -sin(ϕ)]
end
function thetavec(ϕ::Real, θ::Real)
    """ Returns basis vector θ̲ at the point (ϕ, θ)
    """
    [-sin(θ); cos(θ); 0]
end
function getvsh(ST::SphericalHarmonicsTangentSpace, l::Int, m::Int, kind::Int)
    """ Returns anon function of the VSH for l,m. kind=0 is ∇Y, kind=1 is ∇⟂Y.

    Note Φlm = (-1)^m conj(Φ) etc.
    """
    @assert (in(l, (1,2,3)) && abs(m) ≤ l) "Invalid l or m provided"
    @assert (in(kind, (0,1))) "Invalid kind"

    if l == 2
        if m == 2
            if kind == 0
                (ϕ, θ) -> sqrt(15/(8π)) * sin(ϕ) * exp(2im*θ) * (cos(ϕ)*phivec(ϕ, θ) + 1im*thetavec(ϕ, θ))
            elseif kind == 1
                (ϕ, θ) -> sqrt(15/(8π)) * sin(ϕ) * exp(2im*θ) * (-im*phivec(ϕ, θ) + cos(ϕ)*thetavec(ϕ, θ))
            end
        elseif m == 1
            if kind == 0
                (ϕ, θ) -> -sqrt(15/(8π)) * exp(1im*θ) * (cos(2ϕ)*phivec(ϕ, θ) + 1im*cos(ϕ)*thetavec(ϕ, θ))
            elseif kind == 1
                (ϕ, θ) -> sqrt(15/(8π)) * exp(1im*θ) * (1im*cos(ϕ)*phivec(ϕ, θ) - cos(2ϕ)*thetavec(ϕ, θ))
            end
        elseif m == 0
            if kind == 0
                (ϕ, θ) -> -(3/2) * sqrt(5/π) * sin(ϕ) * cos(ϕ) * phivec(ϕ, θ)
            elseif kind == 1
                (ϕ, θ) -> -(3/2) * sqrt(5/π) * sin(ϕ) * cos(ϕ) * thetavec(ϕ, θ)
            end
        elseif m == -2
            if kind == 0
                (ϕ, θ) -> (-1)^m * conj(sqrt(15/(8π)) * sin(ϕ) * exp(2im*θ) * (cos(ϕ)*phivec(ϕ, θ) + 1im*thetavec(ϕ, θ)))
            elseif kind == 1
                (ϕ, θ) -> (-1)^m * conj(sqrt(15/(8π)) * sin(ϕ) * exp(2im*θ) * (-im*phivec(ϕ, θ) + cos(ϕ)*thetavec(ϕ, θ)))
            end
        elseif m == -1
            if kind == 0
                (ϕ, θ) -> (-1)^m * conj(-sqrt(15/(8π)) * exp(1im*θ) * (cos(2ϕ)*phivec(ϕ, θ) + 1im*cos(ϕ)*thetavec(ϕ, θ)))
            elseif kind == 1
                (ϕ, θ) -> (-1)^m * conj(sqrt(15/(8π)) * exp(1im*θ) * (1im*cos(ϕ)*phivec(ϕ, θ) - cos(2ϕ)*thetavec(ϕ, θ)))
            end
        end
    elseif l == 1
        if m == 1
            if kind == 0
                (ϕ, θ) -> -sqrt(3/(8π)) * exp(1im*θ) * (cos(ϕ)*phivec(ϕ, θ) + 1im*thetavec(ϕ, θ))
            elseif kind == 1
                (ϕ, θ) -> sqrt(3/(8π)) * exp(1im*θ) * (1im*phivec(ϕ, θ) - cos(ϕ)*thetavec(ϕ, θ))
            end
        elseif m == 0
            if kind == 0
                (ϕ, θ) -> -sqrt(3/(4π)) * sin(ϕ) * phivec(ϕ, θ)
            elseif kind == 1
                (ϕ, θ) -> -sqrt(3/(4π)) * sin(ϕ) * thetavec(ϕ, θ)
            end
        elseif m == -1
            if kind == 0
                (ϕ, θ) -> (-1)^m * conj(-sqrt(3/(8π)) * exp(1im*θ) * (cos(ϕ)*phivec(ϕ, θ) + 1im*thetavec(ϕ, θ)))
            elseif kind == 1
                (ϕ, θ) -> (-1)^m * conj(sqrt(3/(8π)) * exp(1im*θ) * (1im*phivec(ϕ, θ) - cos(ϕ)*thetavec(ϕ, θ)))
            end
        end
    elseif l == 3
        if m == 1
            if kind == 0
                (ϕ, θ) -> -(1/8) * sqrt(21/π) * exp(1im*θ) * ((cos(ϕ)*(5*cos(ϕ)^2 - 1) - 10*sin(ϕ)^2*cos(ϕ)) * phivec(ϕ, θ)
                                                                + 1im*(5*cos(ϕ)^2 - 1)*thetavec(ϕ, θ))
            elseif kind == 1
                error("not implemented for this kind")
            end
        elseif m == -1
            if kind == 0
                (ϕ, θ) -> (-1)^m * conj(-(1/8) * sqrt(21/π) * exp(1im*θ) * ((cos(ϕ)*(5*cos(ϕ)^2 - 1) - 10*sin(ϕ)^2*cos(ϕ)) * phivec(ϕ, θ)
                                                                + 1im*(5*cos(ϕ)^2 - 1)*thetavec(ϕ, θ)))
            elseif kind == 1
                error("not implemented for this kind")
            end
        elseif m == 2
            if kind == 0
                (ϕ, θ) -> (1/4) * sqrt(105/(2π)) * exp(2im*θ) * ((2*sin(ϕ)*cos(ϕ)^2 - sin(ϕ)^3) * phivec(ϕ, θ)
                                                                    + 2im*sin(ϕ)*cos(ϕ)*thetavec(ϕ, θ))
            elseif kind == 1
                error("not implemented for this kind")
            end
        elseif m == -2
            if kind == 0
                (ϕ, θ) -> (-1)^m * conj((1/4) * sqrt(105/(2π)) * exp(2im*θ) * ((2*sin(ϕ)*cos(ϕ)^2 - sin(ϕ)^3) * phivec(ϕ, θ)
                                                                    + 2im*sin(ϕ)*cos(ϕ)*thetavec(ϕ, θ)))
            elseif kind == 1
                error("not implemented for this kind")
            end
        elseif m == 0
            if kind == 0
                (ϕ, θ) -> (3/4) * sqrt(7/π) * sin(ϕ) * (1 - 5*cos(ϕ)^2) * phivec(ϕ, θ)
            elseif kind == 1
                error("not implemented for this kind")
            end
        else
            error("not implemented for this m $m")
        end
    end
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

    inds = [4; 5; 10]
    N = sum(inds) + 1
    f = (x,y,z) -> x^inds[1] * y^inds[2] * z^inds[3]
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

@testset "Operator Clenshaw" begin
    # TODO
end

#---
# These involve VSHs (tangent space)

@testset "Evaluation" begin
    FAM = SphericalHarmonicsFamily(T, R)
    S = FAM()
    ST = gettangentspace(S)
    N = 20
    resizedata!(ST, N+1)

    # Real coeffs
    for l = 1:2, m = -l:l, kind = 0:1
        vshactual = getvsh(ST, l, m, kind)
        cfs = zeros(T, getopindex(ST, l, l, 1)); cfs[getopindex(ST, l, m, kind)] = 1; cfs
        vsh = evaluate(cfs, ST, pt)
        @test vsh ≈ vshactual(ϕ, θ)
    end
    # Imaginary coeffs
    for l = 1:2, m = -l:l, kind = 0:1
        vshactual = getvsh(ST, l, m, kind)
        cfs = zeros(T, getopindex(ST, l, l, 1)); cfs[getopindex(ST, l, m, kind)] = 1im; cfs
        vsh = evaluate(cfs, ST, pt)
        @test vsh ≈ im * vshactual(ϕ, θ)
    end
end

# @testset "Transform" begin
#
# end
#
# @testset "Declaring a function" begin
#
# end

@testset "Jacobi matrices" begin
    FAM = SphericalHarmonicsFamily(T, R)
    S = FAM()
    ST = gettangentspace(S)
    N = 5
    resizedata!(ST, N+1)
    Jx = jacobix(ST, N)
    Jy = jacobiy(ST, N)
    Jz = jacobiz(ST, N)

    for l = 1:2, m = -l:l, kind = 0:1
        vshactual = getvsh(ST, l, m, kind)
        cfs = zeros(T, getopindex(ST, N, N, 1)); cfs[getopindex(ST, l, m, kind)] = 1; cfs
        @test evaluate(Jx * cfs, ST, pt) ≈ x * vshactual(ϕ, θ)
        @test evaluate(Jy * cfs, ST, pt) ≈ y * vshactual(ϕ, θ)
        @test evaluate(Jz * cfs, ST, pt) ≈ z * vshactual(ϕ, θ)
    end
end

@testset "divergenceoperator" begin
    FAM = SphericalHarmonicsFamily(T, R)
    S = FAM()
    ST = gettangentspace(S)
    N = 20
    D = divergenceoperator(ST, N)

    kind = 0
    for l = 1:N, m = -l:l
        fcfs = zeros(T, getopindex(S, N, N)); fcfs[getopindex(S, l, m)] = 1; fcfs
        factual = (x,y,z) -> -l * (l+1) * Fun(S, fcfs)(x,y,z)
        cfs = zeros(T, getopindex(ST, N, N, 1)); cfs[getopindex(ST, l, m, kind)] = 1; cfs
        f = Fun(S, convertcoeffsveclength(S, D * cfs))
        @test f(pt) ≈ factual(pt...)
    end
    kind = 1
    for l = 1:N, m = -l:l
        fcfs = zeros(T, getopindex(S, N, N)); fcfs[getopindex(S, l, m)] = 1; fcfs
        factual = (x,y,z) -> -l * (l+1) * Fun(S, fcfs)(x,y,z)
        cfs = zeros(T, getopindex(ST, N, N, 1)); cfs[getopindex(ST, l, m, kind)] = 1; cfs
        f = Fun(S, convertcoeffsveclength(S, D * cfs))
        @test f(pt) == 0
    end
end

@testset "gradientoperator" begin
    FAM = SphericalHarmonicsFamily(T, R)
    S = FAM()
    ST = gettangentspace(S)
    N = 20
    G = gradientoperator(ST, N)
    for l = 1:N, m = -l:l
        cfs = zeros(T, getopindex(S, N, N)); cfs[getopindex(S, l, m)] = 1; cfs
        Fcfs = zeros(T, getopindex(ST, N, N, 1)); Fcfs[getopindex(ST, l, m, 0)] = 1; Fcfs
        @test G * convertcoeffsveclength(S, cfs; tosmall=false) == Fcfs
    end
end

@testset "unitcrossproductoperator" begin
    FAM = SphericalHarmonicsFamily(T, R)
    S = FAM()
    ST = gettangentspace(S)
    N = 20
    Rcross = unitcrossproductoperator(ST, N)

    for l = 1:N, m = -l:l, kind=0:1
        cfs = zeros(T, getopindex(ST, N, N, 1)); cfs[getopindex(ST, l, m, kind)] = 1; cfs
        cfs2 = zeros(T, getopindex(ST, N, N, 1)); cfs2[getopindex(ST, l, m, abs(kind-1))] = 1; cfs2
        @test Rcross * cfs == (-1)^kind * cfs2
    end
end
