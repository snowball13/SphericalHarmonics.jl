module SphericalHarmonics

    using BlockBandedMatrices
    using BlockArrays
    using ApproxFun
    import ApproxFun: evaluate, domain,
                        domainspace, rangespace, bandwidths, prectype, canonicaldomain, tocanonical,
                        spacescompatible, points, transform, itransform, AbstractProductSpace,
                        checkpoints, plan_transform, clenshaw
    import ApproxFunOrthogonalPolynomials: PolynomialSpace, recα, recβ, recγ, recA, recB, recC
    import Base: in, *
    using StaticArrays
    using LinearAlgebra
    using SparseArrays
    using GenericLinearAlgebra
    using Test
    using OrthogonalPolynomialFamilies

    include("SphericalHarmonicsScalar.jl")
    include("SphericalHarmonicsTangent.jl")

end
