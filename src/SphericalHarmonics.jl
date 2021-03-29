module SphericalHarmonics

    using BlockBandedMatrices
    using BlockArrays
    using Base.Test
    using Base.LinAlg
    using FastTransforms

    export Jx, Jy, Jz, sh_eval, func_eval, func_eval_operator, func_eval_jacobi,
            grad_Jx, grad_Jy, grad_Jz, grad_sh, grad_perp_sh, laplacian_sh, tangent_basis_eval,
            get_clenshaw_matrices, tangent_func_eval, func_eval_grad_jacobi,
            function2sph, gradP1_dot_prodcuct_operators, tangent_space_dot_product

    include("SphericalHarmonicsScalar.jl")
    include("SphericalHarmonicsTangent.jl")
    include("DotProduct.jl")

end
