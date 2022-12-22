import Pkg;
Pkg.activate(".")
using LinearAlgebra
using SparseArrays

using Ferrite
using FerriteGmsh
using Logging, LoggingExtras

debug_logger = ConsoleLogger(stderr, Logging.Debug)
global_logger(debug_logger);

element_dim = 2
ambient_dim = 2

gmsh.initialize()
gmsh.open("box.msh")

gmsh.model.mesh.renumberNodes()
gmsh.model.mesh.renumberElements()

nodes = tonodes()
elements, gmsh_elementidx = toelements(element_dim)
cellsets = tocellsets(element_dim, gmsh_elementidx)

gmsh.finalize()

gridi = Grid(elements, nodes)

abstract type VectorInterpolation{dim, shape, order} <: Interpolation{dim, shape, order} end
struct RaviartThomas{dim, shape, order} <: VectorInterpolation{dim, shape,order} end
#struct Lagrange{refdim, ambientdim, shape, order} <: Interpolation{refdim, ambientdim, shape, order} end

import .Ferrite.AbstractRefShape
Ferrite.debug_mode()
Ferrite.getnbasefunctions(::RaviartThomas{2,RefTetrahedron,1}) = 3
#Ferrite.getlowerdim(::RaviartThomas{2, RefTetrahedron, 1}) = RaviartThomas{1, RefCube, 1}
Ferrite.nvertexdofs(::RaviartThomas{2, RefTetrahedron, 1}) = 0
Ferrite.nedgedofs(::RaviartThomas{2, RefTetrahedron, 1}) = 1
Ferrite.nfacedofs(::RaviartThomas{2, RefTetrahedron, 1}) = 0
getdim(::RaviartThomas{2, RefTetrahedron, 1}) = 2
getdim(::Lagrange{2, RefTetrahedron, 1}) = 2
getrefshape(::Interpolation{dim, shape, order}) where {dim, shape, order} = shape
getrefshape(::VectorInterpolation{dim, shape, order}) where {dim, shape, order} = shape
Ferrite.faces(::RaviartThomas{2,RefTetrahedron,1}) = ((1,2,3))
Ferrite.edges(::RaviartThomas{2,RefTetrahedron,1}) = ((1,2), (2,3), (3,1))
Ferrite.edges(c::Triangle) = ((c.nodes[1], c.nodes[2]), (c.nodes[2], c.nodes[3]), (c.nodes[3], c.nodes[1]))

function Ferrite.reference_coordinates(::RaviartThomas{2, RefTetrahedron, 1})
    return [Vec{2, Float64}((0.5, 0.0)),
        Vec{2, Float64}((0.0, 0.5)),
        Vec{2, Float64}((0.5, 0.5))]
end

function value(ip::RaviartThomas{2, RefTetrahedron, 1}, i::Int, ξ::Vec{2})
    x = ξ[1]
    y = ξ[2]
    i == 1 && return Vec(((y-1)*0.5,0.5*x))
    i == 2 && return Vec(((x-1)*0.5, 0.5*y))
    i == 3 && return Vec((x*0.5, y*0.5))
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end


# Ferrite.getnbasefunctions(::RaviartThomas{2,RefTetrahedron,1}) = 3
# Ferrite.value(ip::RaviartThomas{2, RefTetrahedron, 1}, i::Int, ξ::Vec{2}) = value(ip, i, ξ)


function value(ip::Lagrange{2,RefTetrahedron,1}, i::Int, ξ::Vec{2})
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return ξ_x
    i == 2 && return ξ_y
    i == 3 && return 1. - ξ_x - ξ_y
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

struct CellRTValues{dim,T<:Real,refshape<:AbstractRefShape,M} <: CellValues{dim,T,refshape}
    N::Matrix{Vec{dim,T}} # Values on reference element
    dNdx::Matrix{Tensor{2,dim,T,M}} # Derivative on element
    dNdξ::Matrix{Tensor{2,dim,T,M}} # Derivative on reference element
    detJdV::Vector{T} # Volume element in reference element frame ($\sqrt{\det g_{ij}}$, where $g_ij = J_i ⋅ J_j$ with $J_i$ being the i:th row of reference element mapping jacobian)
    M::Matrix{T} # Geometric interpolation function values
    dMdξ::Matrix{Vec{dim,T}} # Derivatives of interpolation function wrt reference element coordinates
    qr::QuadratureRule{dim,refshape,T}
    J::Matrix{T} # Jacobian of reference element mapping
    # The following fields are deliberately abstract -- they are never used in
    # performance critical code, just stored here for convenience.
    func_interp::Interpolation{dim,refshape}
    geo_interp::Interpolation{dim,refshape}
end


function CellRTValues( ::Type{T},
    quad_rule::QuadratureRule{dim,shape}, 
    func_interpol::VectorInterpolation{dim, shape, order},
    geom_interpol::Interpolation=func_interpol) where {dim,T,shape<:AbstractRefShape, order}

    @assert getdim(func_interpol) == getdim(geom_interpol)
    @assert getrefshape(func_interpol) == getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N    = fill(zero(Vec{dim,T})      * T(NaN), n_func_basefuncs, n_qpoints)
    dNdx = fill(zero(Tensor{2,dim,T}) * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(Tensor{2,dim,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M    = fill(zero(T)          * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(Vec{dim,T}) * T(NaN), n_geom_basefuncs, n_qpoints)

    for (qp, ξ) in enumerate(quad_rule.points)
        basefunc_count = 1
        for basefunc in 1:getnbasefunctions(func_interpol)
            N_temp = value(func_interpol, basefunc, ξ)
            N[basefunc_count,qp] = N_temp
            basefunc_count += 1
        end
        for basefunc in 1:n_geom_basefuncs
            dMdξ[basefunc, qp], M[basefunc, qp] = gradient(ξ -> value(geom_interpol, basefunc, ξ), ξ, :all)
        end
    end

    detJdV = fill(T(NaN), n_qpoints)
    MM = Tensors.n_components(Tensors.get_base(eltype(dNdx)))

    CellRTValues{dim,T,shape,MM}(N, dNdx, dNdξ, detJdV, M, dMdξ, quad_rule, func_interpol, geom_interpol)
end

function Ferrite.reinit!(cv::CellRTValues, x::AbstractVector{Vec{dim,T}}) where {dim, T}
    @debug display(cv)
    F = (x[3]-x[1], x[2]-x[1])  

end

dh = with_logger(debug_logger) do
    dh = DofHandler(gridi)
    push!(dh, :u, 1, RaviartThomas{2, RefTetrahedron, 1}())
    close!(dh)
    dh
end

K = create_sparsity_pattern(dh)

function loop_cells(cellvalues, 
    K::SparseMatrixCSC, 
    dh::DofHandler)
    n_basefuncs = getnbasefunctions(cellvalues)
    for (cellcount, cell) in enumerate(CellIterator(dh))
        reinit!(cellvalues, cell.coords)
        display(typeof(cellvalues))
        dΩ = getdetJdV(cellvalues, 1)
        δu = shape_value(cellvalues, 1, 1) 
        #map((δu, dΩ)) do x display(x) end
        #display(getnquadpoints(cellvalues))
        coords=getcoordinates(cell)
        dofs=celldofs(cell)
    end
end

ip = RaviartThomas{element_dim, RefTetrahedron, 1}()
qr = QuadratureRule{element_dim, RefTetrahedron}(2)
geom_ip = Lagrange{element_dim, RefTetrahedron, 1}()

cellvalues = CellRTValues(Float64, qr, ip, geom_ip)
loop_cells(cellvalues, K, dh)
