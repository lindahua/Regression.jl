module Regression

using Reexport
using ArrayViews
@reexport using EmpiricalRisks

import Base.LinAlg: BlasReal, axpy!
import Base.LinAlg.LAPACK: gels!, gelsy!, gelsd!
import EmpiricalRisks: value, value!, value_and_grad!, predict

export
	# linearreg
	llsq,
	ridgereg,

	# regproblems
	UnivariateRegression,
	MultivariateRegression,

	linearreg,
	logisticreg,
	mlogisticreg,

	# solve
	GD,
	AGD,
	BFGS,
	LBFGS,

	# proxsolve
	ProximalDescent,
	ProxGD,
	ProxAGD


# source files

include("common.jl")
include("linearreg.jl")
include("regproblems.jl")
include("optimbase.jl")
include("solve.jl")
include("proxsolve.jl")
include("print.jl")

const default_solver = BFGS()

end # module
