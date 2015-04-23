module Regression

using Compat
using Reexport
using ArrayViews
@reexport using EmpiricalRisks

import Base.LinAlg: BlasReal
import Base.LinAlg.LAPACK: gels!, gelsy!, gelsd!
import EmpiricalRisks: value, value_and_grad!

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
	GDSolver,
	BFGSSolver


# source files

include("common.jl")
include("linearreg.jl")
include("regproblems.jl")
include("optimbase.jl")
include("solve.jl")
include("print.jl")

const default_solver = BFGSSolver()

end # module
