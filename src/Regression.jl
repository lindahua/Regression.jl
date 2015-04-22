module Regression

using Compat
using Reexport
using ArrayViews
@reexport using EmpiricalRisks

import Base.LinAlg: BlasReal
import Base.LinAlg.LAPACK: gels!, gelsy!, gelsd!

export
	# linearreg
	llsq,
	ridgereg,

	# solve
	RiskMinSolver,
	GDSolver,
	BFGSSolver


# source files

include("common.jl")
include("linearreg.jl")
include("regproblems.jl")
include("solvers.jl")
include("solve.jl")
include("print.jl")

end # module
