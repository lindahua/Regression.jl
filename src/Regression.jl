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
	GDSolver


# source files

include("common.jl")
include("linearreg.jl")
include("solve.jl")
include("print.jl")
include("regproblems.jl")

end # module
