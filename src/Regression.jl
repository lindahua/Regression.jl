module Regression

using Reexport
using ArrayViews
@reexport using EmpiricalRisks

import Base.LinAlg: BlasReal
import Base.LinAlg.LAPACK: gels!, gelsy!, gelsd!

export
	# linearreg
	llsq,
	ridgereg


# source files

include("common.jl")
include("linearreg.jl")

end # module
