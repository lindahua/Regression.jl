module Regression

using Reexport
@reexport using EmpiricalRisks

import Base.LinAlg: BlasReal
import Base.LinAlg.LAPACK: gels!, gelsy!, gelsd!

export

	# linearreg
	llsq_qrlq, wllsq_qrlq,
	llsq_orth, wllsq_orth,
	llsq_svd, wllsq_svd,
	linearreg_lsq, ridgereg


# source files

include("linearreg.jl")

end # module
