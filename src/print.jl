

function print_iter_head(;with_gnorm::Bool=true)
    if with_gnorm
        @printf("%5s   %12s   %12s   %12s   %12s\n",
            "Iter", "f.value", "f.change", "g.norm", "step")
        println("==================================================================")
    else
        @printf("%5s   %12s   %12s   %12s\n",
            "Iter", "f.value", "f.change", "step")
        println("======================================================")
    end
end


function print_iter(t::Int, v::Real)
    @printf("%5d   %12.4e\n", t, v)
end

function print_iter(t::Int, v::Real, v_pre::Real, g::StridedArray, α::Real)
    @printf("%5d   %12.4e   %12.4e   %12.4e   %12.4e\n", t, v, v - v_pre, vecnorm(g), α)
end

function print_iter(t::Int, v::Real, v_pre::Real, α::Real)
    @printf("%5d   %12.4e   %12.4e   %12.4e\n", t, v, v - v_pre, α)
end

function print_final(t::Int, v::Real, converged::Bool)
    if converged
        @printf("Converged with %d iterations @ f.value = %.4e\n", t, v)
    else
        @printf("Terminated without convergence with %d iterations @ f.value = %.4e\n", t, v)
    end
end
