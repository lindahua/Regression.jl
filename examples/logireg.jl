# Demo of logistic regression

println("Import packages ...")
using Regression
using Winston

println("Generate data ...")

r = rand() * (2pi)
rot = [cos(r) -sin(r); sin(r) cos(r)]

n = 500
xp = randn(2, n) .* [0.25, 1.0] .+ [2.0; 0.0]
xn = randn(2, n) .* [0.25, 1.0] .+ [1.0; 0.0]
xp = rot * xp
xn = rot * xn

x = [xp xn]
y = [ones(n); -ones(n)]

println("Logistic regression ...")

theta0 = zeros(3)
theta, objv = logisticreg(x, y, 1.0e-3, theta0; by_columns=true, bias=true, show_trace=true)
println("   theta = $(theta)")
println("   objv  = $(objv)")

println("Visualize ...")

function visboundary(theta, rhs, obs, color)
	x = obs[1,:]
	y = obs[2,:]
	xmin = minimum(x)
	xmax = maximum(x)
	ymin = minimum(y)
	ymax = maximum(y)
	a = theta[1]
	b = theta[2]
	c = theta[3]
	if abs(a) > abs(b)
		y0 = ymin
		y1 = ymax
		x0 = (rhs - c - b * y0) / a
		x1 = (rhs - c - b * y1) / a
	else
		x0 = xmin
		x1 = xmax
		y0 = (rhs - c - a * x0) / b
		y1 = (rhs - c - a * x1) / b
	end
	Curve([x0 x1], [y0 y1], "color", color)
end

pl = FramedPlot()
add(pl, Points(xp[1,:], xp[2,:], "type", "dot", "color", "red"))
add(pl, Points(xn[1,:], xn[2,:], "type", "dot", "color", "blue"))
add(pl, visboundary(theta, 0., x, "green"))
add(pl, visboundary(theta, 1., x, "red"))
add(pl, visboundary(theta, -1., x, "blue"))
Winston.display(pl)

println("Presss enter to exit.")
readline(STDIN)




