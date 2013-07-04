# Linear regression

println("Import packages ...")
using Winston
using Regression

println("Generate data ...")
n = 500
x = rand(n) * 10.
y = 2x + 1 + randn(n) * 0.5

println("Linear regression ...")
xmat = reshape(x, n, 1)
a = linearreg_lsq(xmat, y, bias=true)
println(" result = $a")

println("Visualize ...")

p = FramedPlot()
add(p, Points(x, y, "type", "dot", "color", "blue"))
add(p, Curve([0. 10.], a[1] * [0. 10.] + a[2], "color", "red"))
Winston.display(p)

println("Press enter to exit.\n")
readline(STDIN)

