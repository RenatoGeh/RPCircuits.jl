using Plots
using Plots.PlotMeasures
using Serialization
using Statistics

using GLM
using DataFrames

gr()
theme(:ggplot2)

T = ["LearnSPN", "Strudel", "LearnPSDD", "XPC", "LearnRP-F", "LearnRP-100"]
D = ["accidents", "ad", "audio", "bbc", "netflix", "book", "20news", "reut52", "webkb", "dna",
     "jester", "kdd", "kosarek", "msnbc", "msweb", "nltcs", "plants", "pumsb-star", "eachmovie",
     "retail"]
# D = ["nltcs", "plants", "audio", "jester", "netflix", "accidents", "book", "dna"]
# compl = [16*16181, 69*17412, 100*15000, 100*9000, 100*15000, 111*12758, 500*8700, 180*1600]
compl_f(t) = t[1]*t[2]
# compl = compl_f.([(16, 16181), (69, 17412), (100, 15000), (100, 9000), (100, 15000), (111, 12758),
                  # (500, 8700), (180, 1600)])
compl = compl_f.([(12758, 111), (2461, 1556), (15000, 100), (1670, 1058), (15000, 100), (8700, 500), (11293, 910), (6532, 889), (2803, 839), (1600, 180), (9000, 100), (180092, 64), (33375, 190), (291326, 17), (29441, 294), (16181, 16), (17412, 69), (12262, 163), (4524, 500), (22041, 135)])
compl_factor = sum(compl)
compl /= sum(compl)

noise(x; none = true) = none ? compl : rand(length(x))/1000 .+ compl

# lspn = deserialize("/tmp/lspn_times.data")
# mix = deserialize("/tmp/mix.data")

# lspn = [508.43, 2294.26, 6398.44, 2031.83, 3001.07, 2967.81, 17877.36, 371.37]
lspn = [2967.81, 3990.37, 6398.44, 2356.04, 3001.07, 17877.36, 86400.0, 14011.39, 6213.18, 371.37, 2031.83, 86400.0, 30526.23, 86400.0, 27845.07, 508.43, 2294.26, 2372.17, 1857.31, 11348.8]
#rpf = [199, 26*60, 37*60, 29*60, 46*60, 33*60, 2*60*60, 11*60]
rpf = [2033.53740514,3829.86576853,2439.89931409,6585.95959358,2822.85072957,9479.56396775,54528.85751028,21326.59330875,9455.95653484,903.76631313,2021.67023010,9056.82513101,9151.09300262,4049.05010583,12411.49250528,165.35538411,1442.27518563,3399.82128624,3817.41615839,4656.65541645,]
#rp100 = [15, 127, 243, 288, 276, 240, 4*25+13, 7*60+38]
rp100 = [370.82758998,1720.20343046,373.66080795,3623.36339598,424.59810697,1988.51490025,11699.11715961,5795.96426348,4102.43949388,561.51098697,496.03201525,639.52612708,967.28065251,480.87854870,1246.19634963,33.98674323,223.97567658,570.44301595,1177.62049659,537.85553393,]
#strudel = [3*60, 41*60, 33*60, 24*60, 14*60, 20*60, 8*60, 3647.18]
strudel = [9963.293731242, 3812.909606401, 29928.128985536, 11856.666507417, 20917.23873798, 11349.802951845, 16390.553270821, 12841.402733544, 10073.21872818, 3647.181746487, 37173.138120479, 12603.954262648, 17296.45800942, 19191.829953231, 631.214895751, 6036.684964334, -Inf, -Inf, -Inf, -Inf]
# lpsdd = [6*60, 26*60, 51*60, 37*60, 33*60, 41*60, 82*60, 3*60*60]
lpsdd = [6*60, 26*60, 51*60, 37*60, 33*60, 41*60, 82*60, 3*60*60, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf,]
# xpc = [17, 63, 60+58, 80, 128, 107, 146, 17]
xpc = [107.17, 195.33, 118.84, 240.64, 128.27, 146.0, 2379.05, 446.36, 201.71, 17.91, 80.95, 436.64, 132.29, 406.0, 290.0, 17.42, 63.57, 99.61, 158.4, 90.99,]
all = [lspn, strudel, lpsdd, xpc]

function linreg!(y, i, m)
  reg = lm(@formula(Y ~ X), DataFrame(X = compl, Y = y))
  b, a = coef(reg)
  plot!(range(extrema(compl)..., 10), x -> a*x+b; seriescolor = i, linewidth = 1)
end

shapes = [:circle, :diamond, :rect, :star5, :dtriangle, :utriangle]

supers = ['¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹', '⁰']
nxticks = 4
nyticks = 3

P = plot(compl, x -> 24*3600; label = nothing, seriescolor = :gray, linewidth = 2,
         xlabel = "Complexity (vars × insts)", ylabel = "Training time (secs)",
         legend = :outerright, legendfontsize = 8, tickfont = 6,
         xticks = (exp10.(range(log10.(extrema(compl))..., nxticks)), fill("10", nxticks) .* getindex.(Ref(supers), floor.(Int, range(log10.(extrema(compl .* compl_factor))..., nxticks)))),
         # yticks = (exp10.(2:nyticks+1), fill("10", nyticks) .* getindex.(Ref(supers), 2:nyticks+1)),
         titlefont = 10, guidefont = 8, grid = true, minorgrid = false, xscale = :log10,
         minorticks = false)
annotate!(compl[1]/2+(compl[2]-compl[1])/2, 24*3600-1e3, text("24h limit", :gray, 8, :top))
for (i, x) ∈ enumerate(all)
  scatter!(noise(x), x; label = T[i], seriescolor = 2, markershape = shapes[i])
end
scatter!(noise(rpf), rpf; label = T[length(all)+1], seriescolor = 1, markershape = shapes[length(all)+1])
scatter!(noise(rp100), rp100; label = T[length(all)+2], seriescolor = 3, markershape = shapes[length(all)+2])

savefig("/tmp/time.pdf")

P = plot(compl, x -> 24*3600; label = nothing, seriescolor = :gray, linewidth = 2,
         xlabel = "Complexity (vars × insts)", ylabel = "Training time (secs)",
         legend = :outerright, legendfontsize = 8, tickfont = 6,
         xticks = (exp10.(range(log10.(extrema(compl))..., nxticks)), fill("10", nxticks) .* getindex.(Ref(supers), floor.(Int, range(log10.(extrema(compl .* compl_factor))..., nxticks)))),
         # yticks = (exp10.(2:nyticks+1), fill("10", nyticks) .* getindex.(Ref(supers), 2:nyticks+1)),
         titlefont = 10, guidefont = 8, grid = true, minorgrid = false, xscale = :log10,
         minorticks = false)
annotate!(sum(extrema(compl))/2, 24*3600-1e3, text("24h limit", :gray, 8, :top))
for (i, x) ∈ enumerate(all)
  linreg!(x, 2, shapes[i])
end
linreg!(rpf, 1, shapes[length(all)+1])
linreg!(rpf, 3, shapes[length(all)+2])

savefig("/tmp/time_reg.pdf")

Q = plot(compl, x -> 24*3600; label = nothing, seriescolor = :gray, linewidth = 2,
         xlabel = "Complexity (vars × insts)", ylabel = "Training time (secs)",
         legend = :outerright, legendfontsize = 8, tickfont = 6,
         xticks = (exp10.(range(log10.(extrema(compl))..., nxticks)), fill("10", nxticks) .* getindex.(Ref(supers), floor.(Int, range(log10.(extrema(compl .* compl_factor))..., nxticks)))),
         yticks = (exp10.(2:nyticks+1), fill("10", nyticks) .* getindex.(Ref(supers), 2:nyticks+1)),
         titlefont = 10, guidefont = 8, xscale=:log10, yscale=:log10, grid = true, minorgrid = false,
         minorticks = false)
annotate!(mean(compl)/2, 24*3600-1e4, text("24h limit", :gray, 8, :top))
for (i, x) ∈ enumerate(all)
  scatter!(noise(x), x; label = T[i], seriescolor = 2, markershape = shapes[i])
end
scatter!(noise(rpf), rpf; label = T[length(all)+1], seriescolor = 1, markershape = shapes[length(all)+1])
scatter!(noise(rp100), rp100; label = T[length(all)+2], seriescolor = 3, markershape = shapes[length(all)+2])
savefig("/tmp/time_log.pdf")
