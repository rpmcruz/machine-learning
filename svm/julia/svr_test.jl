using MLBase
using RDatasets
include("svr.jl")

df = dataset("datasets", "airquality")
df = df[completecases(df),:]
X = Array{Float64,2}(df[:, 2:6])
y = Array{Float64}(df[:, 1])

function rmse(y, yp)
    sqrt(mean((y-yp).^2))
end

for ix in  Kfold(length(y), 3)
    Xtr = X[ix,:]
    ytr = y[ix]
    not_ix = setdiff(1:length(y), ix)
    Xts = X[not_ix,:]
    yts = y[not_ix]

    # our own OLS implementation for comparison purposes
    ols = hcat(ones(size(Xtr,1)), Xtr) \ ytr
    yp = hcat(ones(size(Xts,1)), Xts) * ols
    @printf("OLS: RMSE: %.3f\n", rmse(yts, yp))

    svr = SVR(0)
    fit(svr, Xtr, ytr)
    yp = predict(svr, Xts)
    @printf("SVR: RMSE: %.3f\n", rmse(yts, yp))
    println()
end
