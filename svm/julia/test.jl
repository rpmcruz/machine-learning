using MLBase
using RDatasets
include("svm.jl")

df = dataset("datasets", "iris")
X = Array(df[:, 1:4])
y = Array{Int}([c == "setosa" for c in df[:, 5]])

svm = SVM(0)
for ix in  StratifiedKfold(y, 3)
    Xtr = X[ix,:]
    ytr = y[ix]
    not_ix = setdiff(1:length(y), ix)
    Xts = X[not_ix,:]
    yts = y[not_ix]
    fit(svm, Xtr, ytr)
    @printf("Accuracy: %.3f\n", sum(predict(svm, Xts) .== yts)/length(y))
end
