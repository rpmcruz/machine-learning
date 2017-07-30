# Ricardo: my implementation of gradient boosting for quantiles

using DecisionTree

## Initial model

type ZerosDummyModel
end

function fit(self::ZerosDummyModel, X::Array{Float64,2}, y::Array{Float64})
end

function predict(self::ZerosDummyModel, X::Array{Float64,2})
    zeros(size(X,1))
end

type TauDummyModel
    tau::Float64
    quantile::Float64

    TauDummyModel(tau) = new(tau, 0)
end

function fit(self::TauDummyModel, X::Array{Float64,2}, y::Array{Float64})
    self.quantile = quantile(y, self.tau)
end

function predict(self::TauDummyModel, X::Array{Float64,2})
    self.quantile * ones(size(X,1))
end

## The model

type QBR
    tau::Float64
    M::Int
    eta::Float64
    models::Array{Any}
    maxdepth::Int

    QBR(tau, M, eta, first_model, maxdepth=3) =
        new(tau, M, eta, [first_model], maxdepth)
end

function fit(self::QBR, X::Array{Float64,2}, y::Array{Float64})
    # step 0
    fit(self.models[1], X, y)
    # step 1
    for m in 1:self.M
        yp = predict(self, X)
        # step 2
        res = (y-yp .>= 0) - (1-self.tau)
        # step 3
        g = DecisionTreeRegressor(maxdepth=self.maxdepth)
        DecisionTree.fit!(g, X, res)
        # step 4
        push!(self.models, g)
    end
end

function predict(self::QBR, X::Array{Float64,2})
    yp0 = predict(self.models[1], X)
    if length(self.models) > 1
        yp0 + sum([self.eta * DecisionTree.predict(f, X) for f in self.models[2:end]])
    else
        yp0
    end
end
