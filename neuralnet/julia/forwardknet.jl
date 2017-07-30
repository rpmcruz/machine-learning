using Knet

## Loss functions

function mse(W, X::Array{Float64,2}, y::Array{Float64})
    sumabs2(y - fpass(W,X)) / length(y)
end
function pinball(W, X::Array{Float64,2}, y::Array{Float64}, tau::Float64)
    yp = fpass(W,X)
    sum(max(y-yp,0)*tau + max(yp-y,0)*(1-tau)) / length(y)
end
function pinball_q25(W, X::Array{Float64,2}, y::Array{Float64})
    pinball(W, X, y, 0.25)
end
function pinball_q50(W, X::Array{Float64,2}, y::Array{Float64})
    pinball(W, X, y, 0.50)
end
function pinball_q75(W, X::Array{Float64,2}, y::Array{Float64})
    pinball(W, X, y, 0.75)
end

## Neural network implementation

function create_weights(layers::Array{Int})
    W = Any[]
    for (fan_in, fan_out) in zip(layers[1:end-1], layers[2:end])
        # weight initialization
        push!(W, randn(fan_in, fan_out))
        push!(W, randn(1, fan_out))
    end
    W
end

function fpass(W, X::Array{Float64,2})
    for i in 1:2:length(W)-2
        X = sigm((X * W[i]) .+ W[i+1])
    end
    (X * W[end-1]) .+ W[end]
end

function minibatch(X, y, batchsize)
    N = length(y)
    ix = randperm(N)
    batches = Any[]
    for i in 1:batchsize:N
        j = min(i+batchsize-1, N)
        ix2 = ix[i:j]
        push!(batches, (X[ix2,:], y[ix2]))
    end
    batches
end

function batch_backprop(W::Array{Any}, X::Array{Float64,2}, y::Array{Float64}, eta::Float64, maxit::Int, loss::Function, batchsize::Int)
    lossgradient = grad(loss)

    for epoch in 1:maxit
        batches = minibatch(X, y, batchsize)
        for (Xb, yb) in batches
            dW = lossgradient(W, Xb, yb)
            for i in 1:length(W)
                W[i] -= eta * dW[i]
            end
        end

        if epoch % (maxit/10) == 0
            err = loss(W, X, y)
            @printf("%4d %.2f\n", epoch, err)
        end
    end
    W
end

function batch_backprop_momentum(W::Array{Any}, X::Array{Float64,2}, y::Array{Float64}, eta::Float64, maxit::Int, loss::Function, batchsize::Int)
    # https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum

    deltas_W = [zeros(size(w)) for w in W]
    alpha = eta/10

    lossgradient = grad(loss)

    for epoch in 1:maxit
        batches = minibatch(X, y, batchsize)
        for (Xb, yb) in batches
            dW = lossgradient(W, Xb, yb)
            for i in 1:length(W)
                deltas_W[i] = (eta * dW[i]) .+ (alpha * deltas_W[i])
                W[i] -= deltas_W[i]
            end
        end

        if epoch % (maxit/10) == 0
            err = loss(W, X, y)
            @printf("%4d %.2f\n", epoch, err)
        end
    end
    W
end

function batch_rprop(W::Array{Any}, X::Array{Float64,2}, y::Array{Float64}, eta::Float64, maxit::Int, loss::Function, batchsize::Int)
    etas = [0.5, 1, 1.2]
    last_dW = [zeros(size(w)) for w in W]
    delta_0 = eta  # 0.1
    delta_min = 1e-6
    delta_max = 50
    deltas_W = [delta_0*ones(size(w)) for w in W]

    lossgradient = grad(loss)

    for epoch in 1:maxit
        batches = minibatch(X, y, batchsize)
        for (Xb, yb) in batches
            dW = lossgradient(W, Xb, yb)

            for i in 1:length(W)
                # update deltas
                gt_0 = last_dW[i] .* dW[i] .> 0
                ge_0 = last_dW[i] .* dW[i] .>= 0
                E = etas[gt_0 + ge_0 + 1]
                deltas_W[i] .*= E
                deltas_W[i] = max(deltas_W[i], delta_min)
                deltas_W[i] = min(deltas_W[i], delta_max)

                # update weights (using the deltas)
                W[i] -= deltas_W[i] .* sign(dW[i])

                # save derivatives
                lt_0 = last_dW[i] .* dW[i] .< 0
                last_dW[i] = dW[i] .* (1-lt_0)
            end
        end

        if epoch % (maxit/10) == 0
            err = loss(W, X, y)
            @printf("%4d %.2f\n", epoch, err)
        end
    end
    W
end
