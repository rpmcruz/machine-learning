## Activation functions
# note: for efficiency, the derivative acts on the activation itself!

function sigmoid(x::Array{Float64,2}, deriv::Bool)
    if deriv
        x .* (1-x)
    else
        1./(1+exp(-x))
    end
end
function linear(x::Array{Float64,2}, deriv::Bool)
    if deriv
        ones(size(x))
    else
        x
    end
end

## Loss functions

function square_loss(y::Array{Float64}, yp::Array{Float64}, deriv::Bool)
    if deriv
        yp - y
    else
        0.5*(yp - y).^2
    end
end
function pinball(y::Array{Float64}, yp::Array{Float64}, tau::Float64, deriv::Bool)
    if deriv
        taus = [tau-1, 0, tau]
        taus[trunc(Int, sign(yp-y)+2)]
    else
        max(y-yp,0)*tau + max(yp-y,0)*(1-tau)
    end
end
function pinball_q25(y::Array{Float64}, yp::Array{Float64}, deriv::Bool)
    pinball(y, yp, 0.25, deriv)
end
function pinball_q50(y::Array{Float64}, yp::Array{Float64}, deriv::Bool)
    pinball(y, yp, 0.50, deriv)
end
function pinball_q75(y::Array{Float64}, yp::Array{Float64}, deriv::Bool)
    pinball(y, yp, 0.75, deriv)
end

## Neural network

type Layer
    W::Array{Float64,2}
    B::Array{Float64,2}
    act::Function

    ## Initialize random weights
    # Glorot and Bengio (2010), sample a Uniform(-r, r)
    # for hyperbolic tangent units, r=sqrt(6/#in+#out)
    # for sigmoid units, r=4*sqrt(6/(#in+#out))

    function Layer(fan_in::Int, fan_out::Int, act::Function)
        # Gorot
        #r = 4*sqrt(6/(fan_in+fan_out))
        #W = r * (2*rand(fan_in, fan_out)-1)
        #B = r * (2*rand(1, fan_out)-1)
        # simple normal
        W = randn(fan_in, fan_out)
        B = randn(1, fan_out)
        new(W, B, act)
    end
end

type NeuralNet
    layers::Array{Layer}
    loss::Function
    verbose::Bool

    NeuralNet(loss, verbose) = new([], loss, verbose)
end

function add(nn::NeuralNet, layer::Layer)
    push!(nn.layers, layer)
end

function fpass(nn::NeuralNet, X::Array{Float64,2})
    a = X
    A = [a]
    for layer in nn.layers
        a = layer.act(a*layer.W .+ layer.B, false)
        push!(A, a)
    end
    A
end

function bpass(nn::NeuralNet, y::Array{Float64}, A::Array{Array{Float64,2}}, eta::Float64)
    dW = [zeros(size(l.W)) for l in nn.layers]
    dB = [zeros(size(l.B)) for l in nn.layers]
    N = length(y)

    # backward pass
    delta = nn.loss(y, A[end], true) .* nn.layers[end].act(A[end], true)
    dW[end] = A[end-1]' * delta * (1/N)
    dB[end] = mean(delta, 1)

    for l in length(nn.layers)-1:-1:2
        sp = nn.layers[l].act(A[l], true)
        delta = (nn.layers[l+1].W' * delta) .* sp
        dW[l] = A[l-1]' * delta * (1/N)
        dB[l] = mean(delta, 1)
    end
    (dW, dB)
end

function backprop(nn::NeuralNet, X::Array{Float64,2}, y::Array{Float64}, eta::Float64, maxit::Int)
    for epoch in 1:maxit
        A = fpass(nn, X)
        dW, dB = bpass(nn, y, A, eta)

        for l in 1:length(nn.layers)
            nn.layers[l].W -= eta * dW[l]
            nn.layers[l].B -= eta * dB[l]
        end

        if nn.verbose & (epoch % (maxit/10) == 0)
            err = mean(nn.loss(y, A[end], false))
            @printf("%4d %.2f\n", epoch, err)
        end
    end
end

function backprop_momentum(nn::NeuralNet, X::Array{Float64,2}, y::Array{Float64}, eta::Float64, maxit::Int)
    # https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum

    deltas_W = [zeros(size(layer.W)) for layer in nn.layers]
    deltas_B = [zeros(size(layer.B)) for layer in nn.layers]
    alpha = eta/10

    for epoch in 1:maxit
        A = fpass(nn, X)
        dW, dB = bpass(nn, y, A, eta)

        for l in 1:length(nn.layers)
            deltas_W[l] = (eta * dW[l]) .+ (alpha * deltas_W[l])
            deltas_B[l] = (eta * dB[l]) .+ (alpha * deltas_B[l])
            nn.layers[l].W -= deltas_W[l]
            nn.layers[l].B -= deltas_B[l]
        end

        if nn.verbose & (epoch % (maxit/10) == 0)
            err = mean(nn.loss(y, A[end], false))
            @printf("%4d %.2f\n", epoch, err)
        end
    end
end

function rprop(nn::NeuralNet, X::Array{Float64,2}, y::Array{Float64}, eta::Float64, maxit::Int)
    etas = [0.5, 1, 1.2]
    last_dW = [zeros(size(layer.W)) for layer in nn.layers]
    last_dB = [zeros(size(layer.B)) for layer in nn.layers]
    delta_0 = eta  # 0.1
    delta_min = 1e-6
    delta_max = 50
    deltas_W = [delta_0*ones(size(layer.W)) for layer in nn.layers]
    deltas_B = [delta_0*ones(size(layer.B)) for layer in nn.layers]

    for epoch in 1:maxit
        A = fpass(nn, X)
        dW, dB = bpass(nn, y, A, eta)

        for l in 1:length(nn.layers)
            # update deltas
            gt_0 = last_dW[l] .* dW[l] .> 0
            ge_0 = last_dW[l] .* dW[l] .>= 0
            E = etas[gt_0 + ge_0 + 1]
            deltas_W[l] .*= E
            deltas_W[l] = max(deltas_W[l], delta_min)
            deltas_W[l] = min(deltas_W[l], delta_max)

            gt_0 = last_dB[l] .* dB[l] .> 0
            ge_0 = last_dB[l] .* dB[l] .>= 0
            E = etas[gt_0 + ge_0 + 1]
            deltas_B[l] .*= E
            deltas_B[l] = max(deltas_B[l], delta_min)
            deltas_B[l] = min(deltas_B[l], delta_max)

            # update weights (using the deltas)
            nn.layers[l].W -= deltas_W[l] .* sign(dW[l])
            nn.layers[l].B -= deltas_B[l] .* sign(dB[l])

            # save derivatives
            lt_0 = last_dW[l] .* dW[l] .< 0
            #if any(lt_0)
            #    println("Piorou ", l)
            #end
            last_dW[l] = dW[l] .* (1-lt_0)
            lt_0 = last_dB[l] .* dB[l] .< 0
            last_dB[l] = dB[l] .* (1-lt_0)
        end

        if nn.verbose & (epoch % (maxit/10) == 0)
            err = mean(nn.loss(y, A[end], false))
            @printf("%4d %.2f\n", epoch, err)
        end
    end
end

function predict(nn::NeuralNet, X::Array{Float64,2})
    fpass(nn, X)[end]
end
