function [biases, weights] = hidden_layer_weights(net)
    if length(net.layers) ~= 2
        error('This function is meant for networks with one hidden layer');
    end
    biases = net.b{1};
    weights = net.IW{1,1};
end
