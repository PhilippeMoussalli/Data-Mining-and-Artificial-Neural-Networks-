function [biases, weights] = output_layer_weights(net)
    biases = net.b{end};
    if length(net.layers) == 2        
        weights = net.LW{2,1};
    elseif length(net.layers) == 1
        weights = net.IW{1};
    else
        error('This function is meant for networks with at most one hidden layer');
    end
end
