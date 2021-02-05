function [fun] = hidden_layer_transfer_function(net)
    if length(net.layers) ~= 2
        error('This function is meant for networks with one hidden layer');
    end
    fun = str2func(net.layers{1}.transferFcn);
