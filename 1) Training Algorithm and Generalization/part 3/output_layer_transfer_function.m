function [fun] = output_layer_transfer_function(net)
    fun = str2func(net.layers{end}.transferFcn);
