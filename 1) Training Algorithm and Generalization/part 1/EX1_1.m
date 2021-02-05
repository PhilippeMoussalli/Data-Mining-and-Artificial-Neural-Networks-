x = linspace(0,1,21);
y = -sin(.8*pi*x);
P = polyfit(x,y,1);
yfit = P(1)*x+P(2);


%% Plotting
figure
hold on
plot(x,y,'-bs')
plot(x,yfit,'-mo');
legend('Original plot','Linear fit')

%% Neural network training 
net = fitnet(2);
view(net)
%net = configure(net,x,y);
%plotpc(net.IW{1},net.b{1},linehandle);
net.inputs{1}.processFcns = {};
net.outputs{2}.processFcns = {};
[net, tr] = train(net,x,y);

%activation function and weight values of first and second hidden neurons
[fun_H] = hidden_layer_transfer_function(net); %tansig
[biase_H, weight_H] = hidden_layer_weights(net);


%Hidden layer output
x_h = tansig(weight_H.*x + biase_H)



figure
hold on 
plot(x,x_h(1,:))
plot(x,x_h(2,:))
plot(x,y)
legend('x_i_1','x_i_2','y_i')
% plot(x,xh2)
% plot(x,yh1+yh2)
% plot(x,yh2)

%activation values of output neuron 
[fun_O] =output_layer_transfer_function(net);    %purelin
[biase_O, weight_O] = output_layer_weights(net);

%Network output (Prediction)
yn = purelin(weight_O * x_h + biase_O);


figure
hold on
plot(x,y,'-bs','MarkerFaceColor','b')
plot(x,yn,'-mo')
legend('Real plot','Output function of neuron')
set(gcf, 'Position',  [100, 100, 500, 400])