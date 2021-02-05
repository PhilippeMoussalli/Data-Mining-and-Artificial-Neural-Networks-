function [TrainData,TrainTarget]=getTimeSeriesTrainData(trainset, p)

TrainMatrix=[];
for i=1:p
    TrainMatrix=[TrainMatrix,trainset(i:end-p+i)];
end
TrainData=TrainMatrix(1:end-1,:)';
TrainTarget=trainset(p+1:end)';



figure
for i=1:3
    hold on 
    plot([30:5:120],finErrCopy(:,i))
    xlabel('Lag number')
    ylabel('MSE')
    title('MSE vs lag number on validation set ')
    ylim([0 10000])
end 

legend('20 Neurons','30 Neurons','50 Neurons')
