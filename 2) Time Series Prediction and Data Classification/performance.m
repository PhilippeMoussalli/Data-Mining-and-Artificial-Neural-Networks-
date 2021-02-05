function [acc,sens,spec,TN,FP,FN,TP] = performance(pred_label,true_label)

% Compute the accuracy, sensitivity and specificity of the classifier

% Input: - pred_label: label predicted by the classifier
%        - true_label: true label

% Output: - acc: accuracy
%         - sens: sensitivity
%         - spec: specificity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% wake label is positive and sleep label in negative 

TP = 0;
TN = 0;
FP = 0;
FN = 0;

for i=1:length(pred_label)
    if pred_label(i)==true_label(i) && pred_label(i)==1 
        TP = TP+1;
    elseif pred_label(i)==true_label(i) && pred_label(i)==0
        TN = TN+1;
    elseif pred_label(i)~= true_label(i) && pred_label(i)==0
        FN = FN+1;
    else
        FP = FP+1;
    end
 
 acc = (TP+TN)/(TP+TN+FP+FN);
 sens = (TP)/(TP+FN);
 spec = (TN)/(FP+TN);

end
