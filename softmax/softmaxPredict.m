function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

numCases = size(data, 2);

M = theta*data;
% M is the matrix as described in the text
M = bsxfun(@minus, M, max(M, [], 1));
M = exp(M);
%hipotesis
H = bsxfun(@rdivide,M, sum(M));
[m, pred] = max(H);





% ---------------------------------------------------------------------

end

