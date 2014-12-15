function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

M = theta*data;
% M is the matrix as described in the text
M = bsxfun(@minus, M, max(M, [], 1));
M = exp(M);
%hipotesis
H = bsxfun(@rdivide,M, sum(M));

weightDecayTerm = 0.5 * lambda * sum(sum(theta .* theta));
%cost value
cost  = -(1/numCases) * sum(sum(log(H).*groundTruth)) + weightDecayTerm;

%gradient
thetagrad = -(1/numCases) * (groundTruth-H)*  data'; % this could be other way around
thetagrad = thetagrad + lambda * theta;
%thetagrad = thetagrad' + lambda * theta;

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end