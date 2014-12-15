function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

M = size(data, 2);%number of examples
stackSize = numel(stack); %number of layers (without softmax output layer and fake input layer i.e. input data)
depth = stackSize+1; %full number of layers (with input layer i.e. input data)
layerActivations = cell(depth,1);
layerActivations{1} = data;%first layer "activation" is just input data


%autoencoder forward pass

for d = 1:stackSize
    z1= stack{d}.w *layerActivations{d} + repmat(stack{d}.b,1,M);
    layerActivations{d+1} = sigmoid(z1);
end



%softmax forward pass

tempH = softmaxTheta*layerActivations{stackSize+1};
% M is the matrix as described in the text
tempH = bsxfun(@minus, tempH, max(tempH, [], 1));
tempH = exp(tempH);
H = bsxfun(@rdivide,tempH, sum(tempH));%normalize output of softmax

[~,pred] = max(H);






% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
