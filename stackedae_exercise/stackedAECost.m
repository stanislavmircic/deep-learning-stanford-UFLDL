function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);%number of examples
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%


% ------------------------------------------------------------------
% Forward pass
% ------------------------------------------------------------------
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



% ------------------------------------------------------------------
% Cost
% ------------------------------------------------------------------

weightDecayTerm = 0.5 * lambda * sum(sum(softmaxTheta .* softmaxTheta));
%cost value
cost  = -(1/M) * sum(sum(log(H).*groundTruth)) + weightDecayTerm;


% ------------------------------------------------------------------
% Gradient
% ------------------------------------------------------------------

%softmax gradient

softmaxThetaGrad = -(1/M) * (groundTruth-H)*  layerActivations{stackSize+1}'; % this could be other way round
softmaxThetaGrad = softmaxThetaGrad + lambda * softmaxTheta;




% delta
% d = cell(n+1);
% d{n+1} = -(softmaxTheta' * (y - p)) .* a{n+1} .* (1 -a{n});
% 
% for l = (n:-1:2)
%     d{l} = stack{l}.w' * d{l+1} .* a{l} .* (1-a{l});
% end
% 
% for l = (n:-1:1)
%     stackgrad{l}.w = d{l+1} * a{l}' / m;
%     stackgrad{l}.b = sum(d{l+1}, 2) / m;
% end



delta = cell(stackSize+1);
%derivative of softmax function here is activation(k+1)*(1-activation(k))
%why?
delta{stackSize+1} = -(softmaxTheta' * (groundTruth-H)) .* layerActivations{stackSize+1}.*(1-layerActivations{stackSize+1});


for l = (stackSize:-1:2)
    delta{l} = stack{l}.w' * delta{l+1} .* layerActivations{l} .* (1-layerActivations{l});
end

for l = (stackSize:-1:1)
    stackgrad{l}.w = delta{l+1} * layerActivations{l}' / M;
    stackgrad{l}.b = sum(delta{l+1}, 2) / M;
end

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
