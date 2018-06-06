function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
meanVal = zeros(1, size(X, 2));    % mu is initiate with 1 row and number of col equal to X
standardDiviation  = zeros(1, size(X, 2)); % sigma is initiate with 1 row and number of col equal to X


meanVal = mean(X);
standardDiviation = std(X);

% for loop for col wise operation
% In a single loop all the values of col will be normalize
% size(X, 1 means number of rows and 2 means calc size of colmns)
for col = 1 : size(X,2)

X_norm(:,col) = ( X(:,col)- meanVal(col)) / standardDiviation(col);
end

end
