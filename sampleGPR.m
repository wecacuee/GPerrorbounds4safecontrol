function [val,prob] = sampleGPR(gprModel, X)
%sampleGPR Samples Gaussian Process Regression model.
% 
%   I assume predict gives the mean and sigma for each sample as
%   prediction. 
%   
%   This functions uses the mean and sigma to get one sample value for
%   each test point and also returns the probability of that sample.
% 
n = size(X, 2);
[mu, sigma] = predict(gprModel, X);
val = randn(n).*sigma + mu;
prob = normpdf(val, mu, sigma);
end

