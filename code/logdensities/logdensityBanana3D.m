function [out, gradz] = logdensityBanana3D(z, mu, L, a, b) 
% z  : 1 x n matrix of 
% mu : 1 x n  mean vector
% L  : lower triangular Cholesky decomposition of the 
%      covariance matrix   


x = bananafun3D(z,[a b],1);

n = length(x); 

diff = x - mu; 
diff = diff(:); 
   
Ldiff = L\diff;
   
out = - 0.5*n*log(2*pi) - sum(log(diag(L))) - 0.5*(Ldiff'*Ldiff);

if nargout > 1
  gradz = zeros(1,3); 
  gradx = - L'\Ldiff;
  
  gradz(1) = gradx'*[1/a, 2*a*b*z(1), 2*a*b*z(1)]';
  gradz(2) = gradx'*[0, a, 0]';
  gradz(3) = gradx'*[0, 0, a]';
end