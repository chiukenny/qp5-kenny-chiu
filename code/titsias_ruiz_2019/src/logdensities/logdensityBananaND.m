function [out, gradz] = logdensityBananaND(z, mu, L, a, b) 
% z  : 1 x n matrix of 
% mu : 1 x n  mean vector
% L  : lower triangular Cholesky decomposition of the 
%      covariance matrix   


x = bananafunND(z,[a b],1);

n = length(x); 

diff = x - mu; 
diff = diff(:); 
   
Ldiff = L\diff;
   
out = - 0.5*n*log(2*pi) - sum(log(diag(L))) - 0.5*(Ldiff'*Ldiff);

if nargout > 1
  gradz = zeros(1,3); 
  gradx = - L'\Ldiff;
  
  gradh1 = ones(n,1)*2*a*b*z(1);
  gradh1(1) = 1/a;
  gradz(1) = gradx'*gradh1;
  for i = 2:n
    gradhi = zeros(n,1);
    gradhi(i) = a;
    gradz(i) = gradx'*gradhi;
  end
end