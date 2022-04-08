function y=bananafunND(x,ab,inverse)
%BANANAFUN banana shaped function
if nargin<3; inverse=0;end
a = ab(1); b = ab(2); % parameters
y = x;
n = size(x);
n = n(2);
if inverse
  y(:,1) = x(:,1)./a;
  y(:,2:n) = x(:,2:n).*a + a.*b.*(x(:,1).^2 + a^2);
else
  y(:,1) = a.*x(:,1);
  y(:,2:n) = x(:,2:n)./a - b.*(y(:,1).^2 + a^2);
end
