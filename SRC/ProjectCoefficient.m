function [Region_Coeffi] = ProjectCoefficient(x,Lable)
K = length(Lable);
[m,n] = size(x);
Region_Coeffi = zeros(m,K);
for i = 1:K
    index = find(Lable == i);
    Region_Coeffi(index,i) = x(index,1);
    %plot(Exprssion_Coeffi(:,i))
end
end