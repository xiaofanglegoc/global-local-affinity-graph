function W = compute_region_similarity_Sparse_penalty(histogram,L,Centroid,Area)

histogram = histogram';
[m,n] = size(histogram);
d = sqrt( sum(histogram.^2,1) ); d(d<eps)=1;
histogram = histogram ./ repmat( d, m,1 );

W_Y = [];
Dictionary = [];
param.L=5; % not more than 10 non-zeros coefficients
param.eps=0.1; % squared norm of the residual should be less than 0.1
param.numThreads=-1; % number of processors/cores to use; the default choice is -1
% and uses all the cores of the machine
for i = 1:n
    y = histogram(:,i);
    Dictionary = histogram;
    Dictionary(:,i) = zeros(m,1);
    %x=mexOMP(Dictionary,y,param);
    %     res = norm(y - Dictionary(:,2)*x(1,2))
    x = OMP(Dictionary,y,L);
    ind_x=find(x~=0);
    C_neg=Centroid(ind_x,:);
    C_cur=Centroid(i,:);
    D_C=sqrt(sum((repmat(C_cur,size(C_neg,1),1)-C_neg).*(repmat(C_cur,size(C_neg,1),1)-C_neg),2));
    
    ind_d=find(D_C);
    if numel(ind_d)>1
        bb=find(ind_d==find(D_C==max(D_C)));
        ind_d(bb)=[];
        %ind_x(ind_x(find(D_C==max(D_C))))=[];
        %     [~,idx]=sort(D_C,'descend');
        ind_new=ind_x(ind_d);
         Dictionary_new=histogram(:,ind_new);
%         Dictionary_new=Dictionary;
%         for kk=1:numel(ind_new)
%             Dictionary_new(:,ind_new(kk))=zeros(m,1);
%         end
        %  Dictionary(:,i) = zeros(m,1);
          AreaWeights=ones(numel(ind_new),1);
              ind_area=find(Area(ind_new)<200);
if numel(ind_area)>1
%     ind_area=find(Area(ind_new));
     AreaWeights(ind_area)=2;
end    
%         AreaWeights=1-AreaWeights./sum(AreaWeights);
         x1 = pinv(Dictionary_new)*y.*AreaWeights;
%         x1 = OMP(Dictionary_new,y,2);
%         x0=x1;
        %     x=sparse(x);
        x0=sparse(n,1);
        x0(ind_new)=x1(find(x1~=0));
    else
        x0=x;
    end
    R_Lable = [1:n];
    [Region_Coeffi] = ProjectCoefficient(x0,R_Lable);
    Region = Dictionary*Region_Coeffi;
    Error_Region = repmat(y,1,n) - Region;
    similarity = [];
    similarity = sqrt( sum(Error_Region.^2,1) );
    similarity = (1 - similarity);
    similarity(i) = 0;
    index = find(similarity<0.00001);
    similarity(index) = 0;
    W_Y =[W_Y; similarity];
end
W = W_Y + W_Y';
index = find(W>1);
W(index) =  W(index)/2;


% WWY = symetricy_WY(WY);
% W_Y = (W_Y + W_Y')/2;

end