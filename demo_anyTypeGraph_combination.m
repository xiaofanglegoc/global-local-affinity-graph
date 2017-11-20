%% here is the code to generate the table IV for our paper
%  1,you could also extract only part of this code easily to port single
%  different graph tp generate the table I 
%  2, to get our global/local affinity graph, you could just keep the local
%  graph and LO graph, for example:
%                   W_L0 = compute_region_similarity_Sparse_penalty(feature,3,centroid,Area);
%                  W_GLG=assignGraphValue(W,W_L0,global_nodes,local_nodes);
% to combine different feature descriptor, 
% first just change the following 
%         feature=feat{k}.mlab;
% here is the list you could change:
% {'mlab';
%     'ch';
%     'lbp';
%     'siftbow100';
%     'siftbow150';
%     'siftbow200';
%     'siftbow300'};
% then you compute the affinity graph W_GLG_mlab,W_GLG_lbp, etc., 
% finally, you combine them according to the fusion equation in our paper
% Eq.11-12
% Please Note that you may not generate the exact performance listed in our paper,

clear all; close all;clc;
addpath 'others'
addpath 'evals'
addpath 'SRC'
addpath 'combineAnyTwoGraph'
%%%%%%%%%%%%%         set parameters for bipartite graph    %%%%%%%%%%%%%%%

para.alpha = 0.001; % affinity between pixels and superpixels
para.beta  =  20;   % scale factor in superpixel affinityI
para.nb = 1; % number of neighbors for superpixels
affine=false;
alpha = 20;
rho=1;
L=3; % the parameter for control the sparsity during solving the l0 problem  
%%%%%%%%%%%%%%%%%         read image     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bsdsRoot='/Users/xiaofangwang/Desktop/global-local-graph-for-image-segmentation-/BSD';
load_file='/Users/xiaofangwang/Desktop/global-local-graph-for-image-segmentation-/bsd_300_feat';
%load_file_seg_edges='C:\Users\xiaofang\commondata\BSDS300\BSD300_feature_set';
fid = fopen(fullfile('Nsegs.txt'),'r');
Nimgs = 300; % number of images in BSDS300
[BSDS_INFO] = fscanf(fid,'%d %d \n',[2,Nimgs]);
fclose(fid);
global_graph_mode={'knn+L0','L1+L0','L0+LRR'};
% you could set the number of cluster in the final result
% to obtain the best performance, in the paper, we  set it to [1:40]
nclusters=[2 10 30 40];
for idxI =1:300
    for m=1:length(global_graph_mode)
        mode=global_graph_mode{m};
        out_path= fullfile('result', mode);
        if ~exist(out_path)
            mkdir(out_path)
        end
        % locate image
        img_name = int2str(BSDS_INFO(1,idxI));
        img_loc = fullfile(bsdsRoot,'images','test',[img_name,'.jpg']);
        if ~exist(img_loc,'file')
            img_loc = fullfile(bsdsRoot,'images','train',[img_name,'.jpg']);
        end
        img = im2double(imread(img_loc)); [X,Y,~] = size(img);
        load_name=fullfile(load_file,[img_name '.mat']);
        load(load_name)
        %% construct graph
        Np = X*Y;
        Nsp = 0;
        for k = 1:length(seg)
            Nsp = Nsp + size(seg{k},2);
        end
        
        W_Y = sparse(Nsp,Nsp);
        edgesXY = [];
        j = 1;
        for k = 1:length(seg) % for each over-segmentation
            temp=feat{k}.shape;
            supixel_index=seg{k};
            centroid=temp(:,2:3);
            Area=temp(:,1);
            %%%%%%%%%%%%%%%%%%%%%%%%%
            % superpixel division according area
            [B,IX] = sort(Area);
            ord=diff(B,2);
            indx_T=find(abs(ord)>=abs(ord(end-5)));
            T=B(indx_T(ceil(length(indx_T)./2)));
            small=find(Area<300);
            large=find(Area>T);
            local_nodes=[small' large'];
            global_nodes=find(Area<=T & Area>=300);
            feature=feat{k}.mlab;%you could change the feature descriptor here
            %%%%%%%%%%%%%%%%%%%%%%%%%
            % first we construct the adjacent graph over all nodes
            w = makeweights(seg_edges{k},feature,para.beta);
            W_local = adjacency(seg_edges{k},w);
  
            %%%%%%%%%%%%%%%%%%%%%%%%%
            % assign local graph entries to fused new graph W, we will 
            % replace the nodes belongs to globla_nodes with value of
            % global graph value
            %W=zeros(size(feature,1),size(feature,1));
              W=W_local;
            %%%%%%%%%%%%%%%%%%%%%%%%%
            % randomly generate two set of supperpxiels from Medium set
            p=randperm(length(global_nodes));
            p1=p(1:floor(length(p)/2));
            p2=p(1+floor(length(p)/2):end);
            %%%%%%%%%%%%%%%%%%%%%%%%%
            % please choose two kins of global graph combination
            
            switch mode
                case 'knn+L0'
                    %%%%%%%%%%%%%%%%%%%%%%%%%
                    % knn graph
                    [points edges]=knn(feature,4);
                    w = makeweights(edges,feature,para.beta);
                    W_knn = adjacency(edges,w);
                    %%%%%%%%%%%%%%%%%%%%%%%%%
                    % L0 graph
                    W_L0 = compute_region_similarity_Sparse_penalty(feature,3,centroid,Area);
                    W=assignGraphValue(W,W_knn,p1);
                    W=assignGraphValue(W,W_L0,p2);
               case 'L1+L0'
                    %%%%%%%%%%%%%%%%%%%%%%%%%
                    % L1 graph
                    CMat = admmOutlier_mat_func(feature',affine,alpha);
                    NN = size(feature',2);
                    C = CMat(1:NN,:);
                    W_L1= BuildAdjacency(thrC(C,rho));
                    %%%%%%%%%%%%%%%%%%%%%%%%%
                    % L0 graph
                    W_L0 = compute_region_similarity_Sparse_penalty(feature,3,centroid,Area);
                    %W=assignGraphValue(W,W_L1,p1,local_nodes);
                    %W=assignGraphValue(W,W_L0,p2,local_nodes);
                    W=assignGraphValue(W,W_L1,p1);
                    W=assignGraphValue(W,W_L0,p2);
               
                case'L0+LRR'
                    %%%%%%%%%%%%%%%%%%%%%%%%%
                    % L0 graph
                    W_L0 = compute_region_similarity_Sparse_penalty(feature,3,centroid,Area);
                    %%%%%%%%%%%%%%%%%%%%%%%%%
                    % LRR graph
                    Z = solve_lrr(feature',0.18);
                    [U,S,V] = svd(Z,'econ');
                    S = diag(S);
                    r = sum(S>1e-4*S(1));
                    U = U(:,1:r);S = S(1:r);
                    U = U*diag(sqrt(S));
                    W_LRR = (U*U').^4;
                    W=assignGraphValue(W,W_L0,p1);
                    W=assignGraphValue(W,W_LRR,p2);
            end
            W=sparse(W);
            Nk = size(seg{k},2); % number of superpixels in over-segmentation k
            W_Y(j:j+Nk-1,j:j+Nk-1) = prune_knn(W,para.nb);
            
            % affinities between pixels and superpixels
            for i = 1:Nk
                idxp = seg{k}{i}; % pixel indices in superpixel i
                Nki = length(idxp);
                idxsp = j + zeros(Nki,1);
                edgesXY = [edgesXY; [idxp, idxsp]];
                j = j + 1;
            end
        end
        W_XY = sparse(edgesXY(:,1),edgesXY(:,2),para.alpha,Np,Nsp);
        % affinity between a superpixel and itself is set to be the maximum 1.
        W_Y(1:Nsp+1:end) = 1;
        B = [W_XY;W_Y];
        
        %% Transfer cut
        out_path_gt= fullfile('result',mode, img_name);mkdir(out_path_gt)
        [gt_imgs gt_cnt] = view_gt_segmentation(bsdsRoot,img,BSDS_INFO(1,idxI),out_path_gt,img_name,0);
        E=[];
        for ncluster=1:length(nclusters)
            label_img = Tcut(B,nclusters(ncluster),[X,Y]);
            % display the result
            view_segmentation(img,label_img(:),out_path,img_name,1);
            %% Evaluation and save result
            out_vals = eval_segmentation(label_img,gt_imgs);
            E=[E;nclusters(ncluster) out_vals.PRI out_vals.VoI out_vals.GCE out_vals.BDE];
        end
        outname = fullfile(out_path,[img_name, '.mat']);
        fprintf('saving %dth image to %s\n', idxI,outname);
        save('-v7',outname, 'E');
    end
end



