bsdsRoot='C:\Users\xiaofang\commondata\BSDS300\';
fid = fopen(fullfile(bsdsRoot,'Nsegs.txt'),'r');
Nimgs = 300; % number of images in BSDS300
[BSDS_INFO] = fscanf(fid,'%d %d \n',[2,Nimgs]);
fclose(fid);
global_graph_mode={'knn+L0','knn+L1','knn+LRR','L1+L0','L1+LRR','L0+LRR'};
for mm=1:6
mode=global_graph_mode{mm};
out_path= fullfile('result1', mode); mkdir(out_path)

nclusters=[2 10 30 40];
M=[];
for ncluster=1:length(nclusters)
    E_all=[];
for idxI =1:300
     fprintf('you are processing %d th image\n', idxI)
    img_name = int2str(BSDS_INFO(1,idxI)); %image_name1=[image_name1;str2num(img_name)];
    outname = fullfile(out_path,[img_name, '.mat']);
     load(outname);
     E_all=[E_all;E(ncluster,:)];
end
save(fullfile(out_path,[mode,int2str(nclusters(ncluster)) '.mat']),'E_all')
M=[M;mean(E_all)];
end
xlswrite([mode '_mean.xls'],M)
end