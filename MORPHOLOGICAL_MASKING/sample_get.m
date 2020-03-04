clc;
clear all;
%This code uses only JJ's results
%With correct phenotype information
dirpath='E:\50-plex\final\';
im_dapi=imread(strcat(dirpath,'S1_R1C1.tif'));
im_histone=imread(strcat(dirpath,'S1_R2C2.tif'));

%Oligodendrocytes
im_olig2=imread(strcat(dirpath,'S1_R1C9.tif'));
im_cnpase=imread(strcat(dirpath,'S1_R5C4.tif'));

im_dapi_histone=imadd(im_dapi,im_histone);

%CROP
x0=1;
y0=1;
h0=29398;
w0=43054;

%J's boxes
bbx_file='E:\50-plex\classification_results\classification_table.csv';

bbxs=get_bbxs_csv(bbx_file,w0,h0,x0,y0);

xc=bbxs.centroid_x;
yc=bbxs.centroid_y;

xmin=bbxs.xmin;
xmax=bbxs.xmax;

ymin=bbxs.ymin;
ymax=bbxs.ymax;

%X's

bbxs_1=readtable(strcat(dirpath,'merged_labelmask.txt'));
bbxs_1_array=table2array(bbxs_1);
bbxs_1_array_crop=bbxs_1_array;

bbxs_1_perim=bwperim(bbxs_1_array_crop);

bbxs_1_cc=bwconncomp(bbxs_1_array_crop,4);
bbxs_1_props=regionprops(bbxs_1_cc,'Centroid');
xyCentroids = vertcat(bbxs_1_props.Centroid);
xCentroids = xyCentroids(:,1);
yCentroids = xyCentroids(:,2);

half_w_n=50;
% w_n=50; %window size for nucleus segmentation
% w_s=100; %window size for soma segmentation
% w_p=200;
half_w_p=100;

W=size(im_dapi_histone,2);
H=size(im_dapi_histone,1);

all_cell_type=zeros(size(im_dapi_histone,1),size(im_dapi_histone,2),3);

all_micro_nucleus=zeros(size(im_dapi_histone));
all_micro_soma=zeros(size(im_dapi_histone));
all_micro_processes=zeros(size(im_dapi_histone));
all_micro_cytoplasm=zeros(size(im_dapi_histone));
all_micro_membrane=zeros(size(im_dapi_histone));
all_micro_whole_cell=zeros(size(im_dapi_histone));

array_props=regionprops(bbxs_1_array_crop,'Centroid','Image','BoundingBox');%
array_c=vertcat(array_props.Centroid);

dist_thresh=10;
d=100;

t=cputime;
boxes = struct('index',{},'cell_type',{},'nucleus_x1',{},'nucleus_x2',{},'nucleus_y1',{},'nucleus_y2',{},'nucleus_img',{},...
    'x_c',{},'y_c',{},'soma_x1',{},'soma_x2',{},'soma_y1',{},'soma_y2',{},'soma_img',{},...
    'cell_x1',{},'cell_x2',{},'cell_y1',{},'cell_y2',{},'processes_img',{},'membrane_img',{},'cytoplasm_img',{},'cell_img',{});
%parpool('local',16)

for i=1: length(xmin)
    % for i=length(xmin)-10:length(xmin)
    w_n=50;
    w_s=100;
    w_p=200;
    
    disp(strcat('Iteration: ',int2str(i)));
    row=find_closest_points_index_cnst(array_c(:,1),array_c(:,2),xc(i),yc(i));
    
    neun_val=bbxs.NeuN(i);
    s100_val=bbxs.S100(i);
    olig2_val=bbxs.Olig2(i);
    iba1_val=bbxs.Iba1(i);
    reca1_val=bbxs.RECA1(i);
    
    BB=array_props(row).BoundingBox;
    x1=ceil(BB(1));
    y1=ceil(BB(2));
    x2=x1+BB(3)-1;
    y2=y1+BB(4)-1;
    
    %lbl=bbxs_1_array(ceil((y1+y2)/2),ceil((x1+x2)/2));
    
    x_c=array_c(row,1);
    y_c=array_c(row,2);
    
    
    half_wd=round((x2-x1)/2);
    half_ht=round((y2-y1)/2);
    
    nucleus_mask1=zeros(w_p+1,w_p+1);
    nucleus_mask1(half_w_p-half_ht:half_w_p+y2-y1-half_ht,half_w_p-half_wd:half_w_p+x2-x1-half_wd)=array_props(row).Image;
    
      
  if olig2_val==1
        boxes(i).index=i;
        boxes(i).cell_type=0; % neuron
        disp('Starting Oligodendrocytes Segmentation');
        [soma_mask, processes, cytoplasm, membrane, whole_cell] = oligodendrocyte_segmentation_v4(im_cnpase,nucleus_mask1,x_c,y_c,w_n,w_p);
 end
end