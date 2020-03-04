clc;
clear all;
%This code uses only X's results
%With correct phenotype information
dirpath='E:\50-plex\final\';
im_dapi=imread(strcat(dirpath,'S1_R1C1.tif'));
im_histone=imread(strcat(dirpath,'S1_R2C2.tif'));

%Oligodendrocytes
im_olig2=imread(strcat(dirpath,'S1_R1C9.tif'));
im_cnpase=imread(strcat(dirpath,'S1_R5C4.tif'));

%Microglia
im_iba1=imread(strcat(dirpath,'S1_R1C5.tif'));

%Endothelials
im_gfp=imread(strcat(dirpath,'S1_R1C4.tif'));
im_reca1=imread(strcat(dirpath,'S1_R1C6.tif'));

%Neurons
im_neun=imread(strcat(dirpath,'S1_R2C4.tif'));
im_map2=imread(strcat(dirpath,'S1_R5C9.tif'));
im_nfh=imread(strcat(dirpath,'S1_R5C5.tif'));
im_nfm=imread(strcat(dirpath,'S1_R5C7.tif'));

%Astrocytes
im_s100=imread(strcat(dirpath,'S1_R3C5.tif'));
im_gfap=imread(strcat(dirpath,'S1_R3C3.tif'));
im_sox2=imread(strcat(dirpath,'S1_R4C5.tif'));

im_dapi_histone=imadd(im_dapi,im_histone);

if ~exist('astrocytes_results','dir')
    mkdir('astrocytes_results')
end

if ~exist('endothelial_results','dir')
    mkdir('endothelial_results')
end

if ~exist('microglia_results','dir')
    mkdir('microglia_results')
end

if ~exist('neuron_results','dir')
    mkdir('neuron_results')
end

if ~exist('none_results','dir')
    mkdir('none_results')
end

if ~exist('oligodendrocytes_results','dir')
    mkdir('oligodendrocytes_results')
end


%CROP
x0=1;
y0=1;
h0=29398;
w0=43054;
%

%J's boxes
bbx_file=input('Enter path to classification table','s');
if isempty(bbx_file)
    bbx_file='/project/roysam/50_plex/Set#1_S1/classification_results/classification_table.csv';
end

bbxs=get_bbxs_csv(bbx_file,w0,h0,x0,y0);

xc=bbxs.centroid_x;
yc=bbxs.centroid_y;

xmin=bbxs.xmin;
xmax=bbxs.xmax;

ymin=bbxs.ymin;
ymax=bbxs.ymax;

%X's

segmentation_mask_path=input('Enter path to segmentation masks','s');
if isempty(segmentation_mask_path)
    segmentation_mask_path=strcat('data/merged_labelmask.txt');
end

bbxs_1=readtable(segmentation_mask_path);
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

all_astro_nucleus=zeros(size(im_dapi_histone));
all_astro_soma=zeros(size(im_dapi_histone));
all_astro_processes=zeros(size(im_dapi_histone));
all_astro_cytoplasm=zeros(size(im_dapi_histone));
all_astro_membrane=zeros(size(im_dapi_histone));
all_astro_whole_cell=zeros(size(im_dapi_histone));

all_endo_nucleus=zeros(size(im_dapi_histone));
all_endo_soma=zeros(size(im_dapi_histone));
all_endo_processes=zeros(size(im_dapi_histone));
all_endo_cytoplasm=zeros(size(im_dapi_histone));
all_endo_membrane=zeros(size(im_dapi_histone));
all_endo_whole_cell=zeros(size(im_dapi_histone));

all_none_nucleus=zeros(size(im_dapi_histone));
all_none_soma=zeros(size(im_dapi_histone));
all_none_processes=zeros(size(im_dapi_histone));
all_none_cytoplasm=zeros(size(im_dapi_histone));
all_none_membrane=zeros(size(im_dapi_histone));
all_none_whole_cell=zeros(size(im_dapi_histone));

all_neuron_nucleus=zeros(size(im_dapi_histone));
all_neuron_soma=zeros(size(im_dapi_histone));
all_neuron_processes=zeros(size(im_dapi_histone));
all_neuron_cytoplasm=zeros(size(im_dapi_histone));
all_neuron_membrane=zeros(size(im_dapi_histone));
all_neuron_whole_cell=zeros(size(im_dapi_histone));

all_oligo_nucleus=zeros(size(im_dapi_histone));
all_oligo_soma=zeros(size(im_dapi_histone));
all_oligo_processes=zeros(size(im_dapi_histone));
all_oligo_cytoplasm=zeros(size(im_dapi_histone));
all_oligo_membrane=zeros(size(im_dapi_histone));
all_oligo_whole_cell=zeros(size(im_dapi_histone));

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
boxes = struct('cell_type',{},'nucleus_x1',{},'nucleus_x2',{},'nucleus_y1',{},'nucleus_y2',{},'nucleus_img',{},...
    'x_c',{},'y_c',{},'soma_x1',{},'soma_x2',{},'soma_y1',{},'soma_y2',{},'soma_img',{},...
    'cell_x1',{},'cell_x2',{},'cell_y1',{},'cell_y2',{},'processes_img',{},'membrane_img',{},'cytoplasm_img',{},'cell_img',{});

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
    
    if s100_val==1
        boxes(i).index=i;
        boxes(i).cell_type=1; %Astrocyte, S100 is astrocyte pan-specific
        disp('Starting Astrocyte Segmentation');
        [soma_mask, processes, cytoplasm, membrane, whole_cell] = astrocyte_segmentation_v5(im_s100,im_gfap,nucleus_mask1,x_c,y_c,w_n,w_s,w_p);
        
        boxes(i).nucleus_img = array_props(row).Image;
        boxes(i).x1 = x1;
        boxes(i).x2 = x2;
        boxes(i).y1 = y1;
        boxes(i).y2 = y2;
        boxes(i).x_c=x_c;
        boxes(i).y_c=y_c;
        
        disp('Copying the mask to frame');
        
        soma_props=regionprops(bwconncomp(soma_mask+nucleus_mask1,4)); %added coz soma mask might be split too
        soma_cs=vertcat(soma_props.Centroid);
        soma_cx=soma_cs(:,1);
        soma_cy=soma_cs(:,2);
        
        soma_idx = find_closest_points_index_cnst(soma_cx,soma_cy,size(soma_mask,1)/2,size(soma_mask,2)/2);
        %added coz soma mask might be split too
        soma_BB=soma_props(soma_idx).BoundingBox;
        
        boxes(i).soma_x1=x_c-half_w_p+ceil(soma_BB(1));
        boxes(i).soma_x2=x_c-half_w_p+soma_BB(3)-1+ceil(soma_BB(1));
        
        boxes(i).soma_y1=y_c-half_w_p+ceil(soma_BB(2));
        boxes(i).soma_y2=y_c-half_w_p+soma_BB(4)-1+ceil(soma_BB(2));
        
        boxes(i).soma_img=imcrop(soma_mask,[ceil(soma_BB(1)),ceil(soma_BB(2)),soma_BB(3)-1,soma_BB(4)-1]);
        
        whole_cell_props=regionprops(bwconncomp(whole_cell,4));
        whole_cell_cs=vertcat(whole_cell_props.Centroid);
        whole_cell_cx=whole_cell_cs(:,1);
        whole_cell_cy=whole_cell_cs(:,2);
        
        whole_cell_idx = find_closest_points_index_cnst(whole_cell_cx,whole_cell_cy,w_p/2,w_p/2);
        
        whole_cell_BB=whole_cell_props(whole_cell_idx).BoundingBox;
        
        boxes(i).cell_x1=x_c-half_w_p+ceil(whole_cell_BB(1));
        boxes(i).cell_x2=x_c-half_w_p+ceil(whole_cell_BB(1))+whole_cell_BB(3)-1;
        
        boxes(i).cell_y1=y_c-half_w_p+ceil(whole_cell_BB(2));
        boxes(i).cell_y2=y_c-half_w_p+ceil(whole_cell_BB(2))+whole_cell_BB(4)-1;
        
        boxes(i).cell_img=imcrop(whole_cell,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
        boxes(i).processes_img=imcrop(processes,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
        boxes(i).cytoplasm_img=imcrop(cytoplasm,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
        boxes(i).membrane_img=imcrop(membrane,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
        
        disp('Saving Astrocyte Segmentation');
        %yellow
        all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,1)=255;
        all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,2)=255;
        all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,3)=0;
        
        all_astro_nucleus(boxes(i).y1:boxes(i).y2,boxes(i).x1:boxes(i).x2)=boxes(i).nucleus_img+all_astro_nucleus(boxes(i).y1:boxes(i).y2,boxes(i).x1:boxes(i).x2);
        all_astro_soma(boxes(i).soma_y1:boxes(i).soma_y2,boxes(i).soma_x1:boxes(i).soma_x2)=boxes(i).soma_img+all_astro_soma(boxes(i).soma_y1:boxes(i).soma_y2,boxes(i).soma_x1:boxes(i).soma_x2);
        all_astro_whole_cell(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2)=boxes(i).cell_img+all_astro_whole_cell(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2);
        all_astro_processes(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2)=boxes(i).processes_img+all_astro_processes(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2);
        all_astro_cytoplasm(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2)=boxes(i).cytoplasm_img+all_astro_cytoplasm(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2);
        all_astro_membrane(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2)=boxes(i).membrane_img+all_astro_membrane(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2);
    else
        if  iba1_val==1
            boxes(i).index=i;
            boxes(i).cell_type=2; % microglia
            disp('Starting Microglia Segmentation');
            [soma_mask, processes, cytoplasm, membrane, whole_cell] = microglia_segmentation_v3(im_iba1,nucleus_mask1,x_c,y_c,w_n,w_s,w_p);
            boxes(i).nucleus_img = array_props(row).Image;
            boxes(i).x1 = x1;
            boxes(i).x2 = x2;
            boxes(i).y1 = y1;
            boxes(i).y2 = y2;
            boxes(i).x_c=x_c;
            boxes(i).y_c=y_c;
            
            disp('Copying the mask to frame');
            
            soma_props=regionprops(bwconncomp(soma_mask+nucleus_mask1,4)); %added coz soma mask might be split too
            soma_cs=vertcat(soma_props.Centroid);
            soma_cx=soma_cs(:,1);
            soma_cy=soma_cs(:,2);
            
            soma_idx = find_closest_points_index_cnst(soma_cx,soma_cy,size(soma_mask,1)/2,size(soma_mask,2)/2);
            %added coz soma mask might be split too
            soma_BB=soma_props(soma_idx).BoundingBox;
            
            boxes(i).soma_x1=x_c-half_w_p+ceil(soma_BB(1));
            boxes(i).soma_x2=x_c-half_w_p+soma_BB(3)-1+ceil(soma_BB(1));
            
            boxes(i).soma_y1=y_c-half_w_p+ceil(soma_BB(2));
            boxes(i).soma_y2=y_c-half_w_p+soma_BB(4)-1+ceil(soma_BB(2));
            
            boxes(i).soma_img=imcrop(soma_mask,[ceil(soma_BB(1)),ceil(soma_BB(2)),soma_BB(3)-1,soma_BB(4)-1]);
            
            whole_cell_props=regionprops(bwconncomp(whole_cell,4));
            whole_cell_cs=vertcat(whole_cell_props.Centroid);
            whole_cell_cx=whole_cell_cs(:,1);
            whole_cell_cy=whole_cell_cs(:,2);
            
            whole_cell_idx = find_closest_points_index_cnst(whole_cell_cx,whole_cell_cy,w_p/2,w_p/2);
            
            whole_cell_BB=whole_cell_props(whole_cell_idx).BoundingBox;
            
            boxes(i).cell_x1=x_c-half_w_p+ceil(whole_cell_BB(1));
            boxes(i).cell_x2=x_c-half_w_p+ceil(whole_cell_BB(1))+whole_cell_BB(3)-1;
            
            boxes(i).cell_y1=y_c-half_w_p+ceil(whole_cell_BB(2));
            boxes(i).cell_y2=y_c-half_w_p+ceil(whole_cell_BB(2))+whole_cell_BB(4)-1;
            
            boxes(i).cell_img=imcrop(whole_cell,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
            boxes(i).processes_img=imcrop(processes,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
            boxes(i).cytoplasm_img=imcrop(cytoplasm,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
            boxes(i).membrane_img=imcrop(membrane,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
            
            disp('Saving Microglia Segmentation');
            
            all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,1)=0;
            all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,2)=255;
            all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,3)=0;
            
            all_micro_nucleus(boxes(i).y1:boxes(i).y2,boxes(i).x1:boxes(i).x2)=boxes(i).nucleus_img+all_micro_nucleus(boxes(i).y1:boxes(i).y2,boxes(i).x1:boxes(i).x2);
            all_micro_soma(boxes(i).soma_y1:boxes(i).soma_y2,boxes(i).soma_x1:boxes(i).soma_x2)=boxes(i).soma_img+all_micro_soma(boxes(i).soma_y1:boxes(i).soma_y2,boxes(i).soma_x1:boxes(i).soma_x2);
            all_micro_whole_cell(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2)=boxes(i).cell_img+all_micro_whole_cell(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2);
            all_micro_processes(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2)=boxes(i).processes_img+all_micro_processes(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2);
            all_micro_cytoplasm(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2)=boxes(i).cytoplasm_img+all_micro_cytoplasm(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2);
            all_micro_membrane(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2)=boxes(i).membrane_img+all_micro_membrane(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2);
        else
            if olig2_val==1
                boxes.index(i)=i;
                boxes(i).cell_type=4; % oligodendrocytes
                disp('Starting Oligodendrocyte Segmentation');
                [soma_mask, processes, cytoplasm, membrane, whole_cell] = oligodendrocyte_segmentation_v4(im_cnpase,nucleus_mask1,x_c,y_c,w_n,w_p);
                
                boxes(i).nucleus_img = array_props(row).Image;
                boxes(i).x1 = x1;
                boxes(i).x2 = x2;
                boxes(i).y1 = y1;
                boxes(i).y2 = y2;
                boxes(i).x_c=x_c;
                boxes(i).y_c=y_c;
                
                disp('Copying the mask to frame');
                
                soma_props=regionprops(bwconncomp(soma_mask+nucleus_mask1,4)); %added coz soma mask might be split too
                soma_cs=vertcat(soma_props.Centroid);
                soma_cx=soma_cs(:,1);
                soma_cy=soma_cs(:,2);
                
                soma_idx = find_closest_points_index_cnst(soma_cx,soma_cy,size(soma_mask,1)/2,size(soma_mask,2)/2);
                %added coz soma mask might be split too
                soma_BB=soma_props(soma_idx).BoundingBox;
                
                boxes(i).soma_x1=x_c-half_w_p+ceil(soma_BB(1));
                boxes(i).soma_x2=x_c-half_w_p+soma_BB(3)-1+ceil(soma_BB(1));
                
                boxes(i).soma_y1=y_c-half_w_p+ceil(soma_BB(2));
                boxes(i).soma_y2=y_c-half_w_p+soma_BB(4)-1+ceil(soma_BB(2));
                
                boxes(i).soma_img=imcrop(soma_mask,[ceil(soma_BB(1)),ceil(soma_BB(2)),soma_BB(3)-1,soma_BB(4)-1]);
                
                whole_cell_props=regionprops(bwconncomp(whole_cell,4));
                whole_cell_cs=vertcat(whole_cell_props.Centroid);
                whole_cell_cx=whole_cell_cs(:,1);
                whole_cell_cy=whole_cell_cs(:,2);
                
                whole_cell_idx = find_closest_points_index_cnst(whole_cell_cx,whole_cell_cy,w_p/2,w_p/2);
                
                whole_cell_BB=whole_cell_props(whole_cell_idx).BoundingBox;
                
                boxes(i).cell_x1=x_c-half_w_p+ceil(whole_cell_BB(1));
                boxes(i).cell_x2=x_c-half_w_p+ceil(whole_cell_BB(1))+whole_cell_BB(3)-1;
                
                boxes(i).cell_y1=y_c-half_w_p+ceil(whole_cell_BB(2));
                boxes(i).cell_y2=y_c-half_w_p+ceil(whole_cell_BB(2))+whole_cell_BB(4)-1;
                
                boxes(i).cell_img=imcrop(whole_cell,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
                boxes(i).processes_img=imcrop(processes,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
                boxes(i).cytoplasm_img=imcrop(cytoplasm,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
                boxes(i).membrane_img=imcrop(membrane,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
                
                disp('Saving Oligodendrocytes Segmentation');
                
                all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,1)=255;
                all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,2)=255;
                all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,3)=255;
                
                
                all_oligo_nucleus(boxes(i).y1:boxes(i).y2,boxes(i).x1:boxes(i).x2)=boxes(i).nucleus_img+all_oligo_nucleus(boxes(i).y1:boxes(i).y2,boxes(i).x1:boxes(i).x2);
                all_oligo_soma(boxes(i).soma_y1:boxes(i).soma_y2,boxes(i).soma_x1:boxes(i).soma_x2)=boxes(i).soma_img+all_oligo_soma(boxes(i).soma_y1:boxes(i).soma_y2,boxes(i).soma_x1:boxes(i).soma_x2);
                all_oligo_whole_cell(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2)=boxes(i).cell_img+all_oligo_whole_cell(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2);
                all_oligo_processes(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2)=boxes(i).processes_img+all_oligo_processes(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2);
                all_oligo_cytoplasm(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2)=boxes(i).cytoplasm_img+all_oligo_cytoplasm(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2);
                all_oligo_membrane(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2)=boxes(i).membrane_img+all_oligo_membrane(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2);
                
            else
                if neun_val==1
                    boxes(i).index=i;
                    boxes(i).cell_type=0; % neuron
                    disp('Starting Neuron Segmentation');
                    [soma_mask, processes, cytoplasm, membrane, whole_cell] = neuron_segmentation_v4(im_neun,im_map2,nucleus_mask1,x_c,y_c,w_n,w_s,w_p);
                    
                    boxes(i).nucleus_img = array_props(row).Image;
                    
                    boxes(i).x1 = x1;
                    boxes(i).x2 = x2;
                    boxes(i).y1 = y1;
                    boxes(i).y2 = y2;
                    boxes(i).x_c=x_c;
                    boxes(i).y_c=y_c;
                    
                    disp('Copying the mask to frame');
                    
                    soma_props=regionprops(bwconncomp(soma_mask+nucleus_mask1,4)); %added coz soma mask might be split too
                    soma_cs=vertcat(soma_props.Centroid);
                    soma_cx=soma_cs(:,1);
                    soma_cy=soma_cs(:,2);
                    
                    soma_idx = find_closest_points_index_cnst(soma_cx,soma_cy,size(soma_mask,1)/2,size(soma_mask,2)/2);
                    %added coz soma mask might be split too
                    soma_BB=soma_props(soma_idx).BoundingBox;
                    
                    boxes(i).soma_x1=x_c-half_w_p+ceil(soma_BB(1));
                    boxes(i).soma_x2=x_c-half_w_p+soma_BB(3)-1+ceil(soma_BB(1));
                    
                    boxes(i).soma_y1=y_c-half_w_p+ceil(soma_BB(2));
                    boxes(i).soma_y2=y_c-half_w_p+soma_BB(4)-1+ceil(soma_BB(2));
                    
                    boxes(i).soma_img=imcrop(soma_mask,[ceil(soma_BB(1)),ceil(soma_BB(2)),soma_BB(3)-1,soma_BB(4)-1]);
                    
                    whole_cell_props=regionprops(bwconncomp(whole_cell,4));
                    whole_cell_cs=vertcat(whole_cell_props.Centroid);
                    whole_cell_cx=whole_cell_cs(:,1);
                    whole_cell_cy=whole_cell_cs(:,2);
                    
                    whole_cell_idx = find_closest_points_index_cnst(whole_cell_cx,whole_cell_cy,w_p/2,w_p/2);
                    
                    whole_cell_BB=whole_cell_props(whole_cell_idx).BoundingBox;
                    
                    boxes(i).cell_x1=x_c-half_w_p+ceil(whole_cell_BB(1));
                    boxes(i).cell_x2=x_c-half_w_p+ceil(whole_cell_BB(1))+whole_cell_BB(3)-1;
                    
                    boxes(i).cell_y1=y_c-half_w_p+ceil(whole_cell_BB(2));
                    boxes(i).cell_y2=y_c-half_w_p+ceil(whole_cell_BB(2))+whole_cell_BB(4)-1;
                    
                    boxes(i).cell_img=imcrop(whole_cell,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
                    boxes(i).processes_img=imcrop(processes,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
                    boxes(i).cytoplasm_img=imcrop(cytoplasm,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
                    boxes(i).membrane_img=imcrop(membrane,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
                    
                    disp('Saving Neuron Segmentation');
                    
                    all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,1)=0;
                    all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,2)=0;
                    all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,3)=255;
                    
                    all_neuron_nucleus(boxes(i).y1:boxes(i).y2,boxes(i).x1:boxes(i).x2)=boxes(i).nucleus_img+all_neuron_nucleus(boxes(i).y1:boxes(i).y2,boxes(i).x1:boxes(i).x2);
                    all_neuron_soma(boxes(i).soma_y1:boxes(i).soma_y2,boxes(i).soma_x1:boxes(i).soma_x2)=boxes(i).soma_img+all_neuron_soma(boxes(i).soma_y1:boxes(i).soma_y2,boxes(i).soma_x1:boxes(i).soma_x2);
                    all_neuron_whole_cell(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2)=boxes(i).cell_img+all_neuron_whole_cell(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2);
                    all_neuron_processes(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2)=boxes(i).processes_img+all_neuron_processes(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2);
                    all_neuron_cytoplasm(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2)=boxes(i).cytoplasm_img+all_neuron_cytoplasm(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2);
                    all_neuron_membrane(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2)=boxes(i).membrane_img+all_neuron_membrane(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2);
                    
                else
                    if s100_val==0 && neun_val==0 && iba1_val==0 && reca1_val==0 && olig2_val==0
                        
                        boxes(i).index=i;
                        boxes(i).cell_type=5; % none
                        disp('Handling the None type cell')
                        [soma_mask, processes, cytoplasm, membrane, whole_cell] = none_segmentation_v3(nucleus_mask1,w_n,w_p);
                        
                        boxes(i).nucleus_img = array_props(row).Image;
                        boxes(i).x1 = x1;
                        boxes(i).x2 = x2;
                        boxes(i).y1 = y1;
                        boxes(i).y2 = y2;
                        boxes(i).x_c=x_c;
                        boxes(i).y_c=y_c;
                        
                        disp('Copying the mask to frame');
                        
                        soma_props=regionprops(bwconncomp(soma_mask+nucleus_mask1,4)); %added coz soma mask might be split too
                        soma_cs=vertcat(soma_props.Centroid);
                        soma_cx=soma_cs(:,1);
                        soma_cy=soma_cs(:,2);
                        
                        soma_idx = find_closest_points_index_cnst(soma_cx,soma_cy,size(soma_mask,1)/2,size(soma_mask,2)/2);
                        %added coz soma mask might be split too
                        soma_BB=soma_props(soma_idx).BoundingBox;
                        
                        boxes(i).soma_x1=x_c-half_w_p+ceil(soma_BB(1));
                        boxes(i).soma_x2=x_c-half_w_p+soma_BB(3)-1+ceil(soma_BB(1));
                        
                        boxes(i).soma_y1=y_c-half_w_p+ceil(soma_BB(2));
                        boxes(i).soma_y2=y_c-half_w_p+soma_BB(4)-1+ceil(soma_BB(2));
                        
                        boxes(i).soma_img=imcrop(soma_mask,[ceil(soma_BB(1)),ceil(soma_BB(2)),soma_BB(3)-1,soma_BB(4)-1]);
                        
                        whole_cell_props=regionprops(bwconncomp(whole_cell,4));
                        whole_cell_cs=vertcat(whole_cell_props.Centroid);
                        whole_cell_cx=whole_cell_cs(:,1);
                        whole_cell_cy=whole_cell_cs(:,2);
                        
                        whole_cell_idx = find_closest_points_index_cnst(whole_cell_cx,whole_cell_cy,w_p/2,w_p/2);
                        
                        whole_cell_BB=whole_cell_props(whole_cell_idx).BoundingBox;
                        
                        boxes(i).cell_x1=x_c-half_w_p+ceil(whole_cell_BB(1));
                        boxes(i).cell_x2=x_c-half_w_p+ceil(whole_cell_BB(1))+whole_cell_BB(3)-1;
                        
                        boxes(i).cell_y1=y_c-half_w_p+ceil(whole_cell_BB(2));
                        boxes(i).cell_y2=y_c-half_w_p+ceil(whole_cell_BB(2))+whole_cell_BB(4)-1;
                        
                        boxes(i).cell_img=imcrop(whole_cell,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
                        boxes(i).processes_img=imcrop(processes,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
                        boxes(i).cytoplasm_img=imcrop(cytoplasm,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
                        boxes(i).membrane_img=imcrop(membrane,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
                        
                        disp('Saving None Segmentation');
                        
                        all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,1)=0;
                        all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,2)=255;
                        all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,3)=255;
                        
                        all_none_nucleus(boxes(i).y1:boxes(i).y2,boxes(i).x1:boxes(i).x2)=boxes(i).nucleus_img+all_none_nucleus(boxes(i).y1:boxes(i).y2,boxes(i).x1:boxes(i).x2);
                        all_none_soma(boxes(i).soma_y1:boxes(i).soma_y2,boxes(i).soma_x1:boxes(i).soma_x2)=boxes(i).soma_img+all_none_soma(boxes(i).soma_y1:boxes(i).soma_y2,boxes(i).soma_x1:boxes(i).soma_x2);
                        all_none_whole_cell(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2)=boxes(i).cell_img+all_none_whole_cell(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2);
                        all_none_processes(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2)=boxes(i).processes_img+all_none_processes(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2);
                        all_none_cytoplasm(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2)=boxes(i).cytoplasm_img+all_none_cytoplasm(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2);
                        all_none_membrane(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2)=boxes(i).membrane_img+all_none_membrane(boxes(i).cell_y1:boxes(i).cell_y2,boxes(i).cell_x1:boxes(i).cell_x2);
                        
                    end
                    
                end
            end
        end
    end
    
    
    if mod(i,10000)==0
        
        imwrite(all_cell_type,strcat('astrocytes_results/astro_cell_type_',int2str(i),'.tif'));
        imwrite(im2uint8(all_astro_nucleus),strcat('astrocytes_results/astro_nucleus_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_astro_soma),strcat('astrocytes_results/astro_soma_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_astro_processes),strcat('astrocytes_results/astro_processes_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_astro_membrane),strcat('astrocytes_results/astro_membrane_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_astro_cytoplasm),strcat('astrocytes_results/astro_cytoplasm_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_astro_whole_cell),strcat('astrocytes_results/astro_whole_cell_mask_',int2str(i),'.tif'));
        save(strcat('astrocytes_results/astro_reconstruction_info_',int2str(i),'.mat'),'boxes');
        
        imwrite(all_cell_type,strcat('microglia_results/micro_cell_type_',int2str(i),'.tif'));
        imwrite(im2uint8(all_micro_nucleus),strcat('microglia_results/micro_nucleus_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_micro_soma),strcat('microglia_results/micro_soma_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_micro_processes),strcat('microglia_results/micro_processes_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_micro_membrane),strcat('microglia_results/micro_membrane_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_micro_cytoplasm),strcat('microglia_results/micro_cytoplasm_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_micro_whole_cell),strcat('microglia_results/micro_whole_cell_mask_',int2str(i),'.tif'));
        save(strcat('microglia_results/micro_reconstruction_info_',int2str(i),'.mat'),'boxes');
        
        imwrite(all_cell_type,strcat('oligodendrocytes_results/oligo_cell_type_',int2str(i),'.tif'));
        imwrite(im2uint8(all_oligo_nucleus),strcat('oligodendrocytes_results/oligo_nucleus_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_oligo_soma),strcat('oligodendrocytes_results/oligo_soma_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_oligo_processes),strcat('oligodendrocytes_results/oligo_processes_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_oligo_membrane),strcat('oligodendrocytes_results/oligo_membrane_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_oligo_cytoplasm),strcat('oligodendrocytes_results/oligo_cytoplasm_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_oligo_whole_cell),strcat('oligodendrocytes_results/oligo_whole_cell_mask_',int2str(i),'.tif'));
        save(strcat('oligodendrocytes_results/oligo_reconstruction_info_',int2str(i),'.mat'),'boxes');
        
        imwrite(all_cell_type,strcat('neuron_results_new/neuron_cell_type_',int2str(i),'.tif'));
        imwrite(im2uint8(all_neuron_nucleus),strcat('neuron_results_new/neuron_nucleus_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_neuron_soma),strcat('neuron_results_new/neuron_soma_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_neuron_processes),strcat('neuron_results_new/neuron_processes_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_neuron_membrane),strcat('neuron_results_new/neuron_membrane_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_neuron_cytoplasm),strcat('neuron_results_new/neuron_cytoplasm_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_neuron_whole_cell),strcat('neuron_results_new/neuron_whole_cell_mask_',int2str(i),'.tif'));
        save(strcat('neuron_results_new/neuron_reconstruction_info_',int2str(i),'.mat'),'boxes');
        
        imwrite(all_cell_type,strcat('none_results/none_cell_type_',int2str(i),'.tif'));
        imwrite(im2uint8(all_none_nucleus),strcat('none_results/none_nucleus_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_none_soma),strcat('none_results/none_soma_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_none_processes),strcat('none_results/none_processes_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_none_membrane),strcat('none_results/none_membrane_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_none_cytoplasm),strcat('none_results/none_cytoplasm_mask_',int2str(i),'.tif'));
        imwrite(im2uint8(all_none_whole_cell),strcat('none_results/none_whole_cell_mask_',int2str(i),'.tif'));
        save(strcat('none_results/none_reconstruction_info_',int2str(i),'.mat'),'boxes');
    end
    
end

imwrite(all_cell_type,strcat('astrocytes_results/astro_cell_type.tif'));
imwrite(im2uint8(all_astro_nucleus),strcat('astrocytes_results/astro_nucleus_mask.tif'));
imwrite(im2uint8(all_astro_soma),strcat('astrocytes_results/astro_soma_mask.tif'));
imwrite(im2uint8(all_astro_processes),strcat('astrocytes_results/astro_processes_mask.tif'));
imwrite(im2uint8(all_astro_membrane),strcat('astrocytes_results/astro_membrane_mask.tif'));
imwrite(im2uint8(all_astro_cytoplasm),strcat('astrocytes_results/astro_cytoplasm_mask.tif'));
imwrite(im2uint8(all_astro_whole_cell),strcat('astrocytes_results/astro_whole_cell_mask.tif'));
save(strcat('astrocytes_results/astro_reconstruction_info_',int2str(i),'.mat'),'boxes');

imwrite(all_cell_type,strcat('microglia_results/micro_cell_type_',int2str(i),'.tif'));
imwrite(im2uint8(all_micro_nucleus),strcat('microglia_results/micro_nucleus_mask_',int2str(i),'.tif'));
imwrite(im2uint8(all_micro_soma),strcat('microglia_results/micro_soma_mask_',int2str(i),'.tif'));
imwrite(im2uint8(all_micro_processes),strcat('microglia_results/micro_processes_mask_',int2str(i),'.tif'));
imwrite(im2uint8(all_micro_membrane),strcat('microglia_results/micro_membrane_mask_',int2str(i),'.tif'));
imwrite(im2uint8(all_micro_cytoplasm),strcat('microglia_results/micro_cytoplasm_mask_',int2str(i),'.tif'));
imwrite(im2uint8(all_micro_whole_cell),strcat('microglia_results/micro_whole_cell_mask_',int2str(i),'.tif'));
save(strcat('microglia_results/micro_reconstruction_info_',int2str(i),'.mat'),'boxes');

imwrite(all_cell_type,strcat('oligodendrocytes_results/oligo_cell_type_',int2str(i),'.tif'));
imwrite(im2uint8(all_oligo_nucleus),strcat('oligodendrocytes_results/oligo_nucleus_mask_',int2str(i),'.tif'));
imwrite(im2uint8(all_oligo_soma),strcat('oligodendrocytes_results/oligo_soma_mask_',int2str(i),'.tif'));
imwrite(im2uint8(all_oligo_processes),strcat('oligodendrocytes_results/oligo_processes_mask_',int2str(i),'.tif'));
imwrite(im2uint8(all_oligo_membrane),strcat('oligodendrocytes_results/oligo_membrane_mask_',int2str(i),'.tif'));
imwrite(im2uint8(all_oligo_cytoplasm),strcat('oligodendrocytes_results/oligo_cytoplasm_mask_',int2str(i),'.tif'));
imwrite(im2uint8(all_oligo_whole_cell),strcat('oligodendrocytes_results/oligo_whole_cell_mask_',int2str(i),'.tif'));
save(strcat('oligodendrocytes_results/oligo_reconstruction_info_',int2str(i),'.mat'),'boxes');

imwrite(all_cell_type,strcat('neuron_results_new/neuron_cell_type_',int2str(i),'.tif'));
imwrite(im2uint8(all_neuron_nucleus),strcat('neuron_results_new/neuron_nucleus_mask_',int2str(i),'.tif'));
imwrite(im2uint8(all_neuron_soma),strcat('neuron_results_new/neuron_soma_mask_',int2str(i),'.tif'));
imwrite(im2uint8(all_neuron_processes),strcat('neuron_results_new/neuron_processes_mask_',int2str(i),'.tif'));
imwrite(im2uint8(all_neuron_membrane),strcat('neuron_results_new/neuron_membrane_mask_',int2str(i),'.tif'));
imwrite(im2uint8(all_neuron_cytoplasm),strcat('neuron_results_new/neuron_cytoplasm_mask_',int2str(i),'.tif'));
imwrite(im2uint8(all_neuron_whole_cell),strcat('neuron_results_new/neuron_whole_cell_mask_',int2str(i),'.tif'));
save(strcat('neuron_results_new/neuron_reconstruction_info_',int2str(i),'.mat'),'boxes');

imwrite(all_cell_type,strcat('none_results/none_cell_type_',int2str(i),'.tif'));
imwrite(im2uint8(all_none_nucleus),strcat('none_results/none_nucleus_mask_',int2str(i),'.tif'));
imwrite(im2uint8(all_none_soma),strcat('none_results/none_soma_mask_',int2str(i),'.tif'));
imwrite(im2uint8(all_none_processes),strcat('none_results/none_processes_mask_',int2str(i),'.tif'));
imwrite(im2uint8(all_none_membrane),strcat('none_results/none_membrane_mask_',int2str(i),'.tif'));
imwrite(im2uint8(all_none_cytoplasm),strcat('none_results/none_cytoplasm_mask_',int2str(i),'.tif'));
imwrite(im2uint8(all_none_whole_cell),strcat('none_results/none_whole_cell_mask_',int2str(i),'.tif'));
save(strcat('none_results/none_reconstruction_info_',int2str(i),'.mat'),'boxes');
