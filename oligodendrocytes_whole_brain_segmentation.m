function [] = oligodendrocytes_whole_brain_segmentation(varargin)
    addpath('MORPHOLOGICAL_MASKING');
    defaultDAPIpath='E:\50-plex\final\S1_R1C1.tif';
    defaultHistonepath='E:\50-plex\final\S1_R2C2.tif';
    defaultOlig2path='E:\50-plex\final\S1_R1C5.tif';
    defaultCNpasepath='E:\50-plex\final\S1_R5C4.tif';
    defaultOUTPUTdir='oligodendrocytes_results';
    defaultCLASSIFICATIONtable='E:\50-plex\classification_results\classification_table.csv';
    defaultSEGMENTATIONmasks='data/merged_labelmask.txt';
    p=inputParser;

    addParameter(p,'DAPI_PATH',defaultDAPIpath,@ischar);
    addParameter(p,'HISTONE_PATH',defaultHistonepath,@ischar);
    addParameter(p,'OLIG2_PATH',defaultOlig2path,@ischar);
    addParameter(p,'CNPASE_PATH',defaultCNpasepath,@ischar);
    addParameter(p,'OUTPUT_DIR',defaultOUTPUTdir,@ischar);
    addParameter(p,'CLASSIFICATION_table_path',defaultCLASSIFICATIONtable,@ischar);
    addParameter(p,'SEGMENTATION_masks',defaultSEGMENTATIONmasks,@ischar);

    parse(p,varargin{:});
    
    im_dapi_histone=imadd(imread(fullfile(p.Results.DAPI_PATH)),imread(fullfile(p.Results.HISTONE_PATH)));
    %im_olig2=imread(p.Results.OLIG2_PATH);
    im_cnpase=imread(p.Results.CNPASE_PATH);
    
    %create output dir
    if ~exist(p.Results.OUTPUT_DIR,'dir')
        mkdir(p.Results.OUTPUT_DIR)
    end

    %load boundingboxes
    x0=0;y0=0;
    W=size(im_dapi_histone,2);
    H=size(im_dapi_histone,1);

    bbxs=get_bbxs_csv(p.Results.CLASSIFICATION_table_path,W,H,x0,y0);
    xc=bbxs.centroid_x;
    yc=bbxs.centroid_y;
    IDs=bbxs.ID;
    
    %load segmentation masks
    seg_masks=readtable(p.Results.SEGMENTATION_masks);
    seg_masks_array=table2array(seg_masks);

    seg_masks_props=regionprops(seg_masks_array,'Centroid','Image','BoundingBox');
    seg_masks_c=vertcat(seg_masks_props.Centroid);
    
    half_w_p=100;
    w_n=50;
    w_p=200;

    %all_cell_type=zeros(size(im_dapi_histone,1),size(im_dapi_histone,2),3);

%     all_oligo_nucleus=zeros(size(im_dapi_histone));
%     all_oligo_soma=zeros(size(im_dapi_histone));
%     all_oligo_processes=zeros(size(im_dapi_histone));
%     all_oligo_cytoplasm=zeros(size(im_dapi_histone));
%     all_oligo_membrane=zeros(size(im_dapi_histone));
%     all_oligo_whole_cell=zeros(size(im_dapi_histone));

    oligodendrocytes_results = struct('index',{},'cell_type',{},'nucleus_x1',{},'nucleus_x2',{},'nucleus_y1',{},'nucleus_y2',{},'nucleus_img',{},...
        'x_c',{},'y_c',{},'soma_x1',{},'soma_x2',{},'soma_y1',{},'soma_y2',{},'soma_img',{},...
        'cell_x1',{},'cell_x2',{},'cell_y1',{},'cell_y2',{},'processes_img',{},'membrane_img',{},'cytoplasm_img',{},'cell_img',{});
cnt=1;
    for i=1: length(xc)
        
        [row,dist]=find_closest_points_index_cnst(seg_masks_c(:,1),seg_masks_c(:,2),xc(i),yc(i)); % find corresponding masks for the boundind box
        iba1_val=bbxs.Iba1(i);

        if iba1_val==1 && dist<50
            disp(strcat('Iteration: ',int2str(IDs(i))));
    
            BB=seg_masks_props(row).BoundingBox;
           
            x1=ceil(BB(1));
            y1=ceil(BB(2));
            x2=x1+BB(3)-1;
            y2=y1+BB(4)-1;
            x_c=seg_masks_c(row,1);
            y_c=seg_masks_c(row,2);

            half_wd=round((x2-x1)/2);
            half_ht=round((y2-y1)/2);

            nucleus_mask1=zeros(w_p+1,w_p+1);
            nucleus_mask1(1+half_w_p-half_ht:1+half_w_p+y2-y1-half_ht,1+half_w_p-half_wd:1+half_w_p+x2-x1-half_wd)=seg_masks_props(row).Image;
           
            oligodendrocytes_results(cnt).index=IDs(i);
            oligodendrocytes_results(cnt).cell_type=4; 

            [soma_mask, processes, cytoplasm, membrane, whole_cell] = oligodendrocyte_segmentation_v4(im_cnpase,nucleus_mask1,x_c,y_c,w_n,w_p);
        
            oligodendrocytes_results(cnt).nucleus_img =uint8(nucleus_mask1);
            oligodendrocytes_results(cnt).nucleus_x1 = x1 - x_c +half_w_p;
            oligodendrocytes_results(cnt).nucleus_x2 = x2 - x_c + half_w_p;
            oligodendrocytes_results(cnt).nucleus_y1 = y1 - y_c +half_w_p;
            oligodendrocytes_results(cnt).nucleus_y2 = y2 - y_c +half_w_p;
            oligodendrocytes_results(cnt).x_c=x_c;
            oligodendrocytes_results(cnt).y_c=y_c;

            soma_props=regionprops(bwconncomp(soma_mask+nucleus_mask1,4)); %added coz soma mask might be split too
            soma_cs=vertcat(soma_props.Centroid);
            soma_cx=soma_cs(:,1);
            soma_cy=soma_cs(:,2);

            soma_idx = find_closest_points_index_cnst(soma_cx,soma_cy,size(soma_mask,1)/2,size(soma_mask,2)/2);
            %added coz soma mask might be split too
            soma_BB=soma_props(soma_idx).BoundingBox;

            oligodendrocytes_results(cnt).soma_x1=ceil(soma_BB(1));
            oligodendrocytes_results(cnt).soma_x2=soma_BB(3)-1+ceil(soma_BB(1));

            oligodendrocytes_results(cnt).soma_y1=ceil(soma_BB(2));
            oligodendrocytes_results(cnt).soma_y2=soma_BB(4)-1+ceil(soma_BB(2));

            oligodendrocytes_results(cnt).soma_img=uint8(soma_mask);
            
            whole_cell_props=regionprops(bwconncomp(whole_cell,4));
            whole_cell_cs=vertcat(whole_cell_props.Centroid);
            whole_cell_cx=whole_cell_cs(:,1);
            whole_cell_cy=whole_cell_cs(:,2);

            whole_cell_idx = find_closest_points_index_cnst(whole_cell_cx,whole_cell_cy,w_p/2,w_p/2);

            whole_cell_BB=whole_cell_props(whole_cell_idx).BoundingBox;

            oligodendrocytes_results(cnt).cell_x1=ceil(whole_cell_BB(1));
            oligodendrocytes_results(cnt).cell_x2=ceil(whole_cell_BB(1))+whole_cell_BB(3)-1;

            oligodendrocytes_results(cnt).cell_y1=ceil(whole_cell_BB(2));
            oligodendrocytes_results(cnt).cell_y2=ceil(whole_cell_BB(2))+whole_cell_BB(4)-1;

            oligodendrocytes_results(cnt).cell_img=uint8(whole_cell);
            oligodendrocytes_results(cnt).processes_img=uint8(processes);
            oligodendrocytes_results(cnt).cytoplasm_img=uint8(cytoplasm);
            oligodendrocytes_results(cnt).membrane_img=uint8(membrane);

%             all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,1)=255;
%             all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,2)=255;
%             all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,3)=255;

%             all_oligo_nucleus(oligodendrocytes_results(cnt).y1:oligodendrocytes_results(i).y2,oligodendrocytes_results(i).x1:oligodendrocytes_results(i).x2)=oligodendrocytes_results(i).nucleus_img+all_oligo_nucleus(oligodendrocytes_results(i).y1:oligodendrocytes_results(i).y2,oligodendrocytes_results(i).x1:oligodendrocytes_results(i).x2);
%             all_oligo_soma(oligodendrocytes_results(i).soma_y1:oligodendrocytes_results(i).soma_y2,oligodendrocytes_results(i).soma_x1:oligodendrocytes_results(i).soma_x2)=oligodendrocytes_results(i).soma_img+all_oligo_soma(oligodendrocytes_results(i).soma_y1:oligodendrocytes_results(i).soma_y2,oligodendrocytes_results(i).soma_x1:oligodendrocytes_results(i).soma_x2);
%             all_oligo_whole_cell(oligodendrocytes_results(i).cell_y1:oligodendrocytes_results(i).cell_y2,oligodendrocytes_results(i).cell_x1:oligodendrocytes_results(i).cell_x2)=oligodendrocytes_results(i).cell_img+all_oligo_whole_cell(oligodendrocytes_results(i).cell_y1:oligodendrocytes_results(i).cell_y2,oligodendrocytes_results(i).cell_x1:oligodendrocytes_results(i).cell_x2);
%             all_oligo_processes(oligodendrocytes_results(i).cell_y1:oligodendrocytes_results(i).cell_y2,oligodendrocytes_results(i).cell_x1:oligodendrocytes_results(i).cell_x2)=oligodendrocytes_results(i).processes_img+all_oligo_processes(oligodendrocytes_results(i).cell_y1:oligodendrocytes_results(i).cell_y2,oligodendrocytes_results(i).cell_x1:oligodendrocytes_results(i).cell_x2);
%             all_oligo_cytoplasm(oligodendrocytes_results(i).cell_y1:oligodendrocytes_results(i).cell_y2,oligodendrocytes_results(i).cell_x1:oligodendrocytes_results(i).cell_x2)=oligodendrocytes_results(i).cytoplasm_img+all_oligo_cytoplasm(oligodendrocytes_results(i).cell_y1:oligodendrocytes_results(i).cell_y2,oligodendrocytes_results(i).cell_x1:oligodendrocytes_results(i).cell_x2);
%             all_oligo_membrane(oligodendrocytes_results(i).cell_y1:oligodendrocytes_results(i).cell_y2,oligodendrocytes_results(i).cell_x1:oligodendrocytes_results(i).cell_x2)=oligodendrocytes_results(i).membrane_img+all_oligo_membrane(oligodendrocytes_results(i).cell_y1:oligodendrocytes_results(i).cell_y2,oligodendrocytes_results(i).cell_x1:oligodendrocytes_results(i).cell_x2);
       cnt=cnt+1;
        end

    end
% 
%     imwrite(all_cell_type,fullfile(p.Results.OUTPUT_DIR,'oligo_cell_type.tif'));
%     imwrite(im2uint8(all_oligo_nucleus),fullfile(p.Results.OUTPUT_DIR,'oligo_nucleus_mask.tif'));
%     imwrite(im2uint8(all_oligo_soma),fullfile(p.Results.OUTPUT_DIR,'oligo_soma_mask.tif'));
%     imwrite(im2uint8(all_oligo_processes),fullfile(p.Results.OUTPUT_DIR,'oligo_processes_mask.tif'));
%     imwrite(im2uint8(all_oligo_membrane),fullfile(p.Results.OUTPUT_DIR,'oligo_membrane_mask.tif'));
%     imwrite(im2uint8(all_oligo_cytoplasm),fullfile(p.Results.OUTPUT_DIR,'oligo_cytoplasm_mask.tif'));
%     imwrite(im2uint8(all_oligo_whole_cell),fullfile(p.Results.OUTPUT_DIR,'oligo_whole_cell_mask.tif'));
%     save(fullfile(p.Results.OUTPUT_DIR,'oligo_reconstruction_info.mat'),'oligodendrocytes_results');

 h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/index', size(oligodendrocytes_results));%.index));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/cell_type', size(oligodendrocytes_results));%.cell_type));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/nucleus_x1', size(oligodendrocytes_results));%.nuclues_x1));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/nucleus_x2', size(oligodendrocytes_results));%.nucleus_x2));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/nucleus_y1', size(oligodendrocytes_results));%.nucleus_y1));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/nucleus_y2', size(oligodendrocytes_results));%.nucleus_y2));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/nucleus_img', [w_p+1,w_p+1,size(oligodendrocytes_results,1),size(oligodendrocytes_results,2)]);%.nucleus_img));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/x_c', size(oligodendrocytes_results));%.x_c));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/y_c', size(oligodendrocytes_results));%.y_c));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/soma_x1', size(oligodendrocytes_results));%.soma_x1));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/soma_x2', size(oligodendrocytes_results));%.soma_x2));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/soma_y1', size(oligodendrocytes_results));%.soma_y1));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/soma_y2', size(oligodendrocytes_results));%.soma_y2));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/soma_img',[w_p+1,w_p+1,size(oligodendrocytes_results,1),size(oligodendrocytes_results,2)]);%.soma_img));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/cell_x1', size(oligodendrocytes_results));%.cell_x1));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/cell_x2', size(oligodendrocytes_results));%.cell_x2));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/cell_y1', size(oligodendrocytes_results));%.cell_y1));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/cell_y2', size(oligodendrocytes_results));%.cell_y2));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/processes_img', [w_p+1,w_p+1,size(oligodendrocytes_results,1),size(oligodendrocytes_results,2)]);%.processes_img));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/membrane_img', [w_p+1,w_p+1,size(oligodendrocytes_results,1),size(oligodendrocytes_results,2)]);%.membrane_img));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/cytoplasm_img', [w_p+1,w_p+1,size(oligodendrocytes_results,1),size(oligodendrocytes_results,2)]);%.cytoplasm_img));
    h5create(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/cell_img', [w_p+1,w_p+1,size(oligodendrocytes_results,1),size(oligodendrocytes_results,2)]);%.cell_img));

    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/index', cell2mat({oligodendrocytes_results.index}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/cell_type', cell2mat({oligodendrocytes_results.cell_type}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/nucleus_x1', cell2mat({oligodendrocytes_results.nucleus_x1}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/nucleus_x2', cell2mat({oligodendrocytes_results.nucleus_x2}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/nucleus_y1', cell2mat({oligodendrocytes_results.nucleus_y1}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/nucleus_y2', cell2mat({oligodendrocytes_results.nucleus_y2}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/nucleus_img', reshape(cell2mat({oligodendrocytes_results.nucleus_img}),[w_p+1,w_p+1,size(oligodendrocytes_results,1),size(oligodendrocytes_results,2)]));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/x_c', cell2mat({oligodendrocytes_results.x_c}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/y_c', cell2mat({oligodendrocytes_results.y_c}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/soma_x1', cell2mat({oligodendrocytes_results.soma_x1}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/soma_x2', cell2mat({oligodendrocytes_results.soma_x2}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/soma_y1', cell2mat({oligodendrocytes_results.soma_y1}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/soma_y2', cell2mat({oligodendrocytes_results.soma_y2}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/soma_img', reshape(cell2mat({oligodendrocytes_results.soma_img}),[w_p+1,w_p+1,size(oligodendrocytes_results,1),size(oligodendrocytes_results,2)]));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/cell_x1', cell2mat({oligodendrocytes_results.cell_x1}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/cell_x2', cell2mat({oligodendrocytes_results.cell_x2}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/cell_y1', cell2mat({oligodendrocytes_results.cell_y1}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/cell_y2', cell2mat({oligodendrocytes_results.cell_y2}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/processes_img', reshape(cell2mat({oligodendrocytes_results.processes_img}),[w_p+1,w_p+1,size(oligodendrocytes_results,1),size(oligodendrocytes_results,2)]));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/membrane_img', reshape(cell2mat({oligodendrocytes_results.membrane_img}),[w_p+1,w_p+1,size(oligodendrocytes_results,1),size(oligodendrocytes_results,2)]));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/cytoplasm_img', reshape(cell2mat({oligodendrocytes_results.cytoplasm_img}),[w_p+1,w_p+1,size(oligodendrocytes_results,1),size(oligodendrocytes_results,2)]));
    h5write(fullfile(p.Results.OUTPUT_DIR,'oligodendrocytes_reconstruction_results.h5'),'/cell_img', reshape(cell2mat({oligodendrocytes_results.cell_img}),[w_p+1,w_p+1,size(oligodendrocytes_results,1),size(oligodendrocytes_results,2)]));

end

