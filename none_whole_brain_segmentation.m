function [] = none_whole_brain_segmentation(varargin)
    addpath('MORPHOLOGICAL_MASKING');
    defaultDAPIpath='E:\50-plex\final\S1_R1C1.tif';
    defaultHistonepath='E:\50-plex\final\S1_R2C2.tif';
    defaultOUTPUTdir='none_results';
    defaultCLASSIFICATIONtable='E:\50-plex\classification_results\classification_table.csv';
    defaultSEGMENTATIONmasks='data/merged_labelmask.txt';
    p=inputParser;

    addParameter(p,'DAPI_PATH',defaultDAPIpath,@ischar);
    addParameter(p,'HISTONE_PATH',defaultHistonepath,@ischar);
    addParameter(p,'OUTPUT_DIR',defaultOUTPUTdir,@ischar);
    addParameter(p,'CLASSIFICATION_table_path',defaultCLASSIFICATIONtable,@ischar);
    addParameter(p,'SEGMENTATION_masks',defaultSEGMENTATIONmasks,@ischar);

    parse(p,varargin{:});
    
    im_dapi_histone=imadd(imread(fullfile(p.Results.DAPI_PATH)),imread(fullfile(p.Results.HISTONE_PATH)));
    
    %create output dir
    if ~exist(p.Results.OUTPUT_DIR,'dir')
        mkdir(p.Results.OUTPUT_DIR)
    end

    %load boundingboxes
    x0=1;y0=1;
    W=size(im_dapi_histone,2);
    H=size(im_dapi_histone,1);

    bbxs=get_bbxs_csv(p.Results.CLASSIFICATION_table_path,W,H,x0,y0);
    xc=bbxs.centroid_x;
    yc=bbxs.centroid_y;

    %load segmentation masks
    seg_masks=readtable(p.Results.SEGMENTATION_masks);
    seg_masks_array=table2array(seg_masks);

    seg_masks_props=regionprops(seg_masks_array,'Centroid','Image','BoundingBox');
    seg_masks_c=vertcat(seg_masks_props.Centroid);

    half_w_n=50;
    w_n=50;
    w_s=100;
    w_p=200;
    half_w_p=100;

    all_cell_type=zeros(size(im_dapi_histone,1),size(im_dapi_histone,2),3);

    all_none_nucleus=zeros(size(im_dapi_histone));
    all_none_soma=zeros(size(im_dapi_histone));
    all_none_processes=zeros(size(im_dapi_histone));
    all_none_cytoplasm=zeros(size(im_dapi_histone));
    all_none_membrane=zeros(size(im_dapi_histone));
    all_none_whole_cell=zeros(size(im_dapi_histone));

    none_results = struct('index',{},'cell_type',{},'nucleus_x1',{},'nucleus_x2',{},'nucleus_y1',{},'nucleus_y2',{},'nucleus_img',{},...
        'x_c',{},'y_c',{},'soma_x1',{},'soma_x2',{},'soma_y1',{},'soma_y2',{},'soma_img',{},...
        'cell_x1',{},'cell_x2',{},'cell_y1',{},'cell_y2',{},'processes_img',{},'membrane_img',{},'cytoplasm_img',{},'cell_img',{});

    for i=1: length(xc)
        
        disp(strcat('Iteration: ',int2str(i)));
        row=find_closest_points_index_cnst(seg_masks_c(:,1),seg_masks_c(:,2),xc(i),yc(i)); % find corresponding masks for the boundind box
        neun_val=bbxs.NeuN(i);
        s100_val=bbxs.S100(i);
        olig2_val=bbxs.Olig2(i);
        iba1_val=bbxs.Iba1(i);
        reca1_val=bbxs.RECA1(i);
        
        if s100_val==0 && neun_val==0 && iba1_val==0 && reca1_val==0 && olig2_val==0
           
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
            nucleus_mask1(half_w_p-half_ht:half_w_p+y2-y1-half_ht,half_w_p-half_wd:half_w_p+x2-x1-half_wd)=seg_masks_props(row).Image;

            none_results(i).index=i;
            none_results(i).cell_type=5; %Astrocyte, S100 is astrocyte pan-specific

            [soma_mask, processes, cytoplasm, membrane, whole_cell] = none_segmentation_v3(nucleus_mask1,w_n,w_p);
        
            none_results(i).nucleus_img = seg_masks_props(row).Image;
            none_results(i).x1 = x1;
            none_results(i).x2 = x2;
            none_results(i).y1 = y1;
            none_results(i).y2 = y2;
            none_results(i).x_c=x_c;
            none_results(i).y_c=y_c;

            soma_props=regionprops(bwconncomp(soma_mask+nucleus_mask1,4)); %added coz soma mask might be split too
            soma_cs=vertcat(soma_props.Centroid);
            soma_cx=soma_cs(:,1);
            soma_cy=soma_cs(:,2);

            soma_idx = find_closest_points_index_cnst(soma_cx,soma_cy,size(soma_mask,1)/2,size(soma_mask,2)/2);
            %added coz soma mask might be split too
            soma_BB=soma_props(soma_idx).BoundingBox;

            none_results(i).soma_x1=x_c-half_w_p+ceil(soma_BB(1));
            none_results(i).soma_x2=x_c-half_w_p+soma_BB(3)-1+ceil(soma_BB(1));

            none_results(i).soma_y1=y_c-half_w_p+ceil(soma_BB(2));
            none_results(i).soma_y2=y_c-half_w_p+soma_BB(4)-1+ceil(soma_BB(2));

            none_results(i).soma_img=imcrop(soma_mask,[ceil(soma_BB(1)),ceil(soma_BB(2)),soma_BB(3)-1,soma_BB(4)-1]);

            whole_cell_props=regionprops(bwconncomp(whole_cell,4));
            whole_cell_cs=vertcat(whole_cell_props.Centroid);
            whole_cell_cx=whole_cell_cs(:,1);
            whole_cell_cy=whole_cell_cs(:,2);

            whole_cell_idx = find_closest_points_index_cnst(whole_cell_cx,whole_cell_cy,w_p/2,w_p/2);

            whole_cell_BB=whole_cell_props(whole_cell_idx).BoundingBox;

            none_results(i).cell_x1=x_c-half_w_p+ceil(whole_cell_BB(1));
            none_results(i).cell_x2=x_c-half_w_p+ceil(whole_cell_BB(1))+whole_cell_BB(3)-1;

            none_results(i).cell_y1=y_c-half_w_p+ceil(whole_cell_BB(2));
            none_results(i).cell_y2=y_c-half_w_p+ceil(whole_cell_BB(2))+whole_cell_BB(4)-1;

            none_results(i).cell_img=imcrop(whole_cell,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
            none_results(i).processes_img=imcrop(processes,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
            none_results(i).cytoplasm_img=imcrop(cytoplasm,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);
            none_results(i).membrane_img=imcrop(membrane,[ceil(whole_cell_BB(1)),ceil(whole_cell_BB(2)),whole_cell_BB(3)-1,whole_cell_BB(4)-1]);

            all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,1)=0;
            all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,2)=255;
            all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,3)=255;

            all_none_nucleus(none_results(i).y1:none_results(i).y2,none_results(i).x1:none_results(i).x2)=none_results(i).nucleus_img+all_none_nucleus(none_results(i).y1:none_results(i).y2,none_results(i).x1:none_results(i).x2);
            all_none_soma(none_results(i).soma_y1:none_results(i).soma_y2,none_results(i).soma_x1:none_results(i).soma_x2)=none_results(i).soma_img+all_none_soma(none_results(i).soma_y1:none_results(i).soma_y2,none_results(i).soma_x1:none_results(i).soma_x2);
            all_none_whole_cell(none_results(i).cell_y1:none_results(i).cell_y2,none_results(i).cell_x1:none_results(i).cell_x2)=none_results(i).cell_img+all_none_whole_cell(none_results(i).cell_y1:none_results(i).cell_y2,none_results(i).cell_x1:none_results(i).cell_x2);
            all_none_processes(none_results(i).cell_y1:none_results(i).cell_y2,none_results(i).cell_x1:none_results(i).cell_x2)=none_results(i).processes_img+all_none_processes(none_results(i).cell_y1:none_results(i).cell_y2,none_results(i).cell_x1:none_results(i).cell_x2);
            all_none_cytoplasm(none_results(i).cell_y1:none_results(i).cell_y2,none_results(i).cell_x1:none_results(i).cell_x2)=none_results(i).cytoplasm_img+all_none_cytoplasm(none_results(i).cell_y1:none_results(i).cell_y2,none_results(i).cell_x1:none_results(i).cell_x2);
            all_none_membrane(none_results(i).cell_y1:none_results(i).cell_y2,none_results(i).cell_x1:none_results(i).cell_x2)=none_results(i).membrane_img+all_none_membrane(none_results(i).cell_y1:none_results(i).cell_y2,none_results(i).cell_x1:none_results(i).cell_x2);
        end

    end

    imwrite(all_cell_type,fullfile(p.Results.OUTPUT_DIR,'none_cell_type.tif'));
    imwrite(im2uint8(all_none_nucleus),fullfile(p.Results.OUTPUT_DIR,'none_nucleus_mask.tif'));
    imwrite(im2uint8(all_none_soma),fullfile(p.Results.OUTPUT_DIR,'none_soma_mask.tif'));
    imwrite(im2uint8(all_none_processes),fullfile(p.Results.OUTPUT_DIR,'none_processes_mask.tif'));
    imwrite(im2uint8(all_none_membrane),fullfile(p.Results.OUTPUT_DIR,'none_membrane_mask.tif'));
    imwrite(im2uint8(all_none_cytoplasm),fullfile(p.Results.OUTPUT_DIR,'none_cytoplasm_mask.tif'));
    imwrite(im2uint8(all_none_whole_cell),fullfile(p.Results.OUTPUT_DIR,'none_whole_cell_mask.tif'));
    save(fullfile(p.Results.OUTPUT_DIR,'none_reconstruction_info.mat'),'none_results');
end

