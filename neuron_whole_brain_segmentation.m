function [] = neuron_whole_brain_segmentation(varargin)
    addpath('MORPHOLOGICAL_MASKING');
    defaultDAPIpath='E:\50-plex\final\S1_R1C1.tif';
    defaultHistonepath='E:\50-plex\final\S1_R2C2.tif';
    defaultNeuNpath='E:\50-plex\final\S1_R2C4.tif';
    defaultMAP2path='E:\50-plex\final\S1_R5C9.tif';
    defaultOUTPUTdir='neuron_results';
    defaultCLASSIFICATIONtable='E:\50-plex\classification_results\classification_table.csv';
    defaultSEGMENTATIONmasks='data/merged_labelmask.txt';
    p=inputParser;

    addParameter(p,'DAPI_PATH',defaultDAPIpath,@ischar);
    addParameter(p,'HISTONE_PATH',defaultHistonepath,@ischar);
    addParameter(p,'NeuN_PATH',defaultNeuNpath,@ischar);
    addParameter(p,'MAP2_PATH',defaultMAP2path,@ischar);
    addParameter(p,'OUTPUT_DIR',defaultOUTPUTdir,@ischar);
    addParameter(p,'CLASSIFICATION_table_path',defaultCLASSIFICATIONtable,@ischar);
    addParameter(p,'SEGMENTATION_masks',defaultSEGMENTATIONmasks,@ischar);

    parse(p,varargin{:});
    
    im_dapi_histone=imadd(imread(fullfile(p.Results.DAPI_PATH)),imread(fullfile(p.Results.HISTONE_PATH)));
    im_neun=imread(p.Results.NeuN_PATH);
    im_map2=imread(p.Results.MAP2_PATH);
    
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
    w_s=100;
    w_p=200;

%     all_cell_type=zeros(size(im_dapi_histone,1),size(im_dapi_histone,2),3);

%     all_neuron_nucleus=zeros(size(im_dapi_histone));
%     all_neuron_soma=zeros(size(im_dapi_histone));
%     all_neuron_processes=zeros(size(im_dapi_histone));
%     all_neuron_cytoplasm=zeros(size(im_dapi_histone));
%     all_neuron_membrane=zeros(size(im_dapi_histone));
%     all_neuron_whole_cell=zeros(size(im_dapi_histone));

    neuron_results = struct('index',{},'cell_type',{},'nucleus_x1',{},'nucleus_x2',{},'nucleus_y1',{},'nucleus_y2',{},'nucleus_img',{},...
        'x_c',{},'y_c',{},'soma_x1',{},'soma_x2',{},'soma_y1',{},'soma_y2',{},'soma_img',{},...
        'cell_x1',{},'cell_x2',{},'cell_y1',{},'cell_y2',{},'processes_img',{},'cytoplasm_img',{},'cell_img',{});
cnt=1;
    for i=1:length(xc)

        [row,dist]=find_closest_points_index_cnst(seg_masks_c(:,1),seg_masks_c(:,2),xc(i),yc(i)); % find corresponding masks for the boundind box
        neun_val=bbxs.NeuN(i);

        if neun_val==1 && dist<50
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
            
            
            neuron_results(cnt).index=IDs(i);
            neuron_results(cnt).cell_type=0; %Astrocyte, S100 is astrocyte pan-specific

            [soma_mask, processes, cytoplasm, membrane, whole_cell] = neuron_segmentation_v4(im_neun,im_map2,nucleus_mask1,x_c,y_c,w_n,w_s,w_p);
        
            neuron_results(cnt).nucleus_img = uint8(nucleus_mask1);%seg_masks_props(row).Image;
            neuron_results(cnt).nucleus_x1 = x1 - x_c +half_w_p;
            neuron_results(cnt).nucleus_x2 = x2 - x_c +half_w_p;
            neuron_results(cnt).nucleus_y1 = y1 - y_c +half_w_p;
            neuron_results(cnt).nucleus_y2 = y2 - y_c +half_w_p;
            neuron_results(cnt).x_c=x_c;
            neuron_results(cnt).y_c=y_c;

            soma_props=regionprops(bwconncomp(soma_mask+nucleus_mask1,4)); %added coz soma mask might be split too
            soma_cs=vertcat(soma_props.Centroid);
            soma_cx=soma_cs(:,1);
            soma_cy=soma_cs(:,2);

            soma_idx = find_closest_points_index_cnst(soma_cx,soma_cy,size(soma_mask,1)/2,size(soma_mask,2)/2);
            %added coz soma mask might be split too
            soma_BB=soma_props(soma_idx).BoundingBox;

            neuron_results(cnt).soma_x1=ceil(soma_BB(1));
            neuron_results(cnt).soma_x2=soma_BB(3)-1+ceil(soma_BB(1));

            neuron_results(cnt).soma_y1=ceil(soma_BB(2));
            neuron_results(cnt).soma_y2=soma_BB(4)-1+ceil(soma_BB(2));
            
            neuron_results(cnt).soma_img=uint8(soma_mask);
            whole_cell_props=regionprops(bwconncomp(whole_cell,4));
            whole_cell_cs=vertcat(whole_cell_props.Centroid);
            whole_cell_cx=whole_cell_cs(:,1);
            whole_cell_cy=whole_cell_cs(:,2);

            whole_cell_idx = find_closest_points_index_cnst(whole_cell_cx,whole_cell_cy,w_p/2,w_p/2);

            whole_cell_BB=whole_cell_props(whole_cell_idx).BoundingBox;

            neuron_results(cnt).cell_x1=ceil(whole_cell_BB(1));
            neuron_results(cnt).cell_x2=ceil(whole_cell_BB(1))+whole_cell_BB(3)-1;

            neuron_results(cnt).cell_y1=ceil(whole_cell_BB(2));
            neuron_results(cnt).cell_y2=ceil(whole_cell_BB(2))+whole_cell_BB(4)-1;
           
            neuron_results(cnt).cell_img=uint8(whole_cell);
            neuron_results(cnt).processes_img=uint8(processes);
            neuron_results(cnt).cytoplasm_img=uint8(cytoplasm);
            neuron_results(cnt).membrane_img=uint8(membrane);

%             all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,1)=0;
%             all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,2)=0;
%             all_cell_type(y_c-2:y_c+2,x_c-2:x_c+2,3)=255;

%             all_neuron_nucleus(neuron_results(cnt).nucleus_y1:neuron_results(cnt).nucleus_y2,neuron_results(cnt).nucleus_x1:neuron_results(cnt).nucleus_x2)=neuron_results(cnt).nucleus_img+all_neuron_nucleus(neuron_results(cnt).nucleus_y1:neuron_results(cnt).nucleus_y2,neuron_results(cnt).nucleus_x1:neuron_results(cnt).nucleus_x2);
%             all_neuron_soma(neuron_results(cnt).soma_y1:neuron_results(cnt).soma_y2,neuron_results(cnt).soma_x1:neuron_results(cnt).soma_x2)=neuron_results(cnt).soma_img+all_neuron_soma(neuron_results(cnt).soma_y1:neuron_results(cnt).soma_y2,neuron_results(cnt).soma_x1:neuron_results(cnt).soma_x2);
%             all_neuron_whole_cell(neuron_results(cnt).cell_y1:neuron_results(cnt).cell_y2,neuron_results(cnt).cell_x1:neuron_results(cnt).cell_x2)=neuron_results(cnt).cell_img+all_neuron_whole_cell(neuron_results(cnt).cell_y1:neuron_results(cnt).cell_y2,neuron_results(cnt).cell_x1:neuron_results(cnt).cell_x2);
%             all_neuron_processes(neuron_results(cnt).cell_y1:neuron_results(cnt).cell_y2,neuron_results(cnt).cell_x1:neuron_results(cnt).cell_x2)=neuron_results(cnt).processes_img+all_neuron_processes(neuron_results(cnt).cell_y1:neuron_results(cnt).cell_y2,neuron_results(cnt).cell_x1:neuron_results(cnt).cell_x2);
%             all_neuron_cytoplasm(neuron_results(cnt).cell_y1:neuron_results(cnt).cell_y2,neuron_results(cnt).cell_x1:neuron_results(cnt).cell_x2)=neuron_results(cnt).cytoplasm_img+all_neuron_cytoplasm(neuron_results(cnt).cell_y1:neuron_results(cnt).cell_y2,neuron_results(cnt).cell_x1:neuron_results(cnt).cell_x2);
%             all_neuron_membrane(neuron_results(cnt).cell_y1:neuron_results(cnt).cell_y2,neuron_results(cnt).cell_x1:neuron_results(cnt).cell_x2)=neuron_results(cnt).membrane_img+all_neuron_membrane(neuron_results(cnt).cell_y1:neuron_results(cnt).cell_y2,neuron_results(cnt).cell_x1:neuron_results(cnt).cell_x2);
       cnt=cnt+1;
        end

    end

%     imwrite(all_cell_type,fullfile(p.Results.OUTPUT_DIR,'neuron_cell_type.tif'));
%     imwrite(im2uint8(all_neuron_nucleus),fullfile(p.Results.OUTPUT_DIR,'neuron_nucleus_mask.tif'));
%     imwrite(im2uint8(all_neuron_soma),fullfile(p.Results.OUTPUT_DIR,'neuron_soma_mask.tif'));
%     imwrite(im2uint8(all_neuron_processes),fullfile(p.Results.OUTPUT_DIR,'neuron_processes_mask.tif'));
%     imwrite(im2uint8(all_neuron_membrane),fullfile(p.Results.OUTPUT_DIR,'neuron_membrane_mask.tif'));
%     imwrite(im2uint8(all_neuron_cytoplasm),fullfile(p.Results.OUTPUT_DIR,'neuron_cytoplasm_mask.tif'));
%     imwrite(im2uint8(all_neuron_whole_cell),fullfile(p.Results.OUTPUT_DIR,'neuron_whole_cell_mask.tif'));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/index', size(neuron_results));%.index));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/cell_type', size(neuron_results));%.cell_type));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/nucleus_x1', size(neuron_results));%.nuclues_x1));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/nucleus_x2', size(neuron_results));%.nucleus_x2));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/nucleus_y1', size(neuron_results));%.nucleus_y1));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/nucleus_y2', size(neuron_results));%.nucleus_y2));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/nucleus_img', [w_p+1,w_p+1,size(neuron_results,1),size(neuron_results,2)]);%.nucleus_img));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/x_c', size(neuron_results));%.x_c));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/y_c', size(neuron_results));%.y_c));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/soma_x1', size(neuron_results));%.soma_x1));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/soma_x2', size(neuron_results));%.soma_x2));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/soma_y1', size(neuron_results));%.soma_y1));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/soma_y2', size(neuron_results));%.soma_y2));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/soma_img',[w_p+1,w_p+1,size(neuron_results,1),size(neuron_results,2)]);%.soma_img));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/cell_x1', size(neuron_results));%.cell_x1));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/cell_x2', size(neuron_results));%.cell_x2));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/cell_y1', size(neuron_results));%.cell_y1));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/cell_y2', size(neuron_results));%.cell_y2));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/processes_img', [w_p+1,w_p+1,size(neuron_results,1),size(neuron_results,2)]);%.processes_img));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/membrane_img', [w_p+1,w_p+1,size(neuron_results,1),size(neuron_results,2)]);%.membrane_img));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/cytoplasm_img', [w_p+1,w_p+1,size(neuron_results,1),size(neuron_results,2)]);%.cytoplasm_img));
    h5create(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/cell_img', [w_p+1,w_p+1,size(neuron_results,1),size(neuron_results,2)]);%.cell_img));

    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/index', cell2mat({neuron_results.index}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/cell_type', cell2mat({neuron_results.cell_type}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/nucleus_x1', cell2mat({neuron_results.nucleus_x1}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/nucleus_x2', cell2mat({neuron_results.nucleus_x2}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/nucleus_y1', cell2mat({neuron_results.nucleus_y1}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/nucleus_y2', cell2mat({neuron_results.nucleus_y2}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/nucleus_img', reshape(cell2mat({neuron_results.nucleus_img}),[w_p+1,w_p+1,size(neuron_results,1),size(neuron_results,2)]));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/x_c', cell2mat({neuron_results.x_c}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/y_c', cell2mat({neuron_results.y_c}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/soma_x1', cell2mat({neuron_results.soma_x1}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/soma_x2', cell2mat({neuron_results.soma_x2}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/soma_y1', cell2mat({neuron_results.soma_y1}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/soma_y2', cell2mat({neuron_results.soma_y2}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/soma_img', reshape(cell2mat({neuron_results.soma_img}),[w_p+1,w_p+1,size(neuron_results,1),size(neuron_results,2)]));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/cell_x1', cell2mat({neuron_results.cell_x1}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/cell_x2', cell2mat({neuron_results.cell_x2}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/cell_y1', cell2mat({neuron_results.cell_y1}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/cell_y2', cell2mat({neuron_results.cell_y2}));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/processes_img', reshape(cell2mat({neuron_results.processes_img}),[w_p+1,w_p+1,size(neuron_results,1),size(neuron_results,2)]));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/membrane_img', reshape(cell2mat({neuron_results.membrane_img}),[w_p+1,w_p+1,size(neuron_results,1),size(neuron_results,2)]));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/cytoplasm_img', reshape(cell2mat({neuron_results.cytoplasm_img}),[w_p+1,w_p+1,size(neuron_results,1),size(neuron_results,2)]));
    h5write(fullfile(p.Results.OUTPUT_DIR,'neuron_reconstruction_results.h5'),'/cell_img', reshape(cell2mat({neuron_results.cell_img}),[w_p+1,w_p+1,size(neuron_results,1),size(neuron_results,2)]));

end

