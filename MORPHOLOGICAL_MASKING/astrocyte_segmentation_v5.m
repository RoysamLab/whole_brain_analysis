function [soma_mask, processes, cytoplasm, membrane, whole_cell] = astrocyte_segmentation_v5(im_s100,im_gfap,nucleus_mask,x_c,y_c,~,w_s,w_p)
    %half_w_n=round(w_n/2);
    half_w_s=round(w_s/2);
    half_w_p=round(w_p/2);
    
    %% NUCLEUS MASK %%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %get dapi + histone crop
%     dapi_histone_crop=imcrop(im_dapi_histone,[x_c-half_w_n,y_c-half_w_n,w_n,w_n]);
%     
%     nucleus_mask=zeros(w_p+1,w_p+1);
%     nucleus_mask(half_w_p-half_w_n:half_w_p+half_w_n,half_w_p-half_w_n:half_w_p+half_w_n)=get_center_cell(dapi_histone_crop);
%     
    %% SOMA MASK %%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%
    s100_crop=imcrop(im_s100,[x_c-half_w_s,y_c-half_w_s,w_s,w_s]);
    s100_crop_bw=imbinarize(s100_crop,graythresh(s100_crop));
    
    s100_props=regionprops(s100_crop_bw,'all');
    s100_cs=vertcat(s100_props.Centroid);
    s100_cx=s100_cs(:,1);
    s100_cy=s100_cs(:,2);
    
    [centroid_x, ~] = find_closest_points_cnst(s100_cx,s100_cy,half_w_s,half_w_s);
    if length(centroid_x)>1 %pick larger area
        s100_idx=find(s100_cx==centroid_x(1));
        area_max=s100_props(s100_idx).Area;
        for j=1:length(centroid_x)
            area_curr=s100_props(j).Area;
            if area_curr>area_max
                area_max=s100_props(j).Area;
                s100_idx=j;
            end
        end
    else
        s100_idx=find(s100_cx==centroid_x);
    end
        
    s100_BB=s100_props(s100_idx).BoundingBox;

    s100_cell_bw=zeros(size(s100_crop_bw));
    %center it
    s100_cell_bw(ceil(s100_BB(2)):ceil(s100_BB(2))+s100_BB(4)-1,ceil(s100_BB(1)):ceil(s100_BB(1))+s100_BB(3)-1)=s100_props(s100_idx).Image;
    %fill any possible holes
    s100_cell_bw=imfill(s100_cell_bw,'holes');
    
    s100_cell_bw1=zeros(w_p+1,w_p+1);
    s100_cell_bw1(half_w_p-half_w_s:half_w_p+half_w_s,half_w_p-half_w_s:half_w_p+half_w_s)=s100_cell_bw;
    
    %Using directional ratios
    soma_plus_mask=imbinarize(s100_cell_bw1+nucleus_mask,0.01);
    
    soma_plus_mask_no_process=imfill(get_soma_v2(soma_plus_mask,w_p,w_p),'holes');
    
    windowSize = 11;
    kernel = ones(windowSize) / windowSize ^ 2;
    blurryImage = conv2(im2single(soma_plus_mask_no_process), kernel, 'same');
    soma_no_process = blurryImage > 0.5; % Rethreshold

    soma_plus_nucleus=imbinarize(soma_no_process+nucleus_mask,0.01);
    soma_mask=imbinarize(soma_plus_nucleus-nucleus_mask,0.005);
    
    %% PROCESSES MASK %%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %find if GFAP present in processes
    
    s100_crop=imcrop(im_s100,[x_c-half_w_p,y_c-half_w_p,w_p,w_p]);
    s100_crop_bw=imbinarize(imadjust(s100_crop),graythresh(imadjust(s100_crop)));
    
    %get GFAP crop
    gfap_crop=imcrop(im_gfap,[x_c-half_w_p,y_c-half_w_p,w_p,w_p]);
    gfap_crop_bw=imbinarize(imadjust(gfap_crop),graythresh(imadjust(gfap_crop)));
    
    gfap_val=sum(sum(gfap_crop_bw.*imbinarize(imdilate(soma_plus_nucleus,strel('disk',2))-soma_plus_nucleus,0.005)));
   
    s100_val=sum(sum(s100_crop_bw.*imbinarize(imdilate(soma_plus_nucleus,strel('disk',2))-soma_plus_nucleus,0.005)));

    %region growing
    if gfap_val>s100_val %0.95*s100_val && gfap_val<1.05*s100_val
        negative=~imbinarize(gfap_crop,1.3*graythresh(gfap_crop));
    else
        negative=~imbinarize(s100_crop,1.2*graythresh(s100_crop));
    end
    
    positive=soma_plus_nucleus;
    labels=positive-1*negative; 
    labels_out=grow_from_seeds(labels);

    %skeletonization
    skel= bwmorph(logical(labels_out),'skel',Inf);
    B = bwmorph(skel, 'branchpoints');
    E = bwmorph(skel, 'endpoints');
    
    %get skel and edge
    edge_soma=edge(soma_plus_nucleus);
    skel_plus_edge=edge_soma+skel;
    B1 = bwmorph(skel_plus_edge, 'branchpoints');
   
    %Refine the end and branch points
    [y_e,x_e] = find(E);
    [y_b,x_b]=find(B1-B);
    
    %These points (x_b,y_b) should be a part of skel: Refine points
    pts_i=[];
    
    for j=1:length(x_b)
        val=skel(y_b(j),x_b(j));
        if val>0
            pts_i=[pts_i;[y_b(j),x_b(j)]];
        end
    end
    
    %These points (x_e,y_e) should not be a part of edge of soma: Refine points
    % These points should not be till the edge of image crop
    pts_j=[];
    for j=1:length(x_e)
        val=edge_soma(y_e(j),x_e(j));
        if val==0
            pts_j=[pts_j;[y_e(j),x_e(j)]];
        end
    end
    
    im_arbors=imbinarize(skel_plus_edge-soma_plus_nucleus,0.005);
    %Get props of this, connectivity 8
    arbor_props=regionprops(im_arbors,'all');
    
    %arbor should have minimum one red and blue point %%THIS SOMETIMES
    %REMOVES IMPORTANT PROCESSES
    im_arbors_ref=zeros(size(im_arbors));
    
    if ~isempty(pts_i)
        idx_i=pts_i(:,2)*size(im_arbors,1)+pts_i(:,1);
    else
        idx_i=-1;
    end
    
    if ~isempty(pts_j)>0
        idx_j=pts_j(:,2)*size(im_arbors,1)+pts_j(:,1);
    else
        idx_j=-1;
    end

    for i=1:length(arbor_props)
        pts=arbor_props(i).PixelList;%(x,y)
        idx=pts(:,1)*size(im_arbors,1)+pts(:,2);
        if any(intersect(idx,idx_i))>0 && any(intersect(idx,idx_j))>0 
            BB=arbor_props(i).BoundingBox;
            im_arbors_ref(ceil(BB(2)):ceil(BB(2))+BB(4)-1,ceil(BB(1)):ceil(BB(1))+BB(3)-1)=arbor_props(i).Image;
        end
    end
    
   processes=imdilate(im_arbors,strel('diamond',1));
    
    processes_plus_mask=imbinarize(soma_plus_nucleus + processes,0.005);
    
    membrane=imbinarize(imdilate(processes_plus_mask,strel('disk',1))-processes_plus_mask,0.005);
    
    whole_cell=imbinarize(soma_plus_nucleus+processes+membrane,0.005);
    
    cytoplasm=imbinarize(whole_cell-nucleus_mask-membrane,0.005);
%     
%     %subplot(3,3,1); imshow(imadjust(imcrop(im_dapi_histone,[x_c-half_w_p,y_c-half_w_p,w_p,w_p]))); title('DAPI Histone');
%     subplot(3,3,2); imshow(imadjust(s100_crop)); title('S100');
%     subplot(3,3,3); imshow(imadjust(gfap_crop)); title('GFAP');
%     subplot(3,3,4); imshow(nucleus_mask); title('Nucleus Mask');
%     subplot(3,3,5); imshow(im2bw(soma_plus_mask_no_process - nucleus_mask,0.005)); title('SOMA Mask');
%     subplot(3,3,6); imshow(processes); title('Processes Mask');
%     subplot(3,3,7); imshow(membrane); title('Membrane Mask');
%     subplot(3,3,8); imshow(cytoplasm); title('Cytoplasm Mask');
%     subplot(3,3,9); imshow(whole_cell); title('Whole Cell Mask');
    
%     subplot(2,2,1); imshow(imadjust(imcrop(im_dapi_histone,[x_c-half_w_p,y_c-half_w_p,w_p,w_p])));
%     subplot(2,2,2); imshow(imadjust(s100_crop));
%     subplot(2,2,3); imshow(nucleus_mask); 
%     subplot(2,2,4); imshow(soma_plus_mask); 
%     
%     subplot(3,3,1); imshow(imadjust(imcrop(im_dapi_histone,[x_c-half_w_p,y_c-half_w_p,w_p,w_p]))); title('DAPI Histone');
%     subplot(3,3,2); imshow(imadjust(s100_crop)); title('S100');
%     %subplot(3,3,3); imshow(imcrop(imread('results_v1.0\Astrocytes_examples\33125_20900_550_550_flatten.png'),[x_c-x1,y_c-y1,w_p,w_p])); title('GFAP');
%     subplot(3,3,4); imshow(nucleus_mask); title('Nucleus Mask');
%     subplot(3,3,5); imshow(im2bw(soma_plus_mask_no_process - nucleus_mask,0.005)); title('SOMA Mask');
%     subplot(3,3,6); imshow(processes); title('Processes Mask');
%     subplot(3,3,7); imshow(membrane); title('Membrane Mask');
%     subplot(3,3,8); imshow(cytoplasm); title('Cytoplasm Mask');
%     subplot(3,3,9); imshow(whole_cell); title('Whole Cell Mask');
    
    
%     imwrite(imadjust(imcrop(im_dapi_histone,[x_c-half_w_p,y_c-half_w_p,w_p,w_p])),strcat('results_v1.0\Astrocytes_examples\',int2str(x_c),'_',int2str(y_c),'_dapi_histone_crop.tif'));
%     imwrite(imadjust(s100_crop),strcat('results_v1.0\Astrocytes_examples\',int2str(x_c),'_',int2str(y_c),'_s100_crop.tif'));
%     imwrite(soma_plus_mask,strcat('results_v1.0\Astrocytes_examples\',int2str(x_c),'_',int2str(y_c),'_soma_plus_mask_crop.tif'));
%     
%     imwrite(imadjust(gfap_crop),strcat('results_v1.0\Astrocytes_examples\',int2str(x_c),'_',int2str(y_c),'_gfap_crop.tif'));
%     imwrite(nucleus_mask,strcat('results_v1.0\Astrocytes_examples\',int2str(x_c),'_',int2str(y_c),'_nucleus.tif'));
%     imwrite(im2bw(soma_plus_mask_no_process - nucleus_mask,0.005),strcat('results_v1.0\Astrocytes_examples\',int2str(x_c),'_',int2str(y_c),'_soma.tif'));
%     imwrite(processes,strcat('results_v1.0\Astrocytes_examples\',int2str(x_c),'_',int2str(y_c),'_processes.tif'));
%     imwrite(membrane,strcat('results_v1.0\Astrocytes_examples\',int2str(x_c),'_',int2str(y_c),'_membrane.tif'));
%     imwrite(cytoplasm,strcat('results_v1.0\Astrocytes_examples\',int2str(x_c),'_',int2str(y_c),'_cytoplasm.tif'));
%     imwrite(whole_cell,strcat('results_v1.0\Astrocytes_examples\',int2str(x_c),'_',int2str(y_c),'_whole_cell.tif'));

end