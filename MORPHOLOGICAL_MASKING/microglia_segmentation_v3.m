function [soma_mask, processes, cytoplasm, membrane, whole_cell] = microglia_segmentation_v3(im_iba1,nucleus_mask,x_c,y_c,~,w_s,w_p)
    
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
    
    %% SOMA MASK %%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%
    iba1_crop=imcrop(im_iba1,[x_c-half_w_s,y_c-half_w_s,w_s,w_s]);
    iba1_crop_bw=imbinarize(iba1_crop,graythresh(iba1_crop));
    
    iba1_props=regionprops(iba1_crop_bw,'all');
    iba1_cs=vertcat(iba1_props.Centroid);
    iba1_cx=iba1_cs(:,1);
    iba1_cy=iba1_cs(:,2);
    
    [centroid_x, ~] = find_closest_points_cnst(iba1_cx,iba1_cy,half_w_s,half_w_s);
     if length(centroid_x)>1 %pick larger area
        iba1_idx=find(iba1_cx==centroid_x(1));
        area_max=iba1_props(iba1_idx).Area;
        for j=1:length(centroid_x)
            area_curr=iba1_props(j).Area;
            if area_curr>area_max
                area_max=iba1_props(j).Area;
                iba1_idx=j;
            end
        end
    else
        iba1_idx=find(iba1_cx==centroid_x);
    end
            
    iba1_BB=iba1_props(iba1_idx).BoundingBox;

    iba1_cell_bw=zeros(size(iba1_crop_bw));
    %center it
    iba1_cell_bw(ceil(iba1_BB(2)):ceil(iba1_BB(2))+iba1_BB(4)-1,ceil(iba1_BB(1)):ceil(iba1_BB(1))+iba1_BB(3)-1)=iba1_props(iba1_idx).Image;
    %fill any possible holes
    iba1_cell_bw=imfill(iba1_cell_bw,'holes');
    
    iba1_cell_bw1=zeros(w_p+1,w_p+1);
    iba1_cell_bw1(half_w_p-half_w_s:half_w_p+half_w_s,half_w_p-half_w_s:half_w_p+half_w_s)=iba1_cell_bw;
    
    %remove potential arbor detected
    windowSize = 11;
    kernel = ones(windowSize) / windowSize ^ 2;
    blurryImage = conv2(im2single(iba1_cell_bw1), kernel, 'same');
    binaryImage = blurryImage > 0.5; % Rethreshold
    
    soma_plus_nucleus=imbinarize(binaryImage+nucleus_mask,0.01);
    soma_mask=imbinarize(soma_plus_nucleus-nucleus_mask,0.005);
    
    %% PROCESSES MASK %%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    iba1_crop=imcrop(im_iba1,[x_c-half_w_p, y_c-half_w_p, w_p, w_p]);
    iba1_minus_soma=iba1_crop.*uint16(~soma_plus_nucleus);
    
    iba1_minus_soma_bw=imbinarize(iba1_minus_soma,1.2*graythresh(iba1_minus_soma));
    
    %region growing
    positive=soma_plus_nucleus;
    negative=~imbinarize(iba1_minus_soma_bw+soma_plus_nucleus,0.005);

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
    
    %arbor should have minimum one red and blue point
    im_arbors_ref=zeros(size(im_arbors));
    
    if ~isempty(pts_i)
        idx_i=pts_i(:,2)*size(im_arbors,1)+pts_i(:,1);
    else
        idx_i=-1;
    end
    
    if ~isempty(pts_j)
        idx_j=pts_j(:,2)*size(im_arbors,1)+pts_j(:,1);
    else
        idx_j=-1;
    end
    
    for i=1:length(arbor_props)
        pts=arbor_props(i).PixelList;%(x,y)
        idx=pts(:,1)*size(im_arbors,1)+pts(:,2);
        if any(intersect(idx,idx_i))>0 && any(intersect(idx,idx_j))>0 %how to find intersection here????
            BB=arbor_props(i).BoundingBox;
            im_arbors_ref(ceil(BB(2)):ceil(BB(2))+BB(4)-1,ceil(BB(1)):ceil(BB(1))+BB(3)-1)=arbor_props(i).Image;
        end
    end
    processes=imdilate(im_arbors_ref,strel('diamond',1));
    
    processes_plus_mask=imbinarize(soma_plus_nucleus + processes,0.005);
    
    membrane=imbinarize(imdilate(processes_plus_mask,strel('disk',1))-processes_plus_mask,0.005);
    
    whole_cell=imbinarize(soma_plus_nucleus+processes+membrane,0.005);
    
    cytoplasm=imbinarize(whole_cell-nucleus_mask-membrane,0.005);
    
    %subplot(2,4,1); imshow(imadjust(imcrop(im_dapi_histone,[x_c-half_w_p,y_c-half_w_p,w_p,w_p]))); title('DAPI Histone');
    subplot(2,4,1); imshow(imadjust(iba1_crop)); title('IBA1');
    subplot(2,4,2); imshow(nucleus_mask); title('Nucleus Mask');
    subplot(2,4,3); imshow(soma_mask+nucleus_mask); title('SOMA Mask');
    subplot(2,4,4); imshow(processes); title('Processes Mask');
    subplot(2,4,5); imshow(membrane); title('Membrane Mask');
    subplot(2,4,6); imshow(cytoplasm); title('Cytoplasm Mask');
    subplot(2,4,7); imshow(whole_cell); title('Whole Cell Mask');
    
    
end
    