function [soma_mask, processes, cytoplasm, membrane, whole_cell] = neuron_segmentation_v4(im_neun,im_map2,nucleus_mask,x_c,y_c,w_n,w_s,w_p)
    
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
    %get Neun crop
    neun_crop=imcrop(im_neun,[x_c-half_w_s,y_c-half_w_s,w_s,w_s]);
    neun_crop_bw=imbinarize(neun_crop,graythresh(neun_crop));
    %remove connecting neuns
    neun_crop_bw=imdilate(imerode(neun_crop_bw,strel('disk',2)),strel('disk',2));
    
    neun_props=regionprops(neun_crop_bw,'all');
    neun_cs=vertcat(neun_props.Centroid);
    neun_cx=neun_cs(:,1);
    neun_cy=neun_cs(:,2);
    
    [centroid_x, ~] = find_closest_points_cnst(neun_cx,neun_cy,w_n,w_n);
    
    if length(centroid_x)>1 %pick larger area
        neun_idx=find(neun_cx==centroid_x(1));
        area_max=neun_props(neun_idx).Area;
        for j=1:length(centroid_x)
            area_curr=neun_props(j).Area;
            if area_curr>area_max
                area_max=neun_props(j).Area;
                neun_idx=j;
            end
        end
    else
        neun_idx=find(neun_cx==centroid_x);
    end
    neun_BB=neun_props(neun_idx).BoundingBox;
    
    neun_cell_bw=zeros(size(neun_crop_bw));
    %center it
    neun_cell_bw(ceil(neun_BB(2)):ceil(neun_BB(2))+neun_BB(4)-1,ceil(neun_BB(1)):ceil(neun_BB(1))+neun_BB(3)-1)=neun_props(neun_idx).Image;
    %fill any possible holes
    neun_cell_bw=imfill(neun_cell_bw,'holes');
    
    %defining nucleus and soma masks
    im1=imbinarize(neun_cell_bw+imcrop(nucleus_mask,[half_w_p-half_w_s,half_w_p-half_w_s,w_s,w_s]),0.005);
    im1_props=regionprops(im1,'all');
    im1_BB=im1_props(1).BoundingBox;
    
    im2=zeros(size(neun_cell_bw));
    im2(ceil(im1_BB(2)):ceil(im1_BB(2))+im1_BB(4)-1,ceil(im1_BB(1)):ceil(im1_BB(1))+im1_BB(3)-1)=im1_props(1).Image;
        
    soma_plus_nucleus=zeros(w_p+1,w_p+1);
    soma_plus_nucleus(half_w_p-half_w_s:half_w_p+half_w_s,half_w_p-half_w_s:half_w_p+half_w_s)=im2;
    
    soma_mask=imbinarize(soma_plus_nucleus-nucleus_mask,0.005);
    
    %% PROCESSES MASK %%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    map2_crop=imcrop(im_map2,[x_c-half_w_p, y_c-half_w_p, w_p, w_p]);
    map2_crop_bw=imbinarize(map2_crop,1.2*graythresh(map2_crop));
    
    %region growing    
    positive=soma_plus_nucleus;
    negative=~map2_crop_bw;

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
    pts_i=[];%zeros(1,length(x_b));
    
    for j=1:length(x_b)
        val=skel(y_b(j),x_b(j));
        if val>0
            pts_i=[pts_i;y_b(j),x_b(j)];
        end
    end
    
    %These points (x_e,y_e) should not be a part of edge of soma: Refine points
    pts_j=[];%zeros(1,length(x_e));
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
        if any(intersect(idx,idx_i))>0 && any(intersect(idx,idx_j))>0 
            BB=arbor_props(i).BoundingBox;
            im_arbors_ref(ceil(BB(2)):ceil(BB(2))+BB(4)-1,ceil(BB(1)):ceil(BB(1))+BB(3)-1)=arbor_props(i).Image;
        end
    end
    processes=imdilate(im_arbors_ref,strel('diamond',1));
    
    processes_plus_mask=imbinarize(soma_plus_nucleus + processes,0.005);
    
    membrane=imbinarize(imdilate(processes_plus_mask,strel('disk',1))-processes_plus_mask,0.005);
    
    whole_cell=imbinarize(soma_plus_nucleus+processes+membrane,0.005);
    
    cytoplasm=imbinarize(whole_cell-nucleus_mask-membrane,0.005);
    
%     subplot(3,3,1); imshow(imadjust(imcrop(im_dapi_histone,[x_c-half_w_p,y_c-half_w_p,w_p,w_p]))); title('DAPI Histone');
%     subplot(3,3,2); imshow(imadjust(imcrop(im_neun,[x_c-half_w_p,y_c-half_w_p,w_p,w_p]))); title('NeuN');
%     subplot(3,3,3); imshow(imadjust(map2_crop)); title('MAP2');
%     subplot(3,3,4); imshow(nucleus_mask); title('Nucleus Mask');
%     subplot(3,3,5); imshow(soma_mask); title('SOMA Mask');
%     subplot(3,3,6); imshow(processes); title('Processes Mask');
%     subplot(3,3,7); imshow(membrane); title('Membrane Mask');
%     subplot(3,3,8); imshow(cytoplasm); title('Cytoplasm Mask');
%     subplot(3,3,9); imshow(whole_cell); title('Whole Cell Mask');
    
end

