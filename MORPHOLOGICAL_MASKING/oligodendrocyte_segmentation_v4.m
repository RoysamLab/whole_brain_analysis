function [soma_mask, processes, cytoplasm, membrane, whole_cell] = oligodendrocyte_segmentation_v4(im_cnpase,nucleus_mask,x_c,y_c,~,w_p)
    
%     half_w_n=round(w_n/2);
    half_w_p=round(w_p/2);
    
    %% NUCLEUS MASK %%%%%%%
%     %%%%%%%%%%%%%%%%%%%%%%%%%%
%     %get dapi + histone crop
%     dapi_histone_crop=imcrop(im_dapi_histone,[x_c-half_w_n,y_c-half_w_n,w_n,w_n]);
%     
%     nucleus_mask=zeros(w_p+1,w_p+1);
%     nucleus_mask(half_w_p-half_w_n:half_w_p+half_w_n,half_w_p-half_w_n:half_w_p+half_w_n)=get_center_cell(dapi_histone_crop);
%     
    %% SOMA MASK %%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%
    soma_plus_nucleus=imdilate(nucleus_mask,strel('disk',2));
    soma_mask=soma_plus_nucleus-nucleus_mask;
     
    %% PROCESSES MASK %%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    cnpase_crop=imcrop(im_cnpase,[x_c-half_w_p, y_c-half_w_p, w_p, w_p]);
    cnpase_crop_bw=imbinarize(cnpase_crop,1.2*graythresh(cnpase_crop));
    
    %region growing
    positive=soma_plus_nucleus;
    negative=~cnpase_crop_bw;

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
    
%     subplot(3,3,1); imshow(imadjust(imcrop(im_dapi_histone,[x_c-half_w_p,y_c-half_w_p,w_p,w_p]))); title('DAPI Histone');
%     %subplot(3,3,2); imshow(imadjust(imcrop(im_olig2,[x_c-half_w_p,y_c-half_w_p,w_p,w_p]))); title('Olig2');
%     subplot(3,3,3); imshow(imadjust(cnpase_crop)); title('CNPase');
%     subplot(3,3,4); imshow(nucleus_mask); title('Nucleus Mask');
%     subplot(3,3,5); imshow(soma_mask); title('SOMA Mask');
%     subplot(3,3,6); imshow(processes); title('Processes Mask');
%     subplot(3,3,7); imshow(membrane); title('Membrane Mask');
%     subplot(3,3,8); imshow(cytoplasm); title('Cytoplasm Mask');
%     subplot(3,3,9); imshow(whole_cell); title('Whole Cell Mask');
%     
end

