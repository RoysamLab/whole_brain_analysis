 function [soma_mask, processes, cytoplasm, membrane, whole_cell] = endothelial_segmentation_v3(im_gfp,im_reca1,nucleus_mask, x_c,y_c,w_n,w_s,w_p)
    %%Needs fix
    half_w_n=round(w_n/2);
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
    %get GFP crop
    gfp_crop=imcrop(im_gfp,[x_c-half_w_s,y_c-half_w_s,w_s,w_s]);
    gfp_crop_bw=imbinarize(gfp_crop,graythresh(gfp_crop));
    gfp_crop_bw=imdilate(imerode(gfp_crop_bw,strel('disk',2)),strel('disk',2));
    
    gfp_props=regionprops(gfp_crop_bw,'all');
    gfp_cs=vertcat(gfp_props.Centroid);
    gfp_cx=gfp_cs(:,1);
    gfp_cy=gfp_cs(:,2);
    
    [centroid_x, ~] = find_closest_points_cnst(gfp_cx,gfp_cy,half_w_s,half_w_s);
    if length(centroid_x)>1 %pick larger area
        gfp_idx=find(gfp_cx==centroid_x(1));
        area_max=gfp_props(gfp_idx).Area;
        for j=1:length(centroid_x)
            area_curr=gfp_props(j).Area;
            if area_curr>area_max
                area_max=gfp_props(j).Area;
                gfp_idx=j;
            end
        end
    else
        gfp_idx=find(gfp_cx==centroid_x);
    end
    
        
    gfp_BB=gfp_props(gfp_idx).BoundingBox;

    gfp_cell_bw=zeros(size(gfp_crop_bw));
    %center it
    gfp_cell_bw(ceil(gfp_BB(2)):ceil(gfp_BB(2))+gfp_BB(4)-1,ceil(gfp_BB(1)):ceil(gfp_BB(1))+gfp_BB(3)-1)=gfp_props(gfp_idx).Image;
    %fill any possible holes
    gfp_cell_bw=imfill(gfp_cell_bw,'holes');
    
    gfp_cell_full=zeros(w_p+1,w_p+1);
    gfp_cell_full(half_w_p-half_w_s:half_w_p+half_w_s,half_w_p-half_w_s:half_w_p+half_w_s)=gfp_cell_bw;
    
    soma_plus_nucleus=imbinarize(gfp_cell_full+nucleus_mask,0.01);
    soma_plus_nucleus=imdilate(nucleus_mask,strel('disk',2)).*soma_plus_nucleus;
    
    soma_mask=imbinarize(soma_plus_nucleus-nucleus_mask,0.005);
    
    %% PROCESSES MASK %%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    reca1_crop=imcrop(im_reca1,[x_c-half_w_p, y_c-half_w_p, w_p, w_p]);
    gfp_crop=imcrop(im_gfp,[x_c-half_w_p, y_c-half_w_p, w_p, w_p]);

    %Get cytoplasm
    gfp_crop_bw=imbinarize(gfp_crop,graythresh(gfp_crop));
    %remove small objects
    gfp_crop_bw = bwareafilt(gfp_crop_bw,[50,40401]);
    gfp_props=regionprops(gfp_crop_bw,'all');
    
    gfp_cs=vertcat(gfp_props.Centroid);
    if length(gfp_cs)>1
        gfp_cx=gfp_cs(:,1);
        gfp_cy=gfp_cs(:,2);

        [centroid_x, ~] = find_closest_points_cnst(gfp_cx,gfp_cy,half_w_p,half_w_p);

         if length(centroid_x)>1 %pick larger area
            gfp_idx=find(gfp_cx==centroid_x(1));
            area_max=gfp_props(gfp_idx).Area;
            for j=1:length(centroid_x)
                area_curr=gfp_props(j).Area;
                if area_curr>area_max
                    area_max=gfp_props(j).Area;
                    gfp_idx=j;
                end
            end
        else
            gfp_idx=find(gfp_cx==centroid_x);
         end

        gfp_BB=gfp_props(gfp_idx).BoundingBox;

        gfp_cell_bw=zeros(size(gfp_crop_bw));
        %center it
        gfp_cell_bw(ceil(gfp_BB(2)):ceil(gfp_BB(2))+gfp_BB(4)-1,ceil(gfp_BB(1)):ceil(gfp_BB(1))+gfp_BB(3)-1)=gfp_props(gfp_idx).Image;
        %fill any possible holes
        gfp_cell_bw=imfill(gfp_cell_bw,'holes');

        cytoplasm=gfp_cell_bw;
    else
        cytoplasm=zeros(size(gfp_crop));
    end
    %Get membrane
    reca1_crop_bw=imbinarize(imadjust(reca1_crop),graythresh(imadjust(reca1_crop)));
    reca1_crop_bw = bwareafilt(reca1_crop_bw,[50,40401]);
    
    reca1_props=regionprops(reca1_crop_bw,'all');
    
    reca1_cs=vertcat(reca1_props.Centroid);
    reca1_cx=reca1_cs(:,1);
    reca1_cy=reca1_cs(:,2);
    
    [centroid_x, ~] = find_closest_points_cnst(reca1_cx,reca1_cy,half_w_p,half_w_p);

    if length(centroid_x)>1 %pick larger area
        reca1_idx=find(reca1_cx==centroid_x(1));
        area_max=reca1_props(reca1_idx).Area;
        for j=1:length(centroid_x)
            area_curr=reca1_props(j).Area;
            if area_curr>area_max
                area_max=reca1_props(j).Area;
                reca1_idx=j;
            end
        end
    else
        reca1_idx=find(reca1_cx==centroid_x);
     end
    
    reca1_BB=reca1_props(reca1_idx).BoundingBox;

    reca1_cell_bw=zeros(size(reca1_crop_bw));
    %center it
    reca1_cell_bw(ceil(reca1_BB(2)):ceil(reca1_BB(2))+reca1_BB(4)-1,ceil(reca1_BB(1)):ceil(reca1_BB(1))+reca1_BB(3)-1)=reca1_props(reca1_idx).Image;
    %fill any possible holes
    reca1_cell_bw=imfill(reca1_cell_bw,'holes');
    
    membrane=edge(reca1_cell_bw);
    
    
    %% WHOLE CELL
  
    processes=zeros(size(cytoplasm));
    
    whole_cell=imfill(soma_plus_nucleus+cytoplasm+membrane,'holes');
    
%     subplot(2,4,1); imshow(imadjust(imcrop(im_gfp,[x_c-half_w_p, y_c-half_w_p, w_p, w_p])));
%     subplot(2,4,2); imshow(imadjust(imcrop(im_gfp,[x_c-half_w_p, y_c-half_w_p, w_p, w_p])));
% %    
%     subplot(2,4,3); imshow(nucleus_mask); title('Nucleus');
%     subplot(2,4,4); imshow(soma_mask); title('SOMA');
%     subplot(2,4,5); imshow(cytoplasm); title('Cytoplasm');
%     subplot(2,4,6); imshow(membrane); title('Membrane');
%     subplot(2,4,7); imshow(whole_cell); title('Whole cell');
%     
%    figure;
%    imshow(imfuse(imadjust(imcrop(im_dapi_histone,[x_c-half_w_p, y_c-half_w_p, w_p, w_p])),imadjust(imcrop(im_gfp,[x_c-half_w_p, y_c-half_w_p, w_p, w_p]))));
%    
%    
%    figure;
%    imshow(imfuse(imadjust(imcrop(im_dapi_histone,[x_c-half_w_p, y_c-half_w_p, w_p, w_p])),imadjust(imcrop(im_reca1,[x_c-half_w_p, y_c-half_w_p, w_p, w_p]))));
end