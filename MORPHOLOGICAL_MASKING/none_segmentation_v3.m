function [soma_mask, processes, cytoplasm, membrane, whole_cell] = none_segmentation_v3(nucleus_mask,~,w_p)
    
%     half_w_n=round(w_n/2);
%     half_w_p=round(w_p/2);
    
    %% NUCLEUS MASK %%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%
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
    
    processes=zeros(w_p+1,w_p+1);
    processes_plus_mask=imbinarize(soma_plus_nucleus + processes,0.005);
    
    membrane=imbinarize(imdilate(processes_plus_mask,strel('disk',1))-processes_plus_mask,0.005);
    
    whole_cell=imbinarize(soma_plus_nucleus+processes+membrane,0.005);
    
    cytoplasm=imbinarize(whole_cell-nucleus_mask-membrane,0.005);
    
%     subplot(1,7,1); imshow(imadjust(imcrop(im_dapi_histone,[x_c-half_w_p,y_c-half_w_p,w_p,w_p]))); title('DAPI Histone');
%     subplot(1,7,2); imshow(nucleus_mask); title('Nucleus Mask');
%     subplot(1,7,3); imshow(soma_mask); title('SOMA Mask');
%     subplot(1,7,4); imshow(processes); title('Processes Mask');
%     subplot(1,7,5); imshow(membrane); title('Membrane Mask');
%     subplot(1,7,6); imshow(cytoplasm); title('Cytoplasm Mask');
%     subplot(1,7,7); imshow(whole_cell); title('Whole Cell Mask');

    
end

