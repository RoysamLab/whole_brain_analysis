function img_cell_bw = get_soma_v2(img,W,H)

    skel_pts=bwskel(img);
%     W=50; 
%     H=50;

    [y,x]=find(skel_pts==1);

    parfor i=1:length(x)
        vals(:,i)=get_directional_ratio_vals_v2(img,x(i),y(i));
    end

    %segregate points as per these values

    pts_x=[];
    pts_y=[];

%     imshow(img);
%     hold on;
%     for i=1:length(x)
%         mean_val=mean(vals(:,i));
%         if mean_val>0.70
%             plot(x(i),y(i), 'r+', 'MarkerSize', 5, 'LineWidth', 2);
%         else
%             if mean_val<0.55
%                 plot(x(i),y(i), 'b+', 'MarkerSize', 5, 'LineWidth', 2);
%                 pts_x=[pts_x;x(i)];
%                 pts_y=[pts_y;y(i)];
%             end
%         end
%     end

    S=zeros(size(img));
    for i=1:length(x)
        mean_val=mean(vals(:,i));
        if mean_val<0.55
            S(y(i),x(i))=1;
        end
    end

    S1=imdilate(S,strel('disk',2));
    S1_cc=bwconncomp(S1);
    S1_comps=regionprops(S1_cc,'all');

%     subplot(1,2,1); imshow(S);
%     subplot(1,2,2); imshow(S1);

    for i=1:length(S1_comps)
        processes_pixels=S1_comps(i).PixelList;
        %[closest_x,closest_y]=find_closest_points_dist_cnst(processes_pixels(:,1),processes_pixels(:,2),50,50);
        pts_x=processes_pixels(:,1);
        pts_y=processes_pixels(:,2);
        line_params=polyfit(pts_x,pts_y,1);
        m=line_params(1);
        c=line_params(2);

        parll_m=-1/m;

        %closest point to 50 x 50
        [x1,y1]=find_closest_points_dist_cnst(pts_x,pts_y,W,H);
        if length(y1)>1 && length(x1)>1
            x1=x1(1);
            y1=y1(1);
        end
        %cutting along points on the line: Y = y1 + parll_m*(X - x1)
        for x_pt=2:size(img,1)-1
            y_pt = round(y1 + parll_m*(x_pt-x1));
            if y_pt>=2 && y_pt<=size(img,2)-1
                img(y_pt,x_pt)=0;
                img(y_pt,x_pt+1)=0;
                img(y_pt,x_pt-1)=0;

                img(y_pt+1,x_pt)=0;
                img(y_pt+1,x_pt-1)=0;
                img(y_pt+1,x_pt+1)=0;

                img(y_pt-1,x_pt)=0;
                img(y_pt-1,x_pt+1)=0;
                img(y_pt-1,x_pt-1)=0;

            end
        end
    end

    img_props=regionprops(img,'all');
    img_cs=vertcat(img_props.Centroid);
    img_cx=img_cs(:,1);
    img_cy=img_cs(:,2);

    [centroid_x, centroid_y] = find_closest_points_cnst(img_cx,img_cy,W,H);
    img_idx=find(img_cx==centroid_x);

    img_BB=img_props(img_idx).BoundingBox;

    img_cell_bw=zeros(size(img));
    %center it
    img_cell_bw(ceil(img_BB(2))+1:ceil(img_BB(2))+img_BB(4),ceil(img_BB(1))+1:ceil(img_BB(1))+img_BB(3))=img_props(img_idx).Image;

end
