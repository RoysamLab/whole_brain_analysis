function d_js = get_directional_ratio_vals_v2(img,p_x,p_y)
c1=1;
c2=1;
masks=cell(5,37);
k=0.5;% length/width ratio

for j=3:0.5:5
    c2=1;
    for l=0:5:180
        theta=deg2rad(l);
        R= [[cos(theta), -sin(theta)];[ sin(theta), cos(theta)]];
        L=2^j;
        H=k*2^j;
        
        X=([-L/2, L/2, L/2, -L/2]);
        Y=([-H/2, -H/2, H/2, H/2]);
        T=R*[X;Y];
        
        x_lower_left=p_x+T(1,1);
        x_lower_right=p_x+T(1,2);
        x_upper_right=p_x+T(1,3);
        x_upper_left=p_x+T(1,4);
        y_lower_left=p_y+T(2,1);
        y_lower_right=p_y+T(2,2);
        y_upper_right=p_y+T(2,3);
        y_upper_left=p_y+T(2,4);
        x_coor=[x_lower_left x_lower_right x_upper_right x_upper_left];
        y_coor=[y_lower_left y_lower_right y_upper_right y_upper_left];

        S=zeros(size(img));
        S_rgb = insertShape(S,'FilledPolygon',[x_lower_left,y_lower_left,x_lower_right,y_lower_right,x_upper_right,y_upper_right,x_upper_left,y_upper_left]);
        
        S=im2bw(rgb2gray(S_rgb),0.005);
        %imshow(S);
        masks{c1,c2}=S(:,:);
        c2=c2+1;
    end
    c1=c1+1;
end

%masks saved
d_js=[];
parfor c1=1:5
    val_ls=[];
    for c2=1:37
        val=sum(sum(masks{c1,c2}.*img));
        val_ls=[val_ls;val];
    end
    d_js=[d_js;min(val_ls)/max(val_ls)];
end
end

