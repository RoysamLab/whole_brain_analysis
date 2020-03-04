function [x1,y1,dist_min] = find_closest_points_dist_cnst(x_max,y_max,x_pt,y_pt)
%find the closest points from region and constant point
dist_matrix=zeros(length(x_max),1);

parfor i=1:length(x_max)
    dist=(  (x_max(i)-x_pt).^2 + (y_max(i)-y_pt).^2  ).^0.5;
    dist_matrix(i,1)=dist;
end

[row,col]=find(dist_matrix==min(min(dist_matrix)));

x1=x_max(row);
y1=y_max(row);

dist_min=min(min(dist_matrix));

end

