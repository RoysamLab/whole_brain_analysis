function labels =grow_from_seeds(labels)
% it connects the fg to connected undefined regions, ignores the isolated
% regions
% labels has values: -1 (bg), 1 (fg), or 0 (undef.)

% resulting labels will be either 0 (bg) or 1 (fg)
%  resulting strengths will be between 0 and 1
%find all 1, and check all neighbors
%if 0, make it 1

[r_1,c_1]=find(labels==1);

fg_points = [r_1,c_1];
i=1;
[W,H]=size(labels);

%Start with white pixels
while i<=length(fg_points)
    x_curr=fg_points(i,1);
    y_curr=fg_points(i,2);
    %check for intensity at(x_curr,y_curr-1),(x_curr,y_curr+1),(x_curr+1,y_curr),(x_curr-1,y_curr)
    %check if these points exist
    if x_curr>=2
        val_left=labels(x_curr-1,y_curr);
        if val_left==0
            fg_points=[fg_points;x_curr-1,y_curr];
            labels(x_curr-1,y_curr)=1;
        end
    end
    if y_curr>=2
        val_top=labels(x_curr,y_curr-1);
        if val_top==0
            fg_points=[fg_points;x_curr,y_curr-1];
            labels(x_curr,y_curr-1)=1;
        end
    end
    if x_curr<=W-1
        val_right=labels(x_curr+1,y_curr);
        if val_right==0
            fg_points=[fg_points;x_curr+1,y_curr];
            labels(x_curr+1,y_curr)=1;
        end
    end
    if y_curr<=H-1
        val_down=labels(x_curr,y_curr+1);
        if val_down==0
            fg_points=[fg_points;x_curr,y_curr+1];
            labels(x_curr,y_curr+1)=1;
        end
    end
    i=i+1;
end

[r_neg1,c_neg1]=find(labels==-1);

for i=1:length(r_neg1)
    labels(r_neg1(i),c_neg1(i))=0;
end
end

