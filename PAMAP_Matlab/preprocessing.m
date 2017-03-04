%% preprocessing
% normalization
% We normalize each dimension of 3D time series as (x−μ)/σ , where μ and σ are mean and standard deviation of time series.
stardim=3;
for i=1:size(subject_data,2)
    data=subject_data{1,i};
    for j=stardim:size(data,2)
        data(:,j)=fixgaps(data(:,j)); % remove NaNs by interpolation
    end
    data(:,stardim:end)=normalize2d(data(:,stardim:end));
    subject_data{1,i}=data;
end

for i=1:size(subject_data,2) % loop for all subjects
    subject_data{1,i}(subject_data{1,i}(:,2)==3,2)=0;
    subject_data{1,i}(subject_data{1,i}(:,2)==4,2)=1;
    subject_data{1,i}(subject_data{1,i}(:,2)==12,2)=2;
    subject_data{1,i}(subject_data{1,i}(:,2)==13,2)=3;
end

% slide window
% Then we apply the sliding window algorithm to extract subsequences from 3D time series with different sliding steps.
wl=256;
step=128;
dimlabel=2;
input_data=[];
for i=1:size(subject_data,2) % loop for all subjects
    data=subject_data{1,i};
    for dim=3:11 % loop for all data dimension
        result=[];
        head=1;
        while head-1+wl <= length(data)
            tail=head-1+wl;
            if length(unique(data(head:tail,dimlabel)))~=1 % data(head:tail) do not share a unique activityID
                head= head-1+find(data(head:tail,dimlabel)==data(tail,dimlabel),1);
                continue
            else
                feature = data(head:tail,dim)';
                label=data(tail,dimlabel);
                result = [result; [label,feature]];
                head=head+step;
            end
        end
        input_data{i}{dim-2}=result;
    end
end

% 
